"""SOC Dashboard statik sunumu ve arka plan simülatör tetikleyicisi entegrasyon testleri."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine

from config.settings import FeatureSettings, GeneratorSettings, ModelSettings, Settings
from src.api.main import create_app
from src.api.scoring_service import ScoringService
from src.database.db import create_db_engine, init_db, persist_dataset
from src.features.pipeline import FeatureExtractor
from src.generator.synthetic_data import SyntheticDataGenerator, SyntheticDataset
from src.models.scorer import RiskScorer
from src.models.train import run_training, save_artifacts

GEN_CFG = GeneratorSettings(
    seed=41,
    n_customers=420,
    n_merchants=35,
    simulation_days=21,
    n_fraud_rings=4,
    n_account_takeovers=10,
    n_mule_networks=5,
    n_card_testing=5,
)
MODEL_CFG = ModelSettings(
    train_days=13,
    val_days=3,
    n_search_trials=2,
    num_boost_round=320,
    early_stopping_rounds=25,
)

RAW_IBAN_RE = re.compile(r"\bTR\d{24}\b")


@dataclass
class DashboardTestEnv:
    dataset: SyntheticDataset
    engine: Engine
    service: ScoringService
    client: TestClient


@pytest.fixture(scope="module")
def dashboard_env(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("aegis_dashboard")
    artifacts_dir = root / "artifacts"
    db_path = root / "test_dashboard.db"

    ds = SyntheticDataGenerator(GEN_CFG).generate()
    engine = create_db_engine(f"sqlite:///{db_path.as_posix()}")
    init_db(engine)
    persist_dataset(ds, engine)

    extractor = FeatureExtractor(ds.accounts, ds.devices, FeatureSettings())
    features = extractor.fit_transform(ds.transactions)

    settings = Settings()
    result = run_training(
        features,
        ds.transactions,
        start=GEN_CFG.start_date,
        cfg=MODEL_CFG,
        alert_threshold=settings.FRAUD_THRESHOLD,
    )
    save_artifacts(result, artifacts_dir)
    extractor.save(artifacts_dir / "feature_extractor.joblib")

    scorer = RiskScorer.from_artifacts(artifacts_dir, settings)
    loaded_extractor = FeatureExtractor.load(artifacts_dir / "feature_extractor.joblib")
    service = ScoringService(scorer=scorer, extractor=loaded_extractor, engine=engine, settings=settings)
    app = create_app(service=service, settings=settings)

    try:
        with TestClient(app) as client:
            yield DashboardTestEnv(
                dataset=ds,
                engine=engine,
                service=service,
                client=client,
            )
    finally:
        engine.dispose()


def test_dashboard_root_and_static_assets(dashboard_env: DashboardTestEnv) -> None:
    """`GET /`, `GET /dashboard` ve `/static/*` dosyalarının eksiksiz servis edildiğini doğrular."""
    for route in ("/", "/dashboard"):
        resp = dashboard_env.client.get(route)
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        html = resp.text
        assert "AegisGraph SOC" in html
        assert "cytoscape" in html.lower()
        assert 'data-scenario="FRAUD_RING"' in html
        assert 'data-scenario="ACCOUNT_TAKEOVER"' in html
        assert 'data-scenario="CARD_TESTING"' in html
        assert 'data-scenario="NORMAL"' in html
        assert "/static/styles.css" in html
        assert "/static/app.js" in html

    css_resp = dashboard_env.client.get("/static/styles.css")
    assert css_resp.status_code == 200
    assert "text/css" in css_resp.headers["content-type"]
    assert ".soc-grid" in css_resp.text

    js_resp = dashboard_env.client.get("/static/app.js")
    assert js_resp.status_code == 200
    assert "javascript" in js_resp.headers["content-type"]
    assert "EventSource" in js_resp.text
    assert "/simulator/run" in js_resp.text


@pytest.mark.parametrize(
    ("scenario", "count"),
    [
        ("NORMAL", 4),
        ("FRAUD_RING", 6),
        ("ACCOUNT_TAKEOVER", 3),
        ("CARD_TESTING", 6),
        ("ALL", 6),
    ],
)
def test_simulator_run_valid_scenarios_and_sse_events(
    dashboard_env: DashboardTestEnv,
    scenario: str,
    count: int,
) -> None:
    """`POST /simulator/run` tüm senaryoları arka planda çalıştırıp SSE akışına olay itmelidir."""
    before_processed = dashboard_env.service.extractor.n_processed

    resp = dashboard_env.client.post(
        "/simulator/run",
        json={"scenario": scenario, "count": count, "tps": 400.0},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "started"
    assert body["scenario"] == scenario
    assert body["count"] == count

    assert dashboard_env.service.extractor.n_processed == before_processed + count

    stream_resp = dashboard_env.client.get("/stream/events", params={"replay_recent": 30})
    assert stream_resp.status_code == 200
    assert "event: transaction" in stream_resp.text
    assert "sender_masked_iban" in stream_resp.text
    assert not RAW_IBAN_RE.search(stream_resp.text)

    if scenario == "FRAUD_RING":
        assert "event: alert" in stream_resp.text


def test_simulator_run_rejects_invalid_parameters(dashboard_env: DashboardTestEnv) -> None:
    """Geçersiz senaryo adı, sınır dışı işlem sayısı ve ekstra alanlar 422 ile reddedilmelidir."""
    bad_scenario = dashboard_env.client.post(
        "/simulator/run",
        json={"scenario": "UNKNOWN_SCENARIO", "count": 10},
    )
    assert bad_scenario.status_code == 422

    bad_count_low = dashboard_env.client.post(
        "/simulator/run",
        json={"scenario": "NORMAL", "count": 0},
    )
    assert bad_count_low.status_code == 422

    bad_count_high = dashboard_env.client.post(
        "/simulator/run",
        json={"scenario": "NORMAL", "count": 500},
    )
    assert bad_count_high.status_code == 422

    bad_extra = dashboard_env.client.post(
        "/simulator/run",
        json={"scenario": "NORMAL", "count": 5, "is_fraud": True},
    )
    assert bad_extra.status_code == 422
