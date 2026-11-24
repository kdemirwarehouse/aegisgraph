"""Otonom soruşturma ajanı, kanıt araçları, IBAN maskeleme ve API/SSE entegrasyon testleri."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import Engine, func, inspect, select

from config.settings import AgentSettings, FeatureSettings, GeneratorSettings, ModelSettings, Settings
from src.agent.investigator import FraudInvestigatorAgent, InvestigationReport
from src.agent.tools import (
    get_account_profile,
    get_alert_evidence,
    get_network_connections,
    get_recent_transactions,
)
from src.api.main import create_app
from src.api.scoring_service import ScoringService
from src.compat import UTC
from src.database.db import create_db_engine, init_db, persist_dataset, session_scope
from src.database.models import (
    GROUND_TRUTH_COLUMNS,
    AccountType,
    Alert,
    AlertStatus,
    Base,
    Channel,
    FraudScenario,
    Investigation,
    InvestigationRecord,
    RecommendedAction,
    SuspectedTypology,
    TransactionType,
    mask_holder_name,
    record_fields,
    redact_ibans,
)
from src.features.pipeline import FeatureExtractor
from src.generator.synthetic_data import SyntheticDataGenerator, SyntheticDataset
from src.models.scorer import RiskScorer
from src.models.train import run_training, save_artifacts

GEN_CFG = GeneratorSettings(
    seed=37,
    n_customers=450,
    n_merchants=40,
    simulation_days=21,
    n_fraud_rings=5,
    n_account_takeovers=12,
    n_mule_networks=6,
    n_card_testing=6,
)
MODEL_CFG = ModelSettings(
    train_days=13,
    val_days=3,
    n_search_trials=2,
    num_boost_round=350,
    early_stopping_rounds=25,
)

RAW_IBAN_RE = re.compile(r"\bTR\d{24}\b")
MASKED_IBAN_RE = re.compile(r"^TR\*\* \*{4} \*{4} \*{4} \*{4} \*\*\d{2} \d{2}$")
ALL_GROUND_TRUTH = {c for cols in GROUND_TRUTH_COLUMNS.values() for c in cols}


@dataclass
class AgentTestEnv:
    dataset: SyntheticDataset
    engine: Engine
    extractor: FeatureExtractor
    scorer: RiskScorer
    settings: Settings
    service: ScoringService
    client: TestClient
    scenario_alerts: dict[str, int]


def _collect_keys(obj: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(str(k))
            keys.update(_collect_keys(v))
    elif isinstance(obj, list):
        for item in obj:
            keys.update(_collect_keys(item))
    return keys


def _pick_scenario_tx_ids(ds: SyntheticDataset) -> dict[str, str]:
    tx = ds.transactions.sort_values(["timestamp", "transaction_id"], kind="stable")
    acc_fraudsters = set(ds.accounts.loc[ds.accounts["is_fraudster"], "account_id"])

    # FRAUD_RING: her iki ucu da halka üyesi olan son layering işlemi (döngü kapanmış)
    ring_txs = tx[
        (tx["fraud_scenario"] == FraudScenario.FRAUD_RING.value)
        & tx["sender_account_id"].isin(acc_fraudsters)
        & tx["receiver_account_id"].isin(acc_fraudsters)
    ]
    # MONEY_MULE: hub çıkışındaki (fan-out) son işlem → hem fan-in hem fan-out oranı görünür
    mule_txs = tx[
        (tx["fraud_scenario"] == FraudScenario.MONEY_MULE.value)
        & tx["sender_account_id"].isin(acc_fraudsters)
    ]
    # ACCOUNT_TAKEOVER: yüksek tutarlı tipik ATO işlemi
    ato_txs = tx[tx["fraud_scenario"] == FraudScenario.ACCOUNT_TAKEOVER.value]
    # CARD_TESTING: mikro denemeler sonrası büyük alışveriş
    ct_txs = tx[tx["fraud_scenario"] == FraudScenario.CARD_TESTING.value].sort_values(
        ["amount_minor", "timestamp"], ascending=[False, False]
    )

    return {
        FraudScenario.FRAUD_RING.value: str(ring_txs.iloc[-1]["transaction_id"]),
        FraudScenario.MONEY_MULE.value: str(mule_txs.iloc[-1]["transaction_id"]),
        FraudScenario.ACCOUNT_TAKEOVER.value: str(ato_txs.iloc[-1]["transaction_id"]),
        FraudScenario.CARD_TESTING.value: str(ct_txs.iloc[0]["transaction_id"]),
    }


@pytest.fixture(scope="module")
def agent_env(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("aegis_agent")
    artifacts_dir = root / "artifacts"
    db_path = root / "test_agent.db"

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

    # Her 4 senaryo için veritabanında birer test alarmı hazırla
    picked = _pick_scenario_tx_ids(ds)
    scenario_alerts: dict[str, int] = {}
    with session_scope(engine) as session:
        for scenario, tx_id in picked.items():
            row_feats = features.loc[tx_id].to_dict()
            score_res = scorer.score_features(row_feats, transaction_id=tx_id, explain=True)
            exp = score_res.explanation
            assert exp is not None
            alert = Alert(
                transaction_id=tx_id,
                risk_score=max(round(score_res.risk_score, 6), 0.88),
                ml_score=round(score_res.risk_score, 6),
                graph_score=round(exp.graph_contribution_share, 4),
                status=AlertStatus.OPEN,
                reasons=[f.__dict__ for f in exp.top_factors],
            )
            session.add(alert)
            session.flush()
            scenario_alerts[scenario] = int(alert.alert_id)

    service = ScoringService(scorer=scorer, extractor=loaded_extractor, engine=engine, settings=settings)
    app = create_app(service=service, settings=settings)

    try:
        with TestClient(app) as client:
            yield AgentTestEnv(
                dataset=ds,
                engine=engine,
                extractor=loaded_extractor,
                scorer=scorer,
                settings=settings,
                service=service,
                client=client,
                scenario_alerts=scenario_alerts,
            )
    finally:
        engine.dispose()


def test_tools_mask_iban_and_never_leak_ground_truth(agent_env: AgentTestEnv) -> None:
    """Tüm soruşturma araçları ham IBAN ve ground-truth etiketlerini dışlamalıdır."""
    alert_id = agent_env.scenario_alerts[FraudScenario.FRAUD_RING.value]
    ev = get_alert_evidence(alert_id, engine=agent_env.engine)
    sender = ev["transaction"]["sender_account_id"]
    ref_time = datetime.fromisoformat(ev["transaction"]["timestamp"])

    profile = get_account_profile(sender, engine=agent_env.engine, reference_time=ref_time)
    recent = get_recent_transactions(sender, limit=10, engine=agent_env.engine, reference_time=ref_time)
    network = get_network_connections(
        sender,
        engine=agent_env.engine,
        extractor=agent_env.extractor,
        reference_time=ref_time,
    )

    assert MASKED_IBAN_RE.match(ev["transaction"]["sender_masked_iban"])
    assert MASKED_IBAN_RE.match(ev["transaction"]["receiver_masked_iban"])
    assert MASKED_IBAN_RE.match(profile["masked_iban"])
    assert recent["transactions"], "Son işlem listesi boş olmamalı"
    for tx in recent["transactions"]:
        assert MASKED_IBAN_RE.match(tx["counterparty_masked_iban"])

    # KVKK: ham hesap sahibi adı dönmez, yalnızca maskeli hali döner
    raw_holder = agent_env.dataset.accounts.set_index("account_id").loc[sender, "holder_name"]
    assert "holder_name" not in profile
    assert profile["holder_name_masked"] == mask_holder_name(raw_holder)
    assert raw_holder not in json.dumps(profile, ensure_ascii=False, default=str)

    # Point-in-time: referans andan önceki profil, sonraki alarmları görmez
    early_profile = get_account_profile(
        sender, engine=agent_env.engine, reference_time=ref_time - timedelta(seconds=1)
    )
    assert alert_id not in early_profile["alert_ids"]
    assert alert_id in profile["alert_ids"]
    assert len(profile["alert_ids"]) <= agent_env.settings.agent.max_evidence_ids

    for tool_output in (ev, profile, recent, network):
        serialized = json.dumps(tool_output, ensure_ascii=False, default=str)
        assert not RAW_IBAN_RE.search(serialized), "Çıktıda maskelenmemiş ham IBAN sızıntısı var"
        leaked_keys = ALL_GROUND_TRUTH & _collect_keys(tool_output)
        assert not leaked_keys, f"Ground-truth alanları araç çıktısına sızdı: {leaked_keys}"


def test_human_in_the_loop_schema_enforces_approval() -> None:
    """Human-in-the-Loop ilkesi gereği `requires_human_approval=False` reddedilmelidir."""
    with pytest.raises(ValidationError):
        InvestigationRecord(
            alert_id=1,
            tx_id="TX-00000001",
            created_at=datetime(2026, 10, 15, 12, 0),
            suspected_typology=SuspectedTypology.ACCOUNT_TAKEOVER,
            confidence_score=0.92,
            executive_summary_tr="Test özeti",
            evidence_chain=["Kanıt 1"],
            related_accounts=["ACC-000001"],
            recommended_action=RecommendedAction.BLOCK_ACCOUNT,
            requires_human_approval=False,
        )

    with pytest.raises(ValidationError):
        InvestigationReport(
            alert_id=1,
            tx_id="TX-00000001",
            created_at=datetime(2026, 10, 15, 12, 0),
            suspected_typology=SuspectedTypology.ACCOUNT_TAKEOVER,
            confidence_score=0.92,
            executive_summary_tr="Test özeti",
            evidence_chain=["Kanıt 1"],
            related_accounts=["ACC-000001"],
            recommended_action=RecommendedAction.BLOCK_ACCOUNT,
            requires_human_approval=False,
        )


def test_agent_settings_single_source_and_provider_validation() -> None:
    """AgentSettings `Settings.agent` üzerinden gelir; LLM modu API anahtarı olmadan reddedilir."""
    settings = Settings()
    assert isinstance(settings.agent, AgentSettings)
    assert settings.agent.provider == "local"
    assert 0.0 <= settings.agent.auto_investigate_min_score <= 1.0

    with pytest.raises(ValidationError):
        AgentSettings(provider="openai")
    with pytest.raises(ValidationError):
        AgentSettings(auto_investigate_min_score=1.5)

    llm = AgentSettings(provider="openai", api_key="sk-test")
    assert "sk-test" not in repr(llm), "API anahtarı repr/log çıktısında görünmemeli"


def test_investigations_schema_matches_record_and_is_additive(tmp_path) -> None:
    """ORM kolonları Pydantic sözleşmesiyle birebir aynıdır; `create_all` mevcut tabloları bozmaz."""
    orm_cols = [c.name for c in Investigation.__table__.columns]
    assert orm_cols == record_fields("investigations")

    engine = create_db_engine(f"sqlite:///{(tmp_path / 'legacy.db').as_posix()}")
    try:
        # Soruşturma tablosu henüz yokken oluşturulmuş eski bir veritabanını simüle et
        legacy_tables = [t for name, t in Base.metadata.tables.items() if name != "investigations"]
        Base.metadata.create_all(engine, tables=legacy_tables)
        with engine.begin() as conn:
            conn.exec_driver_sql(
                "INSERT INTO accounts (account_id, iban, holder_name, account_type, kyc_level, city, "
                "merchant_category, opened_at, is_fraudster, fraud_case_id) VALUES "
                "('ACC-000001', 'TR330006100519786457841326', 'Test', 'PERSONAL', 2, 'Ankara', NULL, "
                "'2026-01-01 00:00:00', 0, NULL)"
            )
        assert "investigations" not in inspect(engine).get_table_names()

        init_db(engine)
        assert "investigations" in inspect(engine).get_table_names()
        with session_scope(engine) as session:
            assert session.scalar(select(func.count()).select_from(Base.metadata.tables["accounts"])) == 1
    finally:
        engine.dispose()


def test_redaction_helpers_mask_raw_iban_and_holder_name() -> None:
    """`redact_ibans` iç içe yapılardaki ham IBAN'ları, `mask_holder_name` isimleri maskeler."""
    payload = {"note": "Alıcı TR33 0006 1005 1978 6457 8413 26 hesabı", "items": ["TR330006100519786457841326"]}
    redacted = json.dumps(redact_ibans(payload), ensure_ascii=False)
    assert not RAW_IBAN_RE.search(redacted)
    assert "TR33 0006" not in redacted
    assert "**13 26" in redacted
    assert mask_holder_name("Ahmet Yılmaz") == "A**** Y*****"
    assert mask_holder_name(None) == "***"


@pytest.mark.parametrize(
    ("scenario", "expected_typology"),
    [
        (FraudScenario.FRAUD_RING.value, SuspectedTypology.FRAUD_RING),
        (FraudScenario.ACCOUNT_TAKEOVER.value, SuspectedTypology.ACCOUNT_TAKEOVER),
        (FraudScenario.MONEY_MULE.value, SuspectedTypology.MONEY_MULE),
        (FraudScenario.CARD_TESTING.value, SuspectedTypology.CARD_TESTING),
    ],
)
def test_agent_investigates_all_fraud_scenarios_and_produces_sar(
    agent_env: AgentTestEnv,
    scenario: str,
    expected_typology: SuspectedTypology,
) -> None:
    """Ajan tüm fraud tipolojileri için doğru sınıflandırma, kanıt zinciri ve Türkçe SAR raporu üretmelidir."""
    alert_id = agent_env.scenario_alerts[scenario]
    agent = FraudInvestigatorAgent(
        engine=agent_env.engine,
        extractor=agent_env.extractor,
        settings=agent_env.settings,
    )
    report = agent.investigate(alert_id, persist=True)

    assert report.investigation_id is not None
    assert report.alert_id == alert_id
    assert report.suspected_typology is expected_typology
    assert 0.5 <= report.confidence_score <= 1.0
    assert report.requires_human_approval is True
    assert len(report.evidence_chain) >= 3
    assert len(report.related_accounts) >= 2
    assert len(report.tool_trace) >= 5
    assert "MASAK" in report.executive_summary_tr
    assert "insan analist onayı" in report.executive_summary_tr
    assert not RAW_IBAN_RE.search(report.executive_summary_tr)

    # DB'de investigations tablosuna kaydedilmeli; alert asla otomatik CONFIRMED_FRAUD olmamalı
    with session_scope(agent_env.engine) as session:
        inv_row = session.get(Investigation, report.investigation_id)
        assert inv_row is not None
        assert inv_row.suspected_typology is expected_typology
        assert inv_row.requires_human_approval is True
        alert_row = session.get(Alert, alert_id)
        assert alert_row is not None
        assert alert_row.status is AlertStatus.INVESTIGATING
        assert alert_row.status is not AlertStatus.CONFIRMED_FRAUD


def test_llm_provider_output_is_redacted_and_falls_back_on_error(
    agent_env: AgentTestEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LLM modu: sağlayıcı ham IBAN döndürse bile maskelenir; hata durumunda yerel SAR özeti korunur."""
    import httpx

    from src.agent import investigator as inv_mod

    llm_settings = agent_env.settings.model_copy(
        update={
            "agent": AgentSettings(
                provider="openai",
                api_key="test-key",
                llm_base_url="https://llm.invalid/v1/",
                llm_timeout_seconds=3.0,
            )
        }
    )
    agent = FraudInvestigatorAgent(engine=agent_env.engine, extractor=agent_env.extractor, settings=llm_settings)
    alert_id = agent_env.scenario_alerts[FraudScenario.ACCOUNT_TAKEOVER.value]
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        calls.append({"url": url, **kwargs})
        body = {"choices": [{"message": {"content": "SAR özeti: alıcı TR330006100519786457841326 hesabı."}}]}
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(inv_mod.httpx, "post", fake_post)
    report = agent.investigate(alert_id, persist=False)
    assert calls and calls[0]["url"] == "https://llm.invalid/v1/chat/completions"
    assert calls[0]["timeout"] == 3.0
    assert not RAW_IBAN_RE.search(report.executive_summary_tr)
    assert "**13 26" in report.executive_summary_tr
    assert "insan analist onayı" in report.executive_summary_tr
    assert report.requires_human_approval is True

    def failing_post(url: str, **kwargs: Any) -> httpx.Response:
        raise httpx.ConnectError("bağlantı yok")

    monkeypatch.setattr(inv_mod.httpx, "post", failing_post)
    fallback = agent.investigate(alert_id, persist=False)
    assert fallback.executive_summary_tr.startswith("MASAK Şüpheli İşlem Bildirimi")


def test_agent_settings_cross_field_validation() -> None:
    """Çelişkili ajan eşikleri reddedilir."""
    with pytest.raises(ValidationError):
        AgentSettings(mule_passthrough_min_ratio=1.2, mule_passthrough_max_ratio=1.0)
    with pytest.raises(ValidationError):
        AgentSettings(card_testing_min_burst=5, card_testing_strong_burst=4)


def test_investigation_api_endpoints_and_auto_sse_trigger(agent_env: AgentTestEnv) -> None:
    """API soruşturma uçlarını ve kritik alarmlarda otomatik SSE `investigation` yayınını doğrular."""
    target_alert_id = agent_env.scenario_alerts[FraudScenario.FRAUD_RING.value]

    # Manuel tetikleme (POST /alerts/{alert_id}/investigate)
    post_inv = agent_env.client.post(f"/alerts/{target_alert_id}/investigate")
    assert post_inv.status_code == 200
    inv_body = post_inv.json()
    assert inv_body["alert_id"] == target_alert_id
    assert inv_body["requires_human_approval"] is True
    assert inv_body["suspected_typology"] == SuspectedTypology.FRAUD_RING.value
    assert not RAW_IBAN_RE.search(json.dumps(inv_body, ensure_ascii=False))

    # Kayıtlı raporu okuma (GET /alerts/{alert_id}/investigation)
    get_inv = agent_env.client.get(f"/alerts/{target_alert_id}/investigation")
    assert get_inv.status_code == 200
    assert get_inv.json()["investigation_id"] == inv_body["investigation_id"]

    # Olmayan alarm için 404
    assert agent_env.client.post("/alerts/9999999/investigate").status_code == 404
    assert agent_env.client.get("/alerts/9999999/investigation").status_code == 404

    # Kritik eşik (>= auto_investigate_min_score) üstü canlı işlem → otomatik soruşturma + SSE olayı
    personal = agent_env.dataset.accounts[
        (agent_env.dataset.accounts["account_type"] == AccountType.PERSONAL.value)
        & (~agent_env.dataset.accounts["is_fraudster"])
    ]["account_id"].tolist()
    emulators = agent_env.dataset.devices[agent_env.dataset.devices["is_emulator"]]["device_id"].tolist()
    attacker_dev = emulators[0] if emulators else agent_env.dataset.devices["device_id"].iloc[0]

    last_ts = agent_env.service.extractor.last_timestamp
    assert last_ts is not None
    live_ts = (datetime.fromtimestamp(last_ts, tz=UTC).replace(tzinfo=None) + timedelta(days=1)).replace(
        hour=2, minute=45, second=0
    )

    before_count = 0
    with session_scope(agent_env.engine) as session:
        before_count = int(session.scalar(select(func.count()).select_from(Investigation)) or 0)

    score_resp = agent_env.client.post(
        "/transactions/score",
        json={
            "timestamp": live_ts.isoformat(),
            "sender_account_id": personal[0],
            "receiver_account_id": "ACC-990777",
            "device_id": attacker_dev,
            "amount_minor": 4_800_000,
            "channel": Channel.INTERNET_BANKING.value,
            "transaction_type": TransactionType.TRANSFER.value,
            "ip_address": "185.199.110.55",
            "geo_city": "Amsterdam",
            "geo_country": "NL",
        },
    )
    assert score_resp.status_code == 200
    scored = score_resp.json()
    assert scored["risk_score"] >= agent_env.settings.agent.auto_investigate_min_score
    assert scored["alert_id"] is not None
    assert scored["investigation_id"] is not None

    with session_scope(agent_env.engine) as session:
        after_count = int(session.scalar(select(func.count()).select_from(Investigation)) or 0)
        assert after_count == before_count + 1

    # Kayıtlı raporu yeni alert üzerinden API'den çek
    auto_inv = agent_env.client.get(f"/alerts/{scored['alert_id']}/investigation")
    assert auto_inv.status_code == 200
    assert auto_inv.json()["investigation_id"] == scored["investigation_id"]
    assert auto_inv.json()["requires_human_approval"] is True

    # SSE akışında `investigation` olayının yayınlandığını doğrula
    stream_resp = agent_env.client.get("/stream/events", params={"replay_recent": 10})
    assert stream_resp.status_code == 200
    assert "event: investigation" in stream_resp.text
    assert f'"investigation_id": {scored["investigation_id"]}' in stream_resp.text
    assert '"trigger": "auto"' in stream_resp.text

    # Ajan hatası skorlamayı düşürmemeli: alarm kaydedilir, `investigation_failed` olayı yayınlanır
    def broken_investigate(*_args: Any, **_kwargs: Any) -> InvestigationReport:
        raise RuntimeError("ajan arızası")

    original = agent_env.service.investigator.investigate
    agent_env.service.investigator.investigate = broken_investigate  # type: ignore[method-assign]
    try:
        fail_resp = agent_env.client.post(
            "/transactions/score",
            json={
                "timestamp": (live_ts + timedelta(minutes=3)).isoformat(),
                "sender_account_id": personal[0],
                "receiver_account_id": "ACC-990778",
                "device_id": attacker_dev,
                "amount_minor": 4_900_000,
                "channel": Channel.INTERNET_BANKING.value,
                "transaction_type": TransactionType.TRANSFER.value,
                "ip_address": "185.199.110.56",
                "geo_city": "Amsterdam",
                "geo_country": "NL",
            },
        )
    finally:
        agent_env.service.investigator.investigate = original  # type: ignore[method-assign]
    assert fail_resp.status_code == 200
    failed = fail_resp.json()
    assert failed["alert_id"] is not None
    assert failed["investigation_id"] is None
    stream_after = agent_env.client.get("/stream/events", params={"replay_recent": 10})
    assert "event: investigation_failed" in stream_after.text
