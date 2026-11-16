"""Gün 3 entegrasyon testleri: FastAPI skorlama, SSE akışı, graf uçları ve trafik simülatörü."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select

from config.settings import FeatureSettings, GeneratorSettings, ModelSettings, Settings
from src.api.main import create_app
from src.api.scoring_service import ScoringService, mask_iban
from src.api.simulator import TrafficSimulator
from src.database.db import create_db_engine, init_db, persist_dataset, session_scope
from src.database.models import AccountType, Alert, AlertStatus, Channel, Transaction, TransactionType
from src.features.pipeline import FeatureExtractor
from src.generator.synthetic_data import SyntheticDataGenerator, SyntheticDataset
from src.models.scorer import RiskScorer
from src.models.train import run_training, save_artifacts

GEN_CFG = GeneratorSettings(
    seed=29,
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


@dataclass
class ApiTestEnv:
    dataset: SyntheticDataset
    engine: Engine
    settings: Settings
    service: ScoringService
    client: TestClient
    base_ts: datetime

    def next_ts(self, minutes: int = 1) -> datetime:
        """Özellik çıkarıcının son zaman damgasının daima ilerisinde UTC naive zaman üretir."""
        last = self.service.extractor.last_timestamp
        anchor = datetime.fromtimestamp(last, tz=UTC).replace(tzinfo=None) if last is not None else self.base_ts
        return anchor + timedelta(minutes=minutes)


@pytest.fixture(scope="module")
def api_env(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("aegis_api")
    artifacts_dir = root / "artifacts"
    db_path = root / "test_api.db"

    ds = SyntheticDataGenerator(GEN_CFG).generate()
    engine = create_db_engine(f"sqlite:///{db_path.as_posix()}")
    init_db(engine)
    persist_dataset(ds, engine)

    feat_cfg = FeatureSettings()
    extractor = FeatureExtractor(ds.accounts, ds.devices, feat_cfg)
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
    base_ts = ds.transactions["timestamp"].max().to_pydatetime() + timedelta(minutes=5)

    try:
        with TestClient(app) as client:
            yield ApiTestEnv(
                dataset=ds,
                engine=engine,
                settings=settings,
                service=service,
                client=client,
                base_ts=base_ts,
            )
    finally:
        engine.dispose()


def _parse_sse_frames(raw: str) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for block in raw.strip().split("\n\n"):
        if not block or block.startswith(":"):
            continue
        frame: dict[str, Any] = {}
        for line in block.splitlines():
            if line.startswith("id: "):
                frame["id"] = line[4:]
            elif line.startswith("event: "):
                frame["event"] = line[7:]
            elif line.startswith("data: "):
                frame["data"] = json.loads(line[6:])
        if frame:
            frames.append(frame)
    return frames


def test_health_and_iban_masking(api_env: ApiTestEnv) -> None:
    resp = api_env.client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["processed_transactions"] == len(api_env.dataset.transactions)
    assert body["alert_threshold"] == api_env.settings.FRAUD_THRESHOLD

    masked = mask_iban("TR123456789012345678901234")
    assert masked == "TR** **** **** **** **** **12 34"


def test_score_transaction_normal_and_fraud_alert(api_env: ApiTestEnv) -> None:
    ds = api_env.dataset
    personal = ds.accounts[
        (ds.accounts["account_type"] == AccountType.PERSONAL.value) & (~ds.accounts["is_fraudster"])
    ]["account_id"].tolist()
    merchants = ds.accounts[ds.accounts["account_type"] == AccountType.MERCHANT.value]["account_id"].tolist()
    sender = personal[0]
    merchant = merchants[0]
    drop_account = "ACC-990099"

    # Kurban hesabın normal saatte küçük POS harcaması → düşük risk
    normal_ts = (api_env.next_ts(minutes=5) + timedelta(days=1)).replace(
        hour=14, minute=15, second=0, microsecond=0
    )
    normal_resp = api_env.client.post(
        "/transactions/score",
        json={
            "timestamp": normal_ts.isoformat(),
            "sender_account_id": sender,
            "receiver_account_id": merchant,
            "device_id": None,
            "amount_minor": 18_500,
            "channel": Channel.POS.value,
            "transaction_type": TransactionType.CARD_PURCHASE.value,
            "ip_address": None,
            "geo_city": "İstanbul",
            "geo_country": "TR",
        },
    )
    assert normal_resp.status_code == 200
    normal_data = normal_resp.json()
    assert 0.0 <= normal_data["risk_score"] <= 1.0
    assert normal_data["risk_score"] < api_env.settings.FRAUD_THRESHOLD
    assert normal_data["is_alert"] is False
    assert normal_data["alert_id"] is None
    assert normal_data["latency_ms"] >= 0.0

    # ATO saldırısı: yeni açılmış düşük KYC alıcı + gece 02:30 + yabancı IP + emülatör cihaz + yüksek tutar
    emulators = ds.devices[ds.devices["is_emulator"]]["device_id"].tolist()
    attacker_dev = emulators[0] if emulators else ds.devices["device_id"].iloc[0]
    fraud_ts = (normal_ts + timedelta(days=1)).replace(hour=2, minute=30, second=0, microsecond=0)

    fraud_resp = api_env.client.post(
        "/transactions/score",
        json={
            "timestamp": fraud_ts.isoformat(),
            "sender_account_id": sender,
            "receiver_account_id": drop_account,
            "device_id": attacker_dev,
            "amount_minor": 4_500_000,
            "channel": Channel.INTERNET_BANKING.value,
            "transaction_type": TransactionType.TRANSFER.value,
            "ip_address": "185.199.110.42",
            "geo_city": "Amsterdam",
            "geo_country": "NL",
        },
    )
    assert fraud_resp.status_code == 200
    fraud_data = fraud_resp.json()
    assert fraud_data["risk_score"] >= api_env.settings.FRAUD_THRESHOLD
    assert fraud_data["is_alert"] is True
    assert fraud_data["decision"] in {"ALERT", "BLOCK"}
    assert fraud_data["alert_id"] is not None
    assert 0.0 <= fraud_data["graph_score"] <= 1.0
    assert 1 <= len(fraud_data["reasons"]) <= 3
    assert all(r["shap_value"] > 0 and r["description"] for r in fraud_data["reasons"])

    with session_scope(api_env.engine) as session:
        alert = session.get(Alert, fraud_data["alert_id"])
        assert alert is not None
        assert alert.transaction_id == fraud_data["transaction_id"]
        assert alert.status is AlertStatus.OPEN
        tx_row = session.get(Transaction, fraud_data["transaction_id"])
        assert tx_row is not None and tx_row.amount_minor == 4_500_000


def test_score_transaction_validation_and_what_if(api_env: ApiTestEnv) -> None:
    personal = api_env.dataset.accounts[
        api_env.dataset.accounts["account_type"] == AccountType.PERSONAL.value
    ]["account_id"].tolist()
    dev = api_env.dataset.devices["device_id"].iloc[0]

    # Etiket sızıntısı koruması: ekstra ground-truth alanı reddedilmeli
    leak_resp = api_env.client.post(
        "/transactions/score",
        json={
            "sender_account_id": personal[0],
            "receiver_account_id": personal[1],
            "device_id": dev,
            "amount_minor": 10_000,
            "channel": Channel.MOBILE_APP.value,
            "transaction_type": TransactionType.TRANSFER.value,
            "is_fraud": True,
        },
    )
    assert leak_resp.status_code == 422

    # POS işleminde cihaz verilmesi yasak
    pos_resp = api_env.client.post(
        "/transactions/score",
        json={
            "sender_account_id": personal[0],
            "receiver_account_id": personal[1],
            "device_id": dev,
            "amount_minor": 10_000,
            "channel": Channel.POS.value,
            "transaction_type": TransactionType.CARD_PURCHASE.value,
        },
    )
    assert pos_resp.status_code == 422

    # Geriye giden zaman damgası kronoloji koruması nedeniyle reddedilmeli
    past_resp = api_env.client.post(
        "/transactions/score",
        json={
            "timestamp": (GEN_CFG.start_date + timedelta(hours=1)).isoformat(),
            "sender_account_id": personal[0],
            "receiver_account_id": personal[1],
            "device_id": dev,
            "amount_minor": 10_000,
            "channel": Channel.MOBILE_APP.value,
            "transaction_type": TransactionType.TRANSFER.value,
        },
    )
    assert past_resp.status_code == 422

    # update_state=False ("what-if") iç sayacı ve durumu değiştirmemeli
    before_processed = api_env.service.extractor.n_processed
    what_if_ts = api_env.next_ts(minutes=2)
    w1 = api_env.client.post(
        "/transactions/score",
        json={
            "timestamp": what_if_ts.isoformat(),
            "sender_account_id": personal[2],
            "receiver_account_id": personal[3],
            "device_id": dev,
            "amount_minor": 50_000,
            "channel": Channel.MOBILE_APP.value,
            "transaction_type": TransactionType.TRANSFER.value,
            "update_state": False,
            "persist": False,
        },
    )
    assert w1.status_code == 200
    assert api_env.service.extractor.n_processed == before_processed


def test_alerts_endpoints_and_status_lifecycle(api_env: ApiTestEnv) -> None:
    list_resp = api_env.client.get("/alerts", params={"limit": 10})
    assert list_resp.status_code == 200
    payload = list_resp.json()
    assert payload["total"] >= 1
    assert len(payload["items"]) >= 1

    first = payload["items"][0]
    alert_id = first["alert_id"]

    # Tekil alert sorgusu
    detail_resp = api_env.client.get(f"/alerts/{alert_id}")
    assert detail_resp.status_code == 200
    assert detail_resp.json()["alert_id"] == alert_id

    # Hesap filtresi
    acc = first["sender_account_id"]
    filtered = api_env.client.get("/alerts", params={"account_id": acc, "min_score": 0.7})
    assert filtered.status_code == 200
    assert any(item["alert_id"] == alert_id for item in filtered.json()["items"])

    # Durum güncellemesi: OPEN -> INVESTIGATING -> CONFIRMED_FRAUD -> CLOSED
    patch_resp = api_env.client.patch(
        f"/alerts/{alert_id}",
        json={
            "status": AlertStatus.INVESTIGATING.value,
            "assigned_to": "analyst_1",
            "resolution_note": "Cihaz ve IP korelasyonu inceleniyor",
        },
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["status"] == AlertStatus.INVESTIGATING.value
    assert patch_resp.json()["assigned_to"] == "analyst_1"

    close_resp = api_env.client.patch(
        f"/alerts/{alert_id}",
        json={"status": AlertStatus.CLOSED.value, "resolution_note": "Kapatıldı"},
    )
    assert close_resp.status_code == 200
    assert close_resp.json()["status"] == AlertStatus.CLOSED.value

    # CLOSED sonrası başka duruma geçiş engellenmeli
    invalid_resp = api_env.client.patch(
        f"/alerts/{alert_id}",
        json={"status": AlertStatus.OPEN.value},
    )
    assert invalid_resp.status_code == 422

    # Olmayan alert 404 dönmeli
    assert api_env.client.get("/alerts/9999999").status_code == 404


def test_graph_neighborhood_endpoint(api_env: ApiTestEnv) -> None:
    # Birden fazla hesabı olan bir fraud ring veya aktif hesap seç
    with session_scope(api_env.engine) as session:
        tx_row = session.scalars(select(Transaction).where(Transaction.device_id.is_not(None))).first()
    assert tx_row is not None
    target_acc = tx_row.sender_account_id

    resp = api_env.client.get(f"/graph/{target_acc}", params={"hops": 2, "max_nodes": 80})
    assert resp.status_code == 200
    graph = resp.json()
    assert graph["account_id"] == target_acc
    assert graph["center_account_id"] == target_acc
    assert graph["node_count"] == len(graph["nodes"]) >= 2
    assert graph["edge_count"] == len(graph["edges"]) >= 1

    center_nodes = [n for n in graph["nodes"] if n["is_center"]]
    assert len(center_nodes) == 1
    assert center_nodes[0]["id"] == target_acc
    assert center_nodes[0]["masked_iban"].startswith("TR**")
    assert "****" in center_nodes[0]["masked_iban"]

    edge_types = {e["edge_type"] for e in graph["edges"]}
    assert "money_flow" in edge_types

    # Olmayan hesap 404 dönmeli
    assert api_env.client.get("/graph/ACC-999999").status_code == 404


def test_sse_stream_events(api_env: ApiTestEnv) -> None:
    stream_resp = api_env.client.get("/stream/events", params={"replay_recent": 10})
    assert stream_resp.status_code == 200
    assert stream_resp.headers["content-type"].startswith("text/event-stream")

    frames = _parse_sse_frames(stream_resp.text)
    assert len(frames) >= 2
    event_types = {f["event"] for f in frames}
    assert "transaction" in event_types
    assert "alert" in event_types

    alert_frame = next(f for f in frames if f["event"] == "alert")
    alert_payload = alert_frame["data"]["data"]
    assert "risk_score" in alert_payload
    assert "reasons" in alert_payload

    # Yalnızca alert filtresi testi
    only_alerts_resp = api_env.client.get("/stream/events", params={"replay_recent": 10, "only_alerts": True})
    assert only_alerts_resp.status_code == 200
    alert_frames = _parse_sse_frames(only_alerts_resp.text)
    assert alert_frames and all(f["event"] == "alert" for f in alert_frames)


def test_async_simulator_streams_scenarios(api_env: ApiTestEnv) -> None:
    sim = TrafficSimulator.from_database(
        api_env.engine,
        base_timestamp=api_env.next_ts(minutes=5),
        seed=77,
        settings=api_env.settings,
    )
    app = api_env.client.app
    summary = asyncio.run(sim.run(count=15, tps=200.0, fraud_ratio=0.5, app=app))

    assert summary.total_sent == 15
    assert summary.succeeded == 15
    assert summary.failed == 0
    assert summary.alerts_triggered >= 1
    assert "NORMAL" in summary.by_scenario
    assert any(s in summary.by_scenario for s in ("FRAUD_RING", "ACCOUNT_TAKEOVER", "CARD_TESTING"))


def test_sse_live_push_with_async_client(api_env: ApiTestEnv) -> None:
    """Abone bağlıyken yeni işlem geldiğinde SSE akışına canlı düştüğünü doğrular."""
    personal = api_env.dataset.accounts[
        api_env.dataset.accounts["account_type"] == AccountType.PERSONAL.value
    ]["account_id"].tolist()
    dev = api_env.dataset.devices["device_id"].iloc[0]
    live_ts = api_env.next_ts(minutes=2)

    async def _run_live() -> list[str]:
        app = api_env.client.app
        transport = httpx.ASGITransport(app=app)
        received_events: list[str] = []

        async with httpx.AsyncClient(transport=transport, base_url="http://aegisgraph.local") as client:
            async def _listen() -> None:
                async with client.stream(
                    "GET",
                    "/stream/events",
                    params={"replay_recent": 0, "max_events": 1, "timeout_seconds": 3.0},
                ) as resp:
                    assert resp.status_code == 200
                    async for line in resp.aiter_lines():
                        if line.startswith("event: "):
                            received_events.append(line[7:])

            listener = asyncio.create_task(_listen())
            await asyncio.sleep(0.05)
            post_resp = await client.post(
                "/transactions/score",
                json={
                    "timestamp": live_ts.isoformat(),
                    "sender_account_id": personal[0],
                    "receiver_account_id": personal[2],
                    "device_id": dev,
                    "amount_minor": 25_000,
                    "channel": Channel.MOBILE_APP.value,
                    "transaction_type": TransactionType.TRANSFER.value,
                    "ip_address": "85.105.12.22",
                    "geo_country": "TR",
                },
            )
            assert post_resp.status_code == 200
            await listener
        return received_events

    events = asyncio.run(_run_live())
    assert events == ["transaction"]
