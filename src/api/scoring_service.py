"""AegisGraph canlı skorlama, alert yönetimi ve graf komşuluk servisi.

Eğitilmiş LightGBM modelini, durumlu `FeatureExtractor` örneğini ve SHAP
açıklayıcıyı bellekte tutarak tekil işlemleri milisaniyeler içinde skorlar,
eşik üzerindeki işlemleri `alerts` tablosuna kaydeder ve SSE abonelerine yayınlar.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
from sqlalchemy import Engine, func, or_, select

from config.settings import Settings, get_settings
from src.agent.investigator import FraudInvestigatorAgent
from src.api.simulator import VALID_SIMULATOR_SCENARIOS, TrafficSimulator
from src.compat import UTC, StrEnum
from src.database.db import get_engine, init_db, session_scope
from src.database.models import (
    GROUND_TRUTH_COLUMNS,
    Account,
    AccountDevice,
    AccountType,
    Alert,
    AlertStatus,
    Channel,
    Device,
    DeviceType,
    Transaction,
    TransactionType,
    mask_iban,
)
from src.features.pipeline import TX_INPUT_COLUMNS, FeatureExtractor
from src.features.tabular_features import (
    DEVICE_TYPE_CODES,
    MERCHANT_CATEGORY_CODES,
    AccountProfile,
    DeviceProfile,
)
from src.models.scorer import RiskScorer

logger = logging.getLogger(__name__)

_TX_GROUND_TRUTH = frozenset(GROUND_TRUTH_COLUMNS["transactions"])

_ALLOWED_STATUS_TRANSITIONS: dict[AlertStatus, frozenset[AlertStatus]] = {
    AlertStatus.OPEN: frozenset({
        AlertStatus.INVESTIGATING,
        AlertStatus.ESCALATED,
        AlertStatus.CONFIRMED_FRAUD,
        AlertStatus.FALSE_POSITIVE,
        AlertStatus.CLOSED,
    }),
    AlertStatus.INVESTIGATING: frozenset({
        AlertStatus.ESCALATED,
        AlertStatus.CONFIRMED_FRAUD,
        AlertStatus.FALSE_POSITIVE,
        AlertStatus.CLOSED,
    }),
    AlertStatus.ESCALATED: frozenset({
        AlertStatus.INVESTIGATING,
        AlertStatus.CONFIRMED_FRAUD,
        AlertStatus.FALSE_POSITIVE,
        AlertStatus.CLOSED,
    }),
    AlertStatus.CONFIRMED_FRAUD: frozenset({
        AlertStatus.ESCALATED,
        AlertStatus.CLOSED,
    }),
    AlertStatus.FALSE_POSITIVE: frozenset({
        AlertStatus.INVESTIGATING,
        AlertStatus.CLOSED,
    }),
    AlertStatus.CLOSED: frozenset(),
}


class Decision(StrEnum):
    """Risk skoruna göre işlem aksiyon kararı."""

    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    ALERT = "ALERT"
    BLOCK = "BLOCK"


def _generate_synthetic_iban(account_id: str) -> str:
    """Otomatik oluşturulan hesaplar için mod-97 geçerli sentetik TR IBAN üretir."""
    digits = "".join(ch for ch in account_id if ch.isdigit()).zfill(16)[-16:]
    bban = f"000610{digits}"
    rearranged = f"{bban}TR00"
    numeric = "".join(str(ord(c) - 55) if c.isalpha() else c for c in rearranged)
    check = 98 - (int(numeric) % 97)
    return f"TR{check:02d}{bban}"


def _to_naive_utc(value: Any) -> datetime:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.to_pydatetime()


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """SSE akışı için tekil olay kaydı."""

    event_id: int
    event_type: str
    timestamp: str
    data: dict[str, Any]


class EventBroadcaster:
    """İşlem ve alarm olaylarını bağlı SSE abonelerine güvenli şekilde dağıtır."""

    def __init__(self, max_history: int = 200) -> None:
        self._history: deque[StreamEvent] = deque(maxlen=max_history)
        self._subscribers: dict[asyncio.Queue[StreamEvent], asyncio.AbstractEventLoop | None] = {}
        self._lock = threading.Lock()
        self._counter = 0

    @staticmethod
    def _enqueue(queue: asyncio.Queue[StreamEvent], event: StreamEvent) -> None:
        if queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                queue.get_nowait()
        with contextlib.suppress(asyncio.QueueFull):
            queue.put_nowait(event)

    def publish(self, event_type: str, data: dict[str, Any]) -> StreamEvent:
        """Yeni olayı geçmiş tamponuna ekler ve tüm aktif SSE kuyruklarına iletir."""
        with self._lock:
            self._counter += 1
            event = StreamEvent(
                event_id=self._counter,
                event_type=event_type,
                timestamp=datetime.now(UTC).isoformat(),
                data=data,
            )
            self._history.append(event)
            subscribers = list(self._subscribers.items())

        try:
            current_loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        for queue, sub_loop in subscribers:
            if sub_loop is not None and not sub_loop.is_closed():
                if current_loop is sub_loop:
                    self._enqueue(queue, event)
                else:
                    try:
                        sub_loop.call_soon_threadsafe(self._enqueue, queue, event)
                    except RuntimeError:
                        self.unsubscribe(queue)
            else:
                self._enqueue(queue, event)
        return event

    def subscribe(self, max_queue_size: int = 200) -> asyncio.Queue[StreamEvent]:
        """Yeni bir SSE abonesi kaydeder ve olay kuyruğunu döndürür."""
        try:
            loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue(maxsize=max_queue_size)
        with self._lock:
            self._subscribers[queue] = loop
        return queue

    def unsubscribe(self, queue: asyncio.Queue[StreamEvent]) -> None:
        """SSE abonesini temizler."""
        with self._lock:
            self._subscribers.pop(queue, None)

    def recent_events(self, limit: int = 10, *, event_type: str | None = None) -> list[StreamEvent]:
        """Tampondaki son olayları döndürür."""
        if limit <= 0:
            return []
        with self._lock:
            items = list(self._history)
        if event_type is not None:
            items = [e for e in items if e.event_type == event_type]
        return items[-limit:]


class ScoringService:
    """Canlı işlem skorlama, alert kalıcılığı ve graf sorgu katmanı."""

    def __init__(
        self,
        scorer: RiskScorer,
        extractor: FeatureExtractor,
        engine: Engine | None = None,
        settings: Settings | None = None,
        broadcaster: EventBroadcaster | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.scorer = scorer
        self.extractor = extractor
        self.engine = engine
        self.broadcaster = broadcaster or EventBroadcaster(max_history=self.settings.api.stream_buffer_size)
        self.investigator = FraudInvestigatorAgent(
            engine=self.engine,
            extractor=self.extractor,
            settings=self.settings,
        )
        self._lock = threading.RLock()
        self._sim_lock = threading.Lock()
        self._sim_seq = 0
        self._tx_seq = self._init_tx_sequence()

    @classmethod
    def from_artifacts(
        cls,
        artifacts_dir: Path | None = None,
        engine: Engine | None = None,
        settings: Settings | None = None,
    ) -> ScoringService:
        """`artifacts/` altındaki eğitilmiş model ve sıcak `FeatureExtractor` durumundan servis kurar."""
        settings = settings or get_settings()
        artifacts_dir = artifacts_dir or settings.model.artifacts_dir
        extractor_path = artifacts_dir / "feature_extractor.joblib"
        if not extractor_path.exists():
            raise FileNotFoundError(
                f"FeatureExtractor durumu bulunamadı: {extractor_path}. Önce `python -m src.models.train` çalıştırın."
            )
        scorer = RiskScorer.from_artifacts(artifacts_dir, settings)
        extractor = FeatureExtractor.load(extractor_path)
        db_engine = engine or get_engine()
        init_db(db_engine)
        logger.info(
            "ScoringService yüklendi: işlenen=%d, alert_eşik=%.2f",
            extractor.n_processed,
            scorer.alert_threshold,
        )
        return cls(scorer=scorer, extractor=extractor, engine=db_engine, settings=settings)

    def _init_tx_sequence(self) -> int:
        seq = int(self.extractor.n_processed)
        if self.engine is None:
            return seq
        try:
            with session_scope(self.engine) as session:
                max_id = session.scalar(select(func.max(Transaction.transaction_id)))
            if isinstance(max_id, str) and max_id.startswith("TX-") and max_id[3:].isdigit():
                seq = max(seq, int(max_id[3:]))
        except Exception as exc:  # pragma: no cover - ilk kurulumda tablo yoksa güvenli geri dönüş
            logger.debug("Başlangıç işlem sayacı okunamadı: %s", exc)
        return seq

    def _next_tx_id(self) -> str:
        self._tx_seq += 1
        return f"TX-{self._tx_seq:08d}"

    def _decide(self, risk_score: float) -> Decision:
        thresholds = self.settings.risk
        if risk_score >= thresholds.block_score:
            return Decision.BLOCK
        if risk_score >= thresholds.alert_score:
            return Decision.ALERT
        if risk_score >= thresholds.review_score:
            return Decision.REVIEW
        return Decision.ALLOW

    def _next_timestamp(self) -> datetime:
        now_ts = datetime.now(UTC).replace(tzinfo=None).timestamp()
        last_ts = self.extractor.last_timestamp
        ts = max(now_ts, last_ts + 1.0) if last_ts is not None else now_ts
        return datetime.fromtimestamp(ts, tz=UTC).replace(tzinfo=None)

    def _ensure_tabular_profiles(
        self,
        sender: str,
        receiver: str,
        device_id: str | None,
        stamp: datetime,
    ) -> None:
        """Yeni gelen hesap veya cihazlar için bellek profillerini eksiksiz tutar."""
        accounts_map = self.extractor.tabular._accounts
        devices_map = self.extractor.tabular._devices
        ts_sec = stamp.replace(tzinfo=UTC).timestamp()

        for acc_id in (sender, receiver):
            if acc_id not in accounts_map:
                opened = ts_sec
                kyc = 1
                is_merchant = False
                m_cat = float("nan")
                if self.engine is not None:
                    with session_scope(self.engine) as session:
                        acc = session.get(Account, acc_id)
                        if acc is not None:
                            opened = pd.Timestamp(acc.opened_at).value / 1e9
                            kyc = int(acc.kyc_level)
                            is_merchant = acc.account_type is AccountType.MERCHANT
                            if acc.merchant_category is not None:
                                m_cat = float(MERCHANT_CATEGORY_CODES[acc.merchant_category.value])
                if isinstance(accounts_map, dict):
                    accounts_map[acc_id] = AccountProfile(
                        opened_ts=opened,
                        kyc_level=kyc,
                        is_merchant=is_merchant,
                        merchant_category_code=m_cat,
                    )

        if device_id and device_id not in devices_map and isinstance(devices_map, dict):
            type_code = DEVICE_TYPE_CODES[DeviceType.MOBILE_ANDROID.value]
            is_emu = False
            if self.engine is not None:
                with session_scope(self.engine) as session:
                    dev = session.get(Device, device_id)
                    if dev is not None:
                        type_code = DEVICE_TYPE_CODES[dev.device_type.value]
                        is_emu = bool(dev.is_emulator)
            devices_map[device_id] = DeviceProfile(type_code=type_code, is_emulator=is_emu)

    def _persist_scored_transaction(
        self,
        *,
        tx_id: str,
        stamp: datetime,
        sender: str,
        receiver: str,
        device_id: str | None,
        amount_minor: int,
        currency: str,
        channel: Channel,
        tx_type: TransactionType,
        ip_address: str | None,
        geo_city: str | None,
        geo_country: str,
        risk_score: float,
        ml_score: float,
        graph_score: float,
        is_alert: bool,
        reasons: list[dict[str, Any]],
    ) -> tuple[int | None, str | None, str | None]:
        """İşlemi ve gerekirse Alert kaydını veritabanına yazar; maskelenmiş IBAN'ları döner."""
        assert self.engine is not None
        with session_scope(self.engine) as session:
            sender_acc = session.get(Account, sender)
            if sender_acc is None:
                sender_acc = Account(
                    account_id=sender,
                    iban=_generate_synthetic_iban(sender),
                    holder_name=f"Müşteri {sender}",
                    account_type=AccountType.PERSONAL,
                    kyc_level=1,
                    city=geo_city or "İstanbul",
                    merchant_category=None,
                    opened_at=stamp,
                    is_fraudster=False,
                    fraud_case_id=None,
                )
                session.add(sender_acc)

            receiver_acc = session.get(Account, receiver)
            if receiver_acc is None:
                receiver_acc = Account(
                    account_id=receiver,
                    iban=_generate_synthetic_iban(receiver),
                    holder_name=f"Müşteri {receiver}",
                    account_type=AccountType.PERSONAL,
                    kyc_level=1,
                    city="İstanbul",
                    merchant_category=None,
                    opened_at=stamp,
                    is_fraudster=False,
                    fraud_case_id=None,
                )
                session.add(receiver_acc)

            if device_id is not None:
                dev_row = session.get(Device, device_id)
                if dev_row is None:
                    dev_row = Device(
                        device_id=device_id,
                        fingerprint=f"{abs(hash(device_id)):032x}"[:32],
                        device_type=DeviceType.MOBILE_ANDROID,
                        is_emulator=False,
                        first_seen=stamp,
                    )
                    session.add(dev_row)
                    session.flush()
                link = session.get(AccountDevice, (sender, device_id))
                if link is None:
                    session.add(
                        AccountDevice(account_id=sender, device_id=device_id, first_seen=stamp, last_seen=stamp)
                    )
                elif stamp > link.last_seen:
                    link.last_seen = stamp

            session.flush()

            existing_tx = session.get(Transaction, tx_id)
            if existing_tx is None:
                session.add(
                    Transaction(
                        transaction_id=tx_id,
                        timestamp=stamp,
                        sender_account_id=sender,
                        receiver_account_id=receiver,
                        device_id=device_id,
                        amount_minor=amount_minor,
                        currency=currency,
                        channel=channel,
                        transaction_type=tx_type,
                        ip_address=ip_address,
                        geo_city=geo_city,
                        geo_country=geo_country,
                        is_fraud=False,
                        fraud_case_id=None,
                    )
                )
                session.flush()

            alert_id: int | None = None
            if is_alert:
                existing_alert = session.scalars(select(Alert).where(Alert.transaction_id == tx_id)).first()
                if existing_alert is None:
                    alert = Alert(
                        transaction_id=tx_id,
                        risk_score=round(risk_score, 6),
                        ml_score=round(ml_score, 6),
                        graph_score=round(graph_score, 4),
                        status=AlertStatus.OPEN,
                        reasons=reasons,
                    )
                    session.add(alert)
                    session.flush()
                    alert_id = alert.alert_id
                else:
                    alert_id = existing_alert.alert_id

            return alert_id, sender_acc.iban, receiver_acc.iban

    def score_transaction(
        self,
        tx: Mapping[str, Any],
        *,
        update_state: bool = True,
        persist: bool = True,
    ) -> dict[str, Any]:
        """Tekil işlemi anlık skorlar, SHAP gerekçelerini çıkarır ve gerekirse alert üretir."""
        required = {"sender_account_id", "receiver_account_id", "amount_minor", "channel", "transaction_type"}
        missing = required - set(tx)
        if missing:
            raise ValueError(f"Eksik işlem alanı: {sorted(missing)}")

        sender = str(tx["sender_account_id"])
        receiver = str(tx["receiver_account_id"])
        if sender == receiver:
            raise ValueError("Gönderen ve alıcı aynı hesap olamaz")

        amount_minor = int(tx["amount_minor"])
        if amount_minor <= 0:
            raise ValueError("amount_minor sıfırdan büyük tamsayı olmalı")

        channel_enum = Channel(str(tx["channel"]))
        tx_type_enum = TransactionType(str(tx["transaction_type"]))
        device_id = str(tx["device_id"]) if tx.get("device_id") is not None else None
        if (channel_enum is Channel.POS) != (device_id is None):
            raise ValueError("POS işlemlerinde cihaz olmaz; dijital kanallarda cihaz zorunludur")

        ip_address = str(tx["ip_address"]) if tx.get("ip_address") is not None else None
        geo_city = str(tx["geo_city"]) if tx.get("geo_city") is not None else None
        geo_country = str(tx.get("geo_country") or self.settings.features.home_country).upper()
        currency = str(tx.get("currency") or self.settings.generator.currency).upper()

        # Ground-truth alanları (varsa bile) kesinlikle özellik çıkarımına sokulmaz
        clean_input = {k: v for k, v in tx.items() if k not in _TX_GROUND_TRUTH}

        with self._lock:
            stamp = _to_naive_utc(clean_input["timestamp"]) if clean_input.get("timestamp") else self._next_timestamp()
            tx_id = str(clean_input.get("transaction_id") or self._next_tx_id())

            self._ensure_tabular_profiles(sender, receiver, device_id, stamp)

            payload: dict[str, Any] = {
                "transaction_id": tx_id,
                "timestamp": stamp,
                "sender_account_id": sender,
                "receiver_account_id": receiver,
                "device_id": device_id,
                "amount_minor": amount_minor,
                "channel": channel_enum.value,
                "transaction_type": tx_type_enum.value,
                "ip_address": ip_address,
                "geo_country": geo_country,
            }
            for col in TX_INPUT_COLUMNS:
                payload.setdefault(col, None)

            score_res = self.scorer.score_transaction(self.extractor, payload, update_state=update_state)
            explanation_dict = score_res.explanation.to_dict() if score_res.explanation else None
            reasons: list[dict[str, Any]] = explanation_dict["top_factors"] if explanation_dict else []
            graph_score = round(float(explanation_dict["graph_contribution_share"]), 4) if explanation_dict else 0.0
            ml_score = round(float(score_res.risk_score), 6)
            decision = self._decide(score_res.risk_score)

            alert_id: int | None = None
            sender_iban: str | None = None
            receiver_iban: str | None = None
            if persist and update_state and self.engine is not None:
                alert_id, sender_iban, receiver_iban = self._persist_scored_transaction(
                    tx_id=tx_id,
                    stamp=stamp,
                    sender=sender,
                    receiver=receiver,
                    device_id=device_id,
                    amount_minor=amount_minor,
                    currency=currency,
                    channel=channel_enum,
                    tx_type=tx_type_enum,
                    ip_address=ip_address,
                    geo_city=geo_city,
                    geo_country=geo_country,
                    risk_score=score_res.risk_score,
                    ml_score=ml_score,
                    graph_score=graph_score,
                    is_alert=score_res.is_alert,
                    reasons=reasons,
                )

        result: dict[str, Any] = {
            "transaction_id": tx_id,
            "timestamp": stamp.isoformat(),
            "sender_account_id": sender,
            "receiver_account_id": receiver,
            "amount_minor": amount_minor,
            "amount": amount_minor / 100.0,
            "currency": currency,
            "channel": channel_enum.value,
            "transaction_type": tx_type_enum.value,
            "risk_score": round(float(score_res.risk_score), 6),
            "ml_score": ml_score,
            "graph_score": graph_score,
            "decision": decision.value,
            "is_alert": score_res.is_alert,
            "alert_id": alert_id,
            "investigation_id": None,
            "reasons": reasons,
            "explanation": explanation_dict,
            "latency_ms": round(float(score_res.latency_ms), 3),
        }

        logger.info(
            "İşlem skorlandı id=%s %s (%s) -> %s (%s) tutar=%.2f %s skor=%.4f graf=%.2f karar=%s",
            tx_id,
            sender,
            mask_iban(sender_iban),
            receiver,
            mask_iban(receiver_iban),
            amount_minor / 100.0,
            currency,
            result["risk_score"],
            graph_score,
            decision.value,
        )

        if update_state:
            s_masked = mask_iban(sender_iban)
            r_masked = mask_iban(receiver_iban)
            stream_payload = {
                "transaction_id": tx_id,
                "timestamp": result["timestamp"],
                "sender_account_id": sender,
                "receiver_account_id": receiver,
                "sender_masked_iban": s_masked,
                "receiver_masked_iban": r_masked,
                "amount": result["amount"],
                "currency": currency,
                "channel": channel_enum.value,
                "risk_score": result["risk_score"],
                "ml_score": ml_score,
                "graph_score": graph_score,
                "decision": decision.value,
                "is_alert": score_res.is_alert,
                "alert_id": alert_id,
                "reasons": reasons,
                "latency_ms": result["latency_ms"],
            }
            self.broadcaster.publish("transaction", stream_payload)
            if score_res.is_alert:
                self.broadcaster.publish(
                    "alert",
                    {
                        "alert_id": alert_id,
                        "transaction_id": tx_id,
                        "timestamp": result["timestamp"],
                        "sender_account_id": sender,
                        "receiver_account_id": receiver,
                        "sender_masked_iban": s_masked,
                        "receiver_masked_iban": r_masked,
                        "amount": result["amount"],
                        "currency": currency,
                        "risk_score": result["risk_score"],
                        "ml_score": ml_score,
                        "graph_score": graph_score,
                        "decision": decision.value,
                        "status": AlertStatus.OPEN.value,
                        "reasons": reasons,
                    },
                )
                if (
                    persist
                    and self.engine is not None
                    and alert_id is not None
                    and score_res.risk_score >= self.settings.agent.auto_investigate_min_score
                ):
                    # Ajan hatası skorlama yanıtını asla düşürmez; alarm zaten kaydedilmiştir
                    try:
                        inv_payload = self.investigate_alert(alert_id, trigger="auto")
                        result["investigation_id"] = inv_payload.get("investigation_id")
                    except Exception:
                        logger.exception("Otomatik soruşturma başarısız alert_id=%d; alarm OPEN kalır", alert_id)
                        self.broadcaster.publish(
                            "investigation_failed", {"alert_id": alert_id, "transaction_id": tx_id}
                        )
        return result

    # ------------------------------------------------------------------ #
    # Otonom soruşturma
    # ------------------------------------------------------------------ #
    def investigate_alert(self, alert_id: int, *, trigger: str = "manual") -> dict[str, Any]:
        """İlgili alarm için otonom soruşturmayı çalıştırır, kaydeder ve SSE akışına yayınlar.

        `trigger`: `manual` (API ucu) veya `auto` (eşik üstü kritik alarm) — yalnızca SSE olayına eklenir.
        """
        if self.engine is None:
            raise KeyError(f"Alarm bulunamadı: {alert_id}")
        report = self.investigator.investigate(alert_id, persist=True)
        payload = report.model_dump(mode="json")
        self.broadcaster.publish("investigation", {**payload, "trigger": trigger})
        return payload

    def get_investigation(self, alert_id: int) -> dict[str, Any]:
        """Veritabanında kayıtlı soruşturma raporunu döndürür."""
        if self.engine is None:
            raise KeyError(f"Alarm için kayıtlı soruşturma raporu bulunamadı: {alert_id}")
        report = self.investigator.get_report(alert_id)
        return report.model_dump(mode="json")

    def run_simulation(
        self,
        *,
        scenario: str = "ALL",
        count: int = 15,
        tps: float | None = None,
    ) -> dict[str, Any]:
        """Seçilen senaryo için trafik simülasyonu üretir ve işlemleri canlı skorlama akışına basar."""
        normalized = str(scenario).upper()
        if normalized not in VALID_SIMULATOR_SCENARIOS:
            raise ValueError(f"Geçersiz simülasyon senaryosu: {scenario!r}")
        if self.engine is None:
            raise RuntimeError("Simülasyon için veritabanı bağlantısı gereklidir")

        resolved_tps = tps or self.settings.api.simulator_tps
        interval = 1.0 / resolved_tps if resolved_tps > 0 else 0.0

        with self._sim_lock:
            with self._lock:
                self._sim_seq += 1
                seed = self.settings.generator.seed + self._sim_seq
                last_ts = self.extractor.last_timestamp
                base_ts = (
                    datetime.fromtimestamp(last_ts, tz=UTC).replace(tzinfo=None) + timedelta(minutes=1)
                    if last_ts is not None
                    else self.settings.generator.start_date + timedelta(days=31)
                )

            simulator = TrafficSimulator.from_database(
                self.engine,
                base_timestamp=base_ts,
                seed=seed,
                settings=self.settings,
            )
            stream = simulator.build_stream(count=count, scenario=normalized)

            succeeded = 0
            alerts_triggered = 0
            investigations_triggered = 0
            for idx, item in enumerate(stream):
                t_start = time.perf_counter()
                res = self.score_transaction(item.payload, update_state=True, persist=True)
                succeeded += 1
                if res.get("is_alert"):
                    alerts_triggered += 1
                if res.get("investigation_id") is not None:
                    investigations_triggered += 1
                if idx + 1 < len(stream) and interval > 0:
                    wait_sec = interval - (time.perf_counter() - t_start)
                    if wait_sec > 0:
                        time.sleep(wait_sec)

        logger.info(
            "Simülasyon akışı tamamlandı senaryo=%s işlem=%d alarm=%d soruşturma=%d",
            normalized,
            succeeded,
            alerts_triggered,
            investigations_triggered,
        )
        return {
            "scenario": normalized,
            "total_sent": len(stream),
            "succeeded": succeeded,
            "alerts_triggered": alerts_triggered,
            "investigations_triggered": investigations_triggered,
        }

    # ------------------------------------------------------------------ #
    # Alert sorgulama & durum güncelleme
    # ------------------------------------------------------------------ #
    def _fetch_iban_map(self, session: Any, txs: list[Transaction | None]) -> dict[str, str]:
        acc_ids = {
            acc_id
            for tx in txs
            if tx is not None
            for acc_id in (tx.sender_account_id, tx.receiver_account_id)
        }
        if not acc_ids:
            return {}
        rows = session.execute(
            select(Account.account_id, Account.iban).where(Account.account_id.in_(acc_ids))
        ).all()
        return {str(acc_id): str(iban) for acc_id, iban in rows}

    def _serialize_alert(
        self,
        alert: Alert,
        tx: Transaction | None,
        iban_map: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        ibans = iban_map or {}
        return {
            "alert_id": alert.alert_id,
            "transaction_id": alert.transaction_id,
            "created_at": alert.created_at.isoformat() if alert.created_at else None,
            "updated_at": alert.updated_at.isoformat() if alert.updated_at else None,
            "risk_score": float(alert.risk_score),
            "ml_score": float(alert.ml_score) if alert.ml_score is not None else None,
            "graph_score": float(alert.graph_score) if alert.graph_score is not None else None,
            "status": alert.status.value,
            "reasons": list(alert.reasons or []),
            "assigned_to": alert.assigned_to,
            "resolution_note": alert.resolution_note,
            "sender_account_id": tx.sender_account_id if tx else None,
            "receiver_account_id": tx.receiver_account_id if tx else None,
            "sender_masked_iban": mask_iban(ibans.get(tx.sender_account_id)) if tx else None,
            "receiver_masked_iban": mask_iban(ibans.get(tx.receiver_account_id)) if tx else None,
            "amount_minor": int(tx.amount_minor) if tx else None,
            "amount": float(tx.amount_minor) / 100.0 if tx else None,
            "currency": tx.currency if tx else None,
            "channel": tx.channel.value if tx else None,
            "transaction_timestamp": tx.timestamp.isoformat() if tx else None,
        }

    def list_alerts(
        self,
        *,
        status: AlertStatus | None = None,
        min_score: float | None = None,
        account_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> dict[str, Any]:
        """`alerts` tablosundan filtreli ve sayfalı alarm listesi döndürür."""
        if self.engine is None:
            return {"total": 0, "limit": 0, "offset": offset, "items": []}

        resolved_limit = min(
            limit or self.settings.api.default_alerts_limit,
            self.settings.api.max_alerts_limit,
        )
        filters = []
        if status is not None:
            filters.append(Alert.status == status)
        if min_score is not None:
            filters.append(Alert.risk_score >= min_score)
        if account_id is not None:
            filters.append(
                or_(
                    Transaction.sender_account_id == account_id,
                    Transaction.receiver_account_id == account_id,
                )
            )

        with session_scope(self.engine) as session:
            count_stmt = (
                select(func.count(Alert.alert_id))
                .select_from(Alert)
                .join(Transaction, Alert.transaction_id == Transaction.transaction_id)
                .where(*filters)
            )
            total = int(session.scalar(count_stmt) or 0)

            rows_stmt = (
                select(Alert, Transaction)
                .join(Transaction, Alert.transaction_id == Transaction.transaction_id)
                .where(*filters)
                .order_by(Alert.alert_id.desc())
                .offset(offset)
                .limit(resolved_limit)
            )
            rows = session.execute(rows_stmt).all()
            iban_map = self._fetch_iban_map(session, [tx for _, tx in rows])
            items = [self._serialize_alert(alert, tx, iban_map) for alert, tx in rows]

        return {"total": total, "limit": resolved_limit, "offset": offset, "items": items}

    def get_alert(self, alert_id: int) -> dict[str, Any]:
        """Tekil alarm detayını döndürür."""
        if self.engine is None:
            raise KeyError(f"Alarm bulunamadı: {alert_id}")
        with session_scope(self.engine) as session:
            row = session.execute(
                select(Alert, Transaction)
                .join(Transaction, Alert.transaction_id == Transaction.transaction_id)
                .where(Alert.alert_id == alert_id)
            ).first()
            if row is None:
                raise KeyError(f"Alarm bulunamadı: {alert_id}")
            iban_map = self._fetch_iban_map(session, [row[1]])
            return self._serialize_alert(row[0], row[1], iban_map)

    def update_alert_status(
        self,
        alert_id: int,
        *,
        status: AlertStatus,
        assigned_to: str | None = None,
        resolution_note: str | None = None,
    ) -> dict[str, Any]:
        """Alarm durumunu domain geçiş kurallarına uygun şekilde günceller."""
        if self.engine is None:
            raise KeyError(f"Alarm bulunamadı: {alert_id}")
        with session_scope(self.engine) as session:
            row = session.execute(
                select(Alert, Transaction)
                .join(Transaction, Alert.transaction_id == Transaction.transaction_id)
                .where(Alert.alert_id == alert_id)
            ).first()
            if row is None:
                raise KeyError(f"Alarm bulunamadı: {alert_id}")
            alert, tx = row
            if status is not alert.status and status not in _ALLOWED_STATUS_TRANSITIONS[alert.status]:
                raise ValueError(f"Geçersiz alarm durum geçişi: {alert.status.value} -> {status.value}")
            alert.status = status
            if assigned_to is not None:
                alert.assigned_to = assigned_to
            if resolution_note is not None:
                alert.resolution_note = resolution_note
            session.flush()
            iban_map = self._fetch_iban_map(session, [tx])
            updated = self._serialize_alert(alert, tx, iban_map)

        self.broadcaster.publish("alert_updated", updated)
        return updated

    # ------------------------------------------------------------------ #
    # Graf komşuluk sorgulama
    # ------------------------------------------------------------------ #
    def get_account_graph(
        self,
        account_id: str,
        *,
        hops: int | None = None,
        max_nodes: int | None = None,
    ) -> dict[str, Any]:
        """Şüpheli hesabın etrafındaki para akışı ve cihaz/IP ağ komşuluğunu JSON olarak döndürür."""
        hop_limit = min(max(hops or self.settings.api.graph_default_hops, 1), self.settings.api.graph_max_hops)
        node_limit = max(max_nodes or self.settings.api.graph_max_nodes, 10)

        with self._lock:
            flow = self.extractor.graph.flow
            links = self.extractor.graph.links
            pagerank_map = dict(self.extractor.graph._pagerank)
            clustering_map = dict(self.extractor.graph._clustering)

            in_memory = account_id in flow or ("A", account_id) in links

            account_exists_in_db = False
            if self.engine is not None:
                with session_scope(self.engine) as session:
                    account_exists_in_db = session.get(Account, account_id) is not None

            if not in_memory and not account_exists_in_db:
                raise KeyError(f"Hesap bulunamadı: {account_id}")

            # 1) Para akışı ve bipartit cihaz/IP ağından BFS ile komşuluğu topla
            account_nodes: set[str] = {account_id}
            device_nodes: set[str] = set()
            ip_nodes: set[str] = set()

            frontier: set[str] = {account_id}
            for _ in range(hop_limit):
                if not frontier or (len(account_nodes) + len(device_nodes) + len(ip_nodes)) >= node_limit:
                    break
                next_frontier: set[str] = set()
                for acc in sorted(frontier):
                    if acc in flow:
                        neighbors = sorted(set(flow.successors(acc)) | set(flow.predecessors(acc)))
                        for nbr in neighbors:
                            if len(account_nodes) + len(device_nodes) + len(ip_nodes) >= node_limit:
                                break
                            if nbr not in account_nodes:
                                account_nodes.add(nbr)
                                next_frontier.add(nbr)
                    link_node = ("A", acc)
                    if link_node in links:
                        for kind, val in sorted(links.neighbors(link_node), key=lambda x: (x[0], str(x[1]))):
                            if len(account_nodes) + len(device_nodes) + len(ip_nodes) >= node_limit:
                                break
                            if kind == "D":
                                device_nodes.add(val)
                                for linked_kind, linked_acc in links.neighbors((kind, val)):
                                    if linked_kind == "A" and linked_acc not in account_nodes:
                                        if len(account_nodes) + len(device_nodes) + len(ip_nodes) >= node_limit:
                                            break
                                        account_nodes.add(linked_acc)
                                        next_frontier.add(linked_acc)
                            elif kind == "I" and val is not None:
                                ip_nodes.add(val)
                                for linked_kind, linked_acc in links.neighbors((kind, val)):
                                    if linked_kind == "A" and linked_acc not in account_nodes:
                                        if len(account_nodes) + len(device_nodes) + len(ip_nodes) >= node_limit:
                                            break
                                        account_nodes.add(linked_acc)
                                        next_frontier.add(linked_acc)
                frontier = next_frontier

            # Merkez ve komşu hesaplara bağlı cihaz/IP bağlantılarını kenar bütünlüğü için topla
            for acc in list(account_nodes):
                link_node = ("A", acc)
                if link_node in links:
                    for kind, val in links.neighbors(link_node):
                        if len(account_nodes) + len(device_nodes) + len(ip_nodes) >= node_limit:
                            break
                        if kind == "D":
                            device_nodes.add(val)
                        elif kind == "I" and val is not None:
                            ip_nodes.add(val)

            # Kısa döngü tespiti (merkez hesap içeren en az 2 adımlı yönlü döngü)
            short_cycle_detected = False
            if account_id in flow:
                sub_flow = flow.subgraph(account_nodes)
                for succ in sub_flow.successors(account_id):
                    if succ == account_id:
                        continue
                    try:
                        if nx.has_path(sub_flow, succ, account_id):
                            short_cycle_detected = True
                            break
                    except nx.NetworkXError:
                        continue

            # Snapshot değerlerini kopyala
            flow_edges_data: list[tuple[str, str, dict[str, Any]]] = []
            for u, v, data in flow.edges(data=True):
                if u in account_nodes and v in account_nodes:
                    flow_edges_data.append((u, v, dict(data)))

            degree_map = {
                acc: (flow.in_degree(acc) if acc in flow else 0, flow.out_degree(acc) if acc in flow else 0)
                for acc in account_nodes
            }

            device_links_data: list[tuple[str, str, int]] = []
            for dev in sorted(device_nodes):
                dev_node = ("D", dev)
                shared = len(links.adj[dev_node]) if dev_node in links else 0
                if dev_node in links:
                    for kind, acc in links.neighbors(dev_node):
                        if kind == "A" and acc in account_nodes:
                            device_links_data.append((acc, dev, shared))

            ip_links_data: list[tuple[str, str, int]] = []
            for blk in sorted(ip_nodes):
                ip_node = ("I", blk)
                shared = len(links.adj[ip_node]) if ip_node in links else 0
                if ip_node in links:
                    for kind, acc in links.neighbors(ip_node):
                        if kind == "A" and acc in account_nodes:
                            ip_links_data.append((acc, blk, shared))

        # 2) DB metadata zenginleştirmesi (ground truth alanları hariç!)
        db_accounts: dict[str, Account] = {}
        db_devices: dict[str, Device] = {}
        alert_stats: dict[str, tuple[int, float]] = {}

        if self.engine is not None:
            with session_scope(self.engine) as session:
                if account_nodes:
                    for acc_row in session.scalars(
                        select(Account).where(Account.account_id.in_(account_nodes))
                    ):
                        db_accounts[acc_row.account_id] = acc_row

                    # Hesaba bağlı alarmlar (gönderici veya alıcı olarak)
                    alert_rows = session.execute(
                        select(Transaction.sender_account_id, Transaction.receiver_account_id, Alert.risk_score)
                        .join(Alert, Alert.transaction_id == Transaction.transaction_id)
                        .where(
                            or_(
                                Transaction.sender_account_id.in_(account_nodes),
                                Transaction.receiver_account_id.in_(account_nodes),
                            )
                        )
                    ).all()
                    counts: dict[str, int] = {}
                    max_scores: dict[str, float] = {}
                    for s_id, r_id, score in alert_rows:
                        for target_acc in (s_id, r_id):
                            if target_acc in account_nodes:
                                counts[target_acc] = counts.get(target_acc, 0) + 1
                                max_scores[target_acc] = max(max_scores.get(target_acc, 0.0), float(score))
                    alert_stats = {acc: (counts[acc], max_scores[acc]) for acc in counts}

                if device_nodes:
                    for dev_row in session.scalars(select(Device).where(Device.device_id.in_(device_nodes))):
                        db_devices[dev_row.device_id] = dev_row

        # 3) Düğümleri ve kenarları inşa et
        nodes: list[dict[str, Any]] = []
        for acc in sorted(account_nodes, key=lambda a: (a != account_id, a)):
            acc_row = db_accounts.get(acc)
            in_deg, out_deg = degree_map[acc]
            a_count, a_max = alert_stats.get(acc, (0, 0.0))
            nodes.append({
                "id": acc,
                "node_type": "account",
                "label": acc_row.holder_name if acc_row else acc,
                "is_center": acc == account_id,
                "account_type": acc_row.account_type.value if acc_row else AccountType.PERSONAL.value,
                "kyc_level": int(acc_row.kyc_level) if acc_row else None,
                "city": acc_row.city if acc_row else None,
                "masked_iban": mask_iban(acc_row.iban) if acc_row else None,
                "in_degree": in_deg,
                "out_degree": out_deg,
                "pagerank": round(float(pagerank_map.get(acc, 0.0)), 4),
                "clustering": round(float(clustering_map.get(acc, 0.0)), 4),
                "alert_count": a_count,
                "max_risk_score": round(a_max, 4) if a_count else None,
                "device_type": None,
                "is_emulator": None,
                "shared_account_count": None,
            })

        device_shared_map = {dev: shared for _, dev, shared in device_links_data}
        for dev in sorted(device_nodes):
            dev_row = db_devices.get(dev)
            nodes.append({
                "id": dev,
                "node_type": "device",
                "label": dev_row.device_type.value if dev_row else dev,
                "is_center": False,
                "account_type": None,
                "kyc_level": None,
                "city": None,
                "masked_iban": None,
                "in_degree": None,
                "out_degree": None,
                "pagerank": None,
                "clustering": None,
                "alert_count": 0,
                "max_risk_score": None,
                "device_type": dev_row.device_type.value if dev_row else None,
                "is_emulator": bool(dev_row.is_emulator) if dev_row else None,
                "shared_account_count": device_shared_map.get(dev, 0),
            })

        ip_shared_map = {blk: shared for _, blk, shared in ip_links_data}
        for blk in sorted(ip_nodes):
            nodes.append({
                "id": f"IP:{blk}",
                "node_type": "ip_block",
                "label": f"{blk}.0/24",
                "is_center": False,
                "account_type": None,
                "kyc_level": None,
                "city": None,
                "masked_iban": None,
                "in_degree": None,
                "out_degree": None,
                "pagerank": None,
                "clustering": None,
                "alert_count": 0,
                "max_risk_score": None,
                "device_type": None,
                "is_emulator": None,
                "shared_account_count": ip_shared_map.get(blk, 0),
            })

        edges: list[dict[str, Any]] = []
        for u, v, data in sorted(flow_edges_data, key=lambda item: (item[0], item[1])):
            total_amt = float(data.get("amount", 0.0))
            last_ts_val = data.get("last_ts")
            last_iso = (
                datetime.fromtimestamp(float(last_ts_val), tz=UTC).replace(tzinfo=None).isoformat()
                if last_ts_val is not None
                else None
            )
            edges.append({
                "source": u,
                "target": v,
                "edge_type": "money_flow",
                "tx_count": int(data.get("count", 0)),
                "total_amount": round(total_amt, 2),
                "total_amount_minor": round(total_amt * 100),
                "last_timestamp": last_iso,
            })

        for acc, dev, _ in sorted(device_links_data):
            edges.append({
                "source": acc,
                "target": dev,
                "edge_type": "uses_device",
                "tx_count": None,
                "total_amount": None,
                "total_amount_minor": None,
                "last_timestamp": None,
            })

        for acc, blk, _ in sorted(ip_links_data):
            edges.append({
                "source": acc,
                "target": f"IP:{blk}",
                "edge_type": "uses_ip_block",
                "tx_count": None,
                "total_amount": None,
                "total_amount_minor": None,
                "last_timestamp": None,
            })

        center_devices = {dev for acc, dev, _ in device_links_data if acc == account_id}
        center_ips = {blk for acc, blk, _ in ip_links_data if acc == account_id}
        shared_dev_accounts = sorted({
            acc for acc, dev, shared in device_links_data if dev in center_devices and acc != account_id and shared > 1
        })
        shared_ip_accounts = sorted({
            acc for acc, blk, shared in ip_links_data if blk in center_ips and acc != account_id and shared > 1
        })

        return {
            "account_id": account_id,
            "center_account_id": account_id,
            "hops": hop_limit,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "shared_device_accounts": shared_dev_accounts,
            "shared_ip_accounts": shared_ip_accounts,
            "short_cycle_detected": short_cycle_detected,
            "nodes": nodes,
            "edges": edges,
        }
