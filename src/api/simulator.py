"""AegisGraph asenkron canlı trafik simülatörü.

Veritabanındaki hesap/cihaz profillerini kullanarak normal işlemleri ve
sentetik dolandırıcılık senaryolarını (Fraud Ring, Account Takeover, Card Testing)
kronolojik sırada `POST /transactions/score` ucuna gönderir.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal

import httpx
import numpy as np
import pandas as pd
from fastapi import FastAPI
from sqlalchemy import Engine, func, select

from config.settings import Settings, get_settings
from src.database.db import get_engine, load_table, session_scope
from src.database.models import AccountType, Channel, Transaction, TransactionType

logger = logging.getLogger(__name__)

SimulatorScenario = Literal["ALL", "NORMAL", "FRAUD_RING", "ACCOUNT_TAKEOVER", "CARD_TESTING"]
VALID_SIMULATOR_SCENARIOS: tuple[str, ...] = (
    "ALL",
    "NORMAL",
    "FRAUD_RING",
    "ACCOUNT_TAKEOVER",
    "CARD_TESTING",
)


@dataclass(frozen=True, slots=True)
class SimulatedTransaction:
    """Simülatör tarafından üretilen tekil işlem ve iç senaryo etiketi (API'ye gönderilmez)."""

    scenario: str
    payload: dict[str, Any]


@dataclass
class SimulationSummary:
    """Simülasyon koşusu sonuç özeti."""

    total_sent: int
    succeeded: int
    failed: int
    alerts_triggered: int
    avg_latency_ms: float
    by_scenario: dict[str, dict[str, int]] = field(default_factory=dict)
    responses: list[dict[str, Any]] = field(default_factory=list)


class TrafficSimulator:
    """Deterministik (`np.random.default_rng`) canlı işlem akışı simülatörü."""

    def __init__(
        self,
        *,
        accounts: pd.DataFrame,
        devices: pd.DataFrame,
        account_devices: pd.DataFrame | None = None,
        base_timestamp: datetime | None = None,
        seed: int = 42,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.rng = np.random.default_rng(seed)
        self._cursor = base_timestamp or datetime(2026, 11, 1, 12, 0, 0)

        personal = accounts[accounts["account_type"] != AccountType.MERCHANT.value]
        merchants = accounts[accounts["account_type"] == AccountType.MERCHANT.value]
        if len(personal) < 4:
            raise ValueError("Simülatör için en az 4 kişisel/kurumsal hesap gereklidir")

        # Düşük KYC ve daha genç hesapları halka adayı olarak önceliklendir
        self._personal_ids = personal["account_id"].astype(str).tolist()
        low_kyc = personal.sort_values(["kyc_level", "opened_at"], ascending=[True, False])
        self._ring_pool = low_kyc["account_id"].astype(str).tolist()
        self._merchant_ids = (
            merchants["account_id"].astype(str).tolist() if len(merchants) else self._personal_ids[-2:]
        )

        self._all_devices = devices["device_id"].astype(str).tolist()
        emulators = devices[devices["is_emulator"].astype(bool)]["device_id"].astype(str).tolist()
        self._emulator_devices = emulators if emulators else self._all_devices

        self._known_device: dict[str, str] = {}
        if account_devices is not None and len(account_devices) > 0:
            for row in account_devices.itertuples(index=False):
                self._known_device.setdefault(str(row.account_id), str(row.device_id))

    @classmethod
    def from_database(
        cls,
        engine: Engine | None = None,
        *,
        base_timestamp: datetime | None = None,
        seed: int = 42,
        settings: Settings | None = None,
    ) -> TrafficSimulator:
        """Veritabanındaki hesap, cihaz ve son işlem zaman damgasından simülatör başlatır."""
        resolved_settings = settings or get_settings()
        db_engine = engine or get_engine()
        accounts = load_table("accounts", db_engine)
        devices = load_table("devices", db_engine)
        account_devices = load_table("account_devices", db_engine)

        if base_timestamp is None:
            with session_scope(db_engine) as session:
                max_ts = session.scalar(select(func.max(Transaction.timestamp)))
            if max_ts is not None:
                base_timestamp = max_ts + timedelta(minutes=1)
            else:
                base_timestamp = resolved_settings.generator.start_date + timedelta(days=31)

        return cls(
            accounts=accounts,
            devices=devices,
            account_devices=account_devices,
            base_timestamp=base_timestamp,
            seed=seed,
            settings=resolved_settings,
        )

    def _advance_time(self, min_seconds: int = 5, max_seconds: int = 45) -> datetime:
        step = int(self.rng.integers(min_seconds, max_seconds + 1))
        self._cursor = self._cursor + timedelta(seconds=step)
        return self._cursor

    def _device_for(self, account_id: str) -> str:
        if account_id in self._known_device:
            return self._known_device[account_id]
        return str(self._all_devices[int(self.rng.integers(len(self._all_devices)))])

    def _pick_distinct(self, pool: list[str], k: int) -> list[str]:
        idx = self.rng.choice(len(pool), size=min(k, len(pool)), replace=False)
        return [pool[int(i)] for i in idx]

    def _make_normal(self) -> SimulatedTransaction:
        sender = str(self._personal_ids[int(self.rng.integers(len(self._personal_ids)))])
        use_pos = bool(self.rng.random() < 0.25) and bool(self._merchant_ids)

        if use_pos:
            receiver = str(self._merchant_ids[int(self.rng.integers(len(self._merchant_ids)))])
            if receiver == sender:
                receiver = next(acc for acc in self._personal_ids if acc != sender)
            amount_minor = int(self.rng.integers(5_000, 95_000))
            payload = {
                "timestamp": self._advance_time(10, 60).isoformat(),
                "sender_account_id": sender,
                "receiver_account_id": receiver,
                "device_id": None,
                "amount_minor": amount_minor,
                "currency": self.settings.generator.currency,
                "channel": Channel.POS.value,
                "transaction_type": TransactionType.CARD_PURCHASE.value,
                "ip_address": None,
                "geo_city": "İstanbul",
                "geo_country": self.settings.features.home_country,
            }
        else:
            candidates = [acc for acc in self._personal_ids if acc != sender]
            receiver = str(candidates[int(self.rng.integers(len(candidates)))])
            amount_minor = int(self.rng.integers(10_000, 180_000))
            ip_last = int(self.rng.integers(2, 240))
            payload = {
                "timestamp": self._advance_time(10, 60).isoformat(),
                "sender_account_id": sender,
                "receiver_account_id": receiver,
                "device_id": self._device_for(sender),
                "amount_minor": amount_minor,
                "currency": self.settings.generator.currency,
                "channel": Channel.MOBILE_APP.value,
                "transaction_type": TransactionType.TRANSFER.value,
                "ip_address": f"85.105.12.{ip_last}",
                "geo_city": "Ankara",
                "geo_country": self.settings.features.home_country,
            }
        return SimulatedTransaction(scenario="NORMAL", payload=payload)

    def _make_fraud_ring(self) -> list[SimulatedTransaction]:
        # Yeni açılmış (genç, KYC=1) 4 halka hesabı ortak cihaz + /24 IP bloğu paylaşır
        ring_id = int(self.rng.integers(100, 900))
        members = [f"ACC-9{ring_id}{i:02d}" for i in range(4)]
        shared_dev = str(self._emulator_devices[int(self.rng.integers(len(self._emulator_devices)))])
        ip_subnet = f"185.220.{int(self.rng.integers(10, 200))}"
        amount = int(self.rng.integers(3_000_000, 6_500_000))  # 30.000 - 65.000 TRY
        txs: list[SimulatedTransaction] = []

        # 2 tur dairesel transfer (ortak cihaz + ortak /24 IP + kısa aralıklar)
        for round_idx in range(2):
            for i in range(len(members)):
                sender = members[i]
                receiver = members[(i + 1) % len(members)]
                amount = max(200_000, int(amount * 0.97))
                payload = {
                    "timestamp": self._advance_time(20, 90).isoformat(),
                    "sender_account_id": sender,
                    "receiver_account_id": receiver,
                    "device_id": shared_dev,
                    "amount_minor": amount,
                    "currency": self.settings.generator.currency,
                    "channel": Channel.MOBILE_APP.value,
                    "transaction_type": TransactionType.TRANSFER.value,
                    "ip_address": f"{ip_subnet}.{10 + i + round_idx}",
                    "geo_city": "İstanbul",
                    "geo_country": self.settings.features.home_country,
                }
                txs.append(SimulatedTransaction(scenario="FRAUD_RING", payload=payload))
        return txs

    def _make_account_takeover(self) -> list[SimulatedTransaction]:
        victim = str(self._personal_ids[int(self.rng.integers(len(self._personal_ids)))])
        known = self._known_device.get(victim)
        foreign_devs = [d for d in self._emulator_devices if d != known] or self._all_devices
        attacker_dev = str(foreign_devs[int(self.rng.integers(len(foreign_devs)))])
        drop = f"ACC-98{int(self.rng.integers(1000, 9999))}"

        # Gece saatine denk gelmesi için tarih saatini 02:xx'e hizala (kronolojiyi koruyarak)
        stamp = self._advance_time(15, 60)
        if not (self.settings.features.night_start_hour <= stamp.hour < self.settings.features.night_end_hour):
            next_night = stamp.replace(hour=2, minute=int(self.rng.integers(5, 55)), second=0)
            if next_night <= stamp:
                next_night += timedelta(days=1)
            self._cursor = next_night
            stamp = next_night

        amount_minor = int(self.rng.integers(3_200_000, 5_500_000))  # 32.000 - 55.000 TRY
        payload = {
            "timestamp": stamp.isoformat(),
            "sender_account_id": victim,
            "receiver_account_id": drop,
            "device_id": attacker_dev,
            "amount_minor": amount_minor,
            "currency": self.settings.generator.currency,
            "channel": Channel.INTERNET_BANKING.value,
            "transaction_type": TransactionType.TRANSFER.value,
            "ip_address": f"185.199.110.{int(self.rng.integers(10, 200))}",
            "geo_city": "Amsterdam",
            "geo_country": "NL",
        }
        return [SimulatedTransaction(scenario="ACCOUNT_TAKEOVER", payload=payload)]

    def _make_card_testing(self) -> list[SimulatedTransaction]:
        sender = str(self._personal_ids[int(self.rng.integers(len(self._personal_ids)))])
        merchants = [m for m in self._merchant_ids if m != sender] or [
            acc for acc in self._personal_ids if acc != sender
        ]
        merchant = str(merchants[int(self.rng.integers(len(merchants)))])
        known = self._known_device.get(sender)
        foreign_devs = [d for d in self._emulator_devices if d != known] or self._all_devices
        dev = str(foreign_devs[int(self.rng.integers(len(foreign_devs)))])
        ip_addr = f"91.198.174.{int(self.rng.integers(10, 200))}"
        txs: list[SimulatedTransaction] = []

        for _ in range(10):
            txs.append(
                SimulatedTransaction(
                    scenario="CARD_TESTING",
                    payload={
                        "timestamp": self._advance_time(5, 20).isoformat(),
                        "sender_account_id": sender,
                        "receiver_account_id": merchant,
                        "device_id": dev,
                        "amount_minor": int(self.rng.integers(200, 1_500)),  # 2 - 15 TRY deneme
                        "currency": self.settings.generator.currency,
                        "channel": Channel.ECOMMERCE.value,
                        "transaction_type": TransactionType.CARD_PURCHASE.value,
                        "ip_address": ip_addr,
                        "geo_city": "Frankfurt",
                        "geo_country": "DE",
                    },
                )
            )

        txs.append(
            SimulatedTransaction(
                scenario="CARD_TESTING",
                payload={
                    "timestamp": self._advance_time(10, 45).isoformat(),
                    "sender_account_id": sender,
                    "receiver_account_id": merchant,
                    "device_id": dev,
                    "amount_minor": int(self.rng.integers(1_800_000, 3_500_000)),  # 18.000 - 35.000 TRY
                    "currency": self.settings.generator.currency,
                    "channel": Channel.ECOMMERCE.value,
                    "transaction_type": TransactionType.CARD_PURCHASE.value,
                    "ip_address": ip_addr,
                    "geo_city": "Frankfurt",
                    "geo_country": "DE",
                },
            )
        )
        return txs

    def build_stream(
        self,
        count: int = 20,
        *,
        fraud_ratio: float | None = None,
        scenario: str = "ALL",
    ) -> list[SimulatedTransaction]:
        """Normal ve fraud senaryolarından oluşan kronolojik işlem listesi üretir."""
        normalized = str(scenario).upper()
        if normalized not in VALID_SIMULATOR_SCENARIOS:
            raise ValueError(
                f"Geçersiz simülasyon senaryosu: {scenario!r}. "
                f"Beklenenler: {', '.join(VALID_SIMULATOR_SCENARIOS)}"
            )
        if count <= 0:
            return []
        if normalized != "ALL":
            return self._build_single_scenario(normalized, count)

        ratio = self.settings.api.simulator_fraud_ratio if fraud_ratio is None else fraud_ratio
        n_fraud_target = min(count, max(0, round(count * ratio)))
        n_normal = count - n_fraud_target
        normal_before = n_normal // 2
        normal_after = n_normal - normal_before

        stream: list[SimulatedTransaction] = [self._make_normal() for _ in range(normal_before)]
        if n_fraud_target > 0:
            stream.extend(self._rebuild_chronological_frauds(n_fraud_target))
        stream.extend(self._make_normal() for _ in range(normal_after))
        return stream[:count]

    def _build_single_scenario(self, scenario: str, count: int) -> list[SimulatedTransaction]:
        if scenario == "NORMAL":
            return [self._make_normal() for _ in range(count)]
        builder = {
            "FRAUD_RING": self._make_fraud_ring,
            "ACCOUNT_TAKEOVER": self._make_account_takeover,
            "CARD_TESTING": self._make_card_testing,
        }[scenario]
        events: list[SimulatedTransaction] = []
        while len(events) < count:
            events.extend(builder())
        return events[:count]

    def _rebuild_chronological_frauds(self, target: int) -> list[SimulatedTransaction]:
        events: list[SimulatedTransaction] = []
        generators = (self._make_fraud_ring, self._make_account_takeover, self._make_card_testing)
        idx = 0
        while len(events) < target:
            batch = generators[idx % len(generators)]()
            for item in batch:
                if len(events) < target:
                    events.append(item)
            idx += 1
        return events

    async def run(
        self,
        *,
        count: int = 20,
        tps: float | None = None,
        fraud_ratio: float | None = None,
        base_url: str | None = None,
        app: FastAPI | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> SimulationSummary:
        """Üretilen işlemleri `POST /transactions/score` ucuna asenkron olarak gönderir."""
        resolved_tps = tps or self.settings.api.simulator_tps
        interval = 1.0 / resolved_tps
        stream = self.build_stream(count, fraud_ratio=fraud_ratio)

        owns_client = client is None
        if client is None:
            if app is not None:
                client = httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://aegisgraph.local",
                )
            else:
                url = base_url or f"http://{self.settings.api.host}:{self.settings.api.port}"
                client = httpx.AsyncClient(base_url=url, timeout=10.0)

        succeeded = 0
        failed = 0
        alerts_triggered = 0
        latencies: list[float] = []
        by_scenario: dict[str, dict[str, int]] = {}
        responses: list[dict[str, Any]] = []

        try:
            for idx, item in enumerate(stream):
                t_start = time.perf_counter()
                stats = by_scenario.setdefault(item.scenario, {"sent": 0, "alerts": 0, "failed": 0})
                stats["sent"] += 1

                resp = await client.post("/transactions/score", json=item.payload)
                if resp.status_code == 200:
                    body = resp.json()
                    succeeded += 1
                    latencies.append(float(body.get("latency_ms", 0.0)))
                    if body.get("is_alert"):
                        alerts_triggered += 1
                        stats["alerts"] += 1
                    responses.append({"scenario": item.scenario, **body})
                else:
                    failed += 1
                    stats["failed"] += 1
                    logger.warning("Simülasyon isteği başarısız (%d): %s", resp.status_code, resp.text)

                if idx + 1 < len(stream):
                    elapsed = time.perf_counter() - t_start
                    wait_sec = interval - elapsed
                    if wait_sec > 0:
                        await asyncio.sleep(wait_sec)
        finally:
            if owns_client:
                await client.aclose()

        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        logger.info(
            "Simülasyon tamamlandı: gönderilen=%d başarılı=%d başarısız=%d alarm=%d ort_gecikme=%.2f ms",
            len(stream),
            succeeded,
            failed,
            alerts_triggered,
            avg_latency,
        )
        return SimulationSummary(
            total_sent=len(stream),
            succeeded=succeeded,
            failed=failed,
            alerts_triggered=alerts_triggered,
            avg_latency_ms=round(avg_latency, 3),
            by_scenario=by_scenario,
            responses=responses,
        )


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="AegisGraph canlı işlem trafiği simülatörü")
    parser.add_argument("--count", type=int, default=20, help="Gönderilecek işlem sayısı")
    parser.add_argument("--tps", type=float, default=settings.api.simulator_tps, help="Saniyedeki işlem hızı")
    parser.add_argument("--fraud-ratio", type=float, default=settings.api.simulator_fraud_ratio)
    parser.add_argument("--seed", type=int, default=settings.generator.seed)
    parser.add_argument(
        "--url",
        default=f"http://{settings.api.host}:{settings.api.port}",
        help="Hedef FastAPI sunucu adresi",
    )
    args = parser.parse_args(argv)

    sim = TrafficSimulator.from_database(seed=args.seed, settings=settings)
    summary = asyncio.run(
        sim.run(count=args.count, tps=args.tps, fraud_ratio=args.fraud_ratio, base_url=args.url)
    )
    print(
        f"Simülasyon özeti: {summary.succeeded}/{summary.total_sent} işlem skorlandı, "
        f"{summary.alerts_triggered} alarm üretildi (ort. {summary.avg_latency_ms:.2f} ms)."
    )
    for scenario, stats in summary.by_scenario.items():
        print(f"  - {scenario}: gönderilen={stats['sent']}, alarm={stats['alerts']}, hata={stats['failed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
