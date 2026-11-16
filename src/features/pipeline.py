"""Tabular + graf özelliklerini birleştiren, batch ve online için ortak `FeatureExtractor`.

Garantiler:
  1. **Etiket izolasyonu:** girdi işlemi `TX_INPUT_COLUMNS` dışındaki her alandan arındırılır;
     `GROUND_TRUTH_COLUMNS` ve ID'ler özellik listesine fiziksel olarak giremez (import anında doğrulanır).
  2. **Zaman sızıntısı yok:** işlemler kronolojik işlenir; bir işlemin özellikleri yalnızca kendisinden
     önce işlenmiş işlemlerden hesaplanır. Kronolojiyi bozan çağrı `ValueError` fırlatır.
  3. **Train/serve tutarlılığı:** `fit_transform` (batch) satır satır `transform_one` (online) çağırır;
     iki yol arasında kod farkı yoktur.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config.settings import FeatureSettings, get_settings
from src.database.models import GROUND_TRUTH_COLUMNS, AccountType
from src.features.graph_features import GRAPH_FEATURES, GraphFeatureState
from src.features.tabular_features import (
    DEVICE_TYPE_CODES,
    MERCHANT_CATEGORY_CODES,
    TABULAR_FEATURES,
    AccountProfile,
    DeviceProfile,
    TabularFeatureState,
)

FEATURE_COLUMNS: tuple[str, ...] = (*TABULAR_FEATURES, *GRAPH_FEATURES)

# Online bir işlemden okunmasına izin verilen tek alanlar
TX_INPUT_COLUMNS: tuple[str, ...] = (
    "transaction_id",
    "timestamp",
    "sender_account_id",
    "receiver_account_id",
    "device_id",
    "amount_minor",
    "channel",
    "transaction_type",
    "ip_address",
    "geo_country",
)

FORBIDDEN_FEATURES: frozenset[str] = frozenset(
    {col for cols in GROUND_TRUTH_COLUMNS.values() for col in cols}
    | {"transaction_id", "sender_account_id", "receiver_account_id", "device_id", "account_id", "iban"}
)

_LABELS = {col for cols in GROUND_TRUTH_COLUMNS.values() for col in cols}
if FORBIDDEN_FEATURES & set(FEATURE_COLUMNS) or _LABELS & set(TX_INPUT_COLUMNS):
    raise RuntimeError("Yasaklı kolon özellik/girdi listesine sızdı")
if len(set(FEATURE_COLUMNS)) != len(FEATURE_COLUMNS):
    raise RuntimeError("Özellik isimleri tekil olmalı")


def _epoch_seconds(value: Any) -> float:
    return pd.Timestamp(value).value / 1e9


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


class FeatureExtractor:
    """Durumlu (stateful) özellik motoru. Bir örnek tek bir kronolojik işlem akışını temsil eder."""

    def __init__(
        self,
        accounts: pd.DataFrame,
        devices: pd.DataFrame,
        settings: FeatureSettings | None = None,
    ) -> None:
        self.settings = settings or get_settings().features
        account_profiles = {
            row.account_id: AccountProfile(
                opened_ts=_epoch_seconds(row.opened_at),
                kyc_level=int(row.kyc_level),
                is_merchant=row.account_type == AccountType.MERCHANT.value,
                merchant_category_code=float(MERCHANT_CATEGORY_CODES[row.merchant_category])
                if isinstance(row.merchant_category, str)
                else math.nan,
            )
            # is_fraudster / fraud_case_id bilinçli olarak okunmaz
            for row in accounts[["account_id", "opened_at", "kyc_level", "account_type", "merchant_category"]]
            .itertuples(index=False)
        }
        device_profiles = {
            row.device_id: DeviceProfile(DEVICE_TYPE_CODES[row.device_type], bool(row.is_emulator))
            for row in devices[["device_id", "device_type", "is_emulator"]].itertuples(index=False)
        }
        merchants = {acc for acc, prof in account_profiles.items() if prof.is_merchant}

        s = self.settings
        self.tabular = TabularFeatureState(
            accounts=account_profiles,
            devices=device_profiles,
            home_country=s.home_country,
            night_hours=(s.night_start_hour, s.night_end_hour),
            velocity_windows=s.velocity_windows_seconds,
        )
        self.graph = GraphFeatureState(
            merchant_accounts=merchants,
            flow_window_seconds=s.flow_window_seconds,
            max_cycle_hops=s.max_cycle_hops,
        )
        self._last_ts = -math.inf
        self.n_processed = 0

    @property
    def last_timestamp(self) -> float | None:
        """Şimdiye kadar duruma işlenmiş en son işlem zaman damgası (epoch saniye)."""
        return None if math.isinf(self._last_ts) else self._last_ts

    # ------------------------------------------------------------------ API
    @staticmethod
    def sanitize(tx: Mapping[str, Any]) -> dict[str, Any]:
        """Yalnızca izinli girdi alanlarını bırakır (etiketler burada fiziksel olarak düşer)."""
        missing = {"timestamp", "sender_account_id", "receiver_account_id", "amount_minor"} - set(tx)
        if missing:
            raise KeyError(f"İşlemde zorunlu alan eksik: {sorted(missing)}")
        return {col: (None if _is_missing(tx.get(col)) else tx.get(col)) for col in TX_INPUT_COLUMNS}

    def transform_one(self, tx: Mapping[str, Any], *, update_state: bool = True) -> dict[str, float]:
        """Tek işlemin özelliklerini, yalnızca geçmişi kullanarak hesaplar.

        `update_state=True` (varsayılan): işlem hesaplamadan SONRA duruma eklenir (canlı akış).
        `update_state=False`: "what-if" skorlama; durum değişmez.
        """
        clean = self.sanitize(tx)
        stamp = pd.Timestamp(clean["timestamp"])
        ts = stamp.value / 1e9
        if ts < self._last_ts:
            raise ValueError("İşlemler kronolojik sırada verilmeli (zaman sızıntısı koruması)")

        sender, receiver = clean["sender_account_id"], clean["receiver_account_id"]
        device, ip = clean["device_id"], clean["ip_address"]
        amount = int(clean["amount_minor"]) / 100

        feats = self.tabular.features(
            ts=ts, hour=stamp.hour, sender=sender, receiver=receiver, device=device, ip=ip,
            amount=amount, channel=clean["channel"], tx_type=clean["transaction_type"],
            geo_country=clean["geo_country"],
        )
        feats.update(self.graph.features(ts, sender, receiver, device, ip))

        if update_state:
            self._last_ts = ts
            self.tabular.update(ts=ts, sender=sender, receiver=receiver, device=device, ip=ip, amount=amount)
            self.graph.update(ts, sender, receiver, device, ip, amount)
            self.n_processed += 1
        return {col: float(feats[col]) for col in FEATURE_COLUMNS}

    def fit_transform(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Tüm akışı kronolojik işler; `transaction_id` indeksli özellik matrisi döndürür."""
        cols = [c for c in TX_INPUT_COLUMNS if c in transactions.columns]
        ordered = transactions[cols].sort_values(["timestamp", "transaction_id"], kind="stable")
        rows = [self.transform_one(rec) for rec in ordered.to_dict("records")]
        frame = pd.DataFrame(rows, columns=list(FEATURE_COLUMNS), dtype=np.float64)
        frame.index = pd.Index(ordered["transaction_id"].to_numpy(), name="transaction_id")
        return frame

    @staticmethod
    def to_frame(features: Mapping[str, float] | list[Mapping[str, float]]) -> pd.DataFrame:
        rows = [features] if isinstance(features, Mapping) else features
        return pd.DataFrame(rows, columns=list(FEATURE_COLUMNS), dtype=np.float64)

    # ------------------------------------------------------------ kalıcılık
    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        return path

    @staticmethod
    def load(path: Path) -> FeatureExtractor:
        obj = joblib.load(path)
        if not isinstance(obj, FeatureExtractor):
            raise TypeError(f"{path} bir FeatureExtractor değil")
        return obj
