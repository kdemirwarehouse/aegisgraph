"""SHAP (TreeExplainer) tabanlı, insan-okunur risk açıklamaları.

Her yüksek riskli işlem için tahmini en çok YUKARI çeken ilk k faktör döndürülür. Çıktı
JSON-serileştirilebilir (`Explanation.to_dict`) ve 4. gün soruşturma ajanına doğrudan kanıt olarak verilir.
SHAP değerleri log-odds uzayındadır (pozitif = riski artırır).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from src.database.models import Channel, DeviceType, MerchantCategory, TransactionType
from src.features.graph_features import GRAPH_FEATURES


def _enum_name(enum_cls: type) -> Callable[[float], str]:
    members = list(enum_cls)
    return lambda v: members[int(v)].value if 0 <= int(v) < len(members) else "bilinmiyor"


def _duration(seconds: float) -> str:
    if seconds < 120:
        return f"{seconds:.0f} saniye"
    if seconds < 7_200:
        return f"{seconds / 60:.0f} dakika"
    if seconds < 172_800:
        return f"{seconds / 3_600:.1f} saat"
    return f"{seconds / 86_400:.1f} gün"


_BINARY: dict[str, tuple[str, str]] = {
    "is_night_tx": ("İşlem gece 01:00-05:00 arasında yapıldı", "İşlem gece saatinde değil"),
    "is_new_device_for_account": ("Gönderici bu cihazı ilk kez kullanıyor", "Cihaz göndericinin bilinen cihazı"),
    "is_foreign_ip": ("İşlem yurt dışı IP adresinden geldi", "İşlem yurt içi IP adresinden geldi"),
    "is_foreign_or_new_ip": ("IP adresi yabancı veya bu hesap için ilk kez görülüyor",
                             "IP adresi hesabın bilinen IP'si"),
    "has_reciprocal_flow": ("Son 24 saatte alıcıdan göndericiye ters yönlü para akışı var",
                            "Ters yönlü para akışı yok"),
    "device_is_emulator": ("Cihaz emülatör/sanal cihaz olarak işaretli", "Cihaz fiziksel bir cihaz"),
    "receiver_is_merchant": ("Alıcı bir üye işyeri", "Alıcı bir üye işyeri değil (kişi/kurum hesabı)"),
}

_VALUE: dict[str, Callable[[float], str]] = {
    "amount": lambda v: f"İşlem tutarı {v:,.2f} TRY",
    "log_amount": lambda v: f"İşlem tutarı büyüklüğü yüksek ({math.expm1(v):,.2f} TRY)",
    "hour": lambda v: f"İşlem saati {v:02.0f}:00",
    "account_age_days": lambda v: f"Gönderici hesap yalnızca {v:.0f} günlük",
    "receiver_account_age_days": lambda v: f"Alıcı hesap yalnızca {v:.0f} günlük",
    "sender_kyc_level": lambda v: f"Gönderici KYC seviyesi {v:.0f}/3",
    "receiver_kyc_level": lambda v: f"Alıcı KYC seviyesi {v:.0f}/3",
    "receiver_merchant_category": lambda v: f"Alıcı işyeri kategorisi {_enum_name(MerchantCategory)(v)}",
    "channel_code": lambda v: f"İşlem kanalı {_enum_name(Channel)(v)}",
    "tx_type_code": lambda v: f"İşlem türü {_enum_name(TransactionType)(v)}",
    "device_type_code": lambda v: f"Cihaz tipi {_enum_name(DeviceType)(v)}",
    "tx_count_10m": lambda v: f"Göndericinin son 10 dakikada {v:.0f} işlemi var",
    "tx_count_1h": lambda v: f"Göndericinin son 1 saatte {v:.0f} işlemi var",
    "tx_count_24h": lambda v: f"Göndericinin son 24 saatte {v:.0f} işlemi var",
    "amount_sum_10m": lambda v: f"Göndericinin son 10 dakikadaki toplam tutarı {v:,.2f} TRY",
    "amount_sum_1h": lambda v: f"Göndericinin son 1 saatteki toplam tutarı {v:,.2f} TRY",
    "amount_sum_24h": lambda v: f"Göndericinin son 24 saatteki toplam tutarı {v:,.2f} TRY",
    "sender_unique_receivers_24h": lambda v: f"Gönderici son 24 saatte {v:.0f} farklı alıcıya para gönderdi",
    "sender_tx_count_total": lambda v: f"Göndericinin geçmiş işlem sayısı {v:.0f}",
    "seconds_since_last_tx": lambda v: f"Göndericinin bir önceki işleminden bu yana {_duration(v)} geçti",
    "amount_to_user_avg_ratio": lambda v: f"Tutar, göndericinin ortalama işlem tutarının {v:.1f} katı",
    "device_use_count_for_account": lambda v: f"Bu cihaz gönderici tarafından daha önce {v:.0f} kez kullanıldı",
    "ip_use_count_for_account": lambda v: f"Bu IP gönderici tarafından daha önce {v:.0f} kez kullanıldı",
    "sender_in_degree": lambda v: f"Gönderici şimdiye kadar {v:.0f} farklı hesaptan para aldı",
    "sender_out_degree": lambda v: f"Gönderici şimdiye kadar {v:.0f} farklı hesaba para gönderdi",
    "receiver_in_degree": lambda v: f"Alıcı şimdiye kadar {v:.0f} farklı hesaptan para aldı",
    "receiver_out_degree": lambda v: f"Alıcı şimdiye kadar {v:.0f} farklı hesaba para gönderdi",
    "sender_in_out_ratio": lambda v: f"Göndericinin gelen/giden karşı taraf oranı {v:.2f} (mule hub sinyali)",
    "receiver_in_out_ratio": lambda v: f"Alıcının gelen/giden karşı taraf oranı {v:.2f}",
    "sender_pagerank": lambda v: f"Göndericinin para akışı ağındaki PageRank merkeziliği {v:.2f} (ortalama 1)",
    "receiver_pagerank": lambda v: f"Alıcının para akışı ağındaki PageRank merkeziliği {v:.2f} (ortalama 1)",
    "sender_clustering": lambda v: f"Göndericinin P2P ağındaki kümelenme katsayısı {v:.2f}",
    "receiver_clustering": lambda v: f"Alıcının P2P ağındaki kümelenme katsayısı {v:.2f}",
    "shared_device_account_count": lambda v: f"Bu cihaz {v:.0f} farklı hesap tarafından kullanılıyor",
    "shared_ip_account_count": lambda v: f"Bu IP bloğunu (/24) {v:.0f} farklı hesap kullanıyor",
    "receiver_unique_senders_24h": lambda v: f"Alıcıya son 24 saatte {v:.0f} farklı hesaptan para geldi (fan-in)",
    "short_cycle_risk": lambda v: (
        f"Para, alıcıdan göndericiye {round(2 / v):.0f} hesaplık bir döngüyle geri dönüyor"
        if v > 0 else "Kısa para döngüsü tespit edilmedi"
    ),
}


# Eksik değerin kendisi anlam taşıdığında kullanılan açıklamalar
_MISSING: dict[str, str] = {
    "receiver_merchant_category": "Alıcı bir üye işyeri değil (kişi/kurum hesabına transfer)",
    "device_type_code": "İşlemde dijital cihaz yok (kart-mevcut POS işlemi)",
    "device_is_emulator": "İşlemde dijital cihaz yok (kart-mevcut POS işlemi)",
    "device_use_count_for_account": "İşlemde dijital cihaz yok (kart-mevcut POS işlemi)",
    "ip_use_count_for_account": "İşlemde IP bilgisi yok (POS işlemi)",
    "amount_to_user_avg_ratio": "Göndericinin karşılaştırılacak geçmiş işlemi yok (ilk işlem)",
    "seconds_since_last_tx": "Bu, göndericinin gözlemlenen ilk işlemi",
    "account_age_days": "Gönderici hesap bilgisi bulunamadı",
    "receiver_account_age_days": "Alıcı hesap bilgisi bulunamadı",
}


def describe(feature: str, value: float | None) -> str:
    """Bir özellik değerinin Türkçe, insan-okunur açıklaması."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return _MISSING.get(feature, f"{feature}: veri yok")
    if feature in _BINARY:
        yes, no = _BINARY[feature]
        return yes if value >= 0.5 else no
    if feature in _VALUE:
        return _VALUE[feature](value)
    return f"{feature} = {value:.4g}"


def has_description(feature: str) -> bool:
    return feature in _BINARY or feature in _VALUE


@dataclass(frozen=True)
class FactorContribution:
    feature: str
    value: float | None
    shap_value: float
    family: str  # "graph" | "tabular"
    description: str


@dataclass(frozen=True)
class Explanation:
    transaction_id: str | None
    risk_score: float
    base_value: float
    top_factors: list[FactorContribution]
    graph_contribution_share: float  # pozitif SHAP kütlesinin graf özelliklerinden gelen payı

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        reasons = "; ".join(f"{f.description} (+{f.shap_value:.2f})" for f in self.top_factors)
        return f"Risk {self.risk_score:.3f} — {reasons}"


class RiskExplainer:
    def __init__(self, booster: lgb.Booster, feature_names: Sequence[str]) -> None:
        self.feature_names = list(feature_names)
        self._explainer = shap.TreeExplainer(booster)
        ev = self._explainer.expected_value
        self.base_value = float(np.ravel(ev)[-1])
        self._is_graph = np.array([f in GRAPH_FEATURES for f in self.feature_names])

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        with warnings.catch_warnings():  # shap: LightGBM ikili sınıf çıktı formatı uyarısı
            warnings.simplefilter("ignore", UserWarning)
            values = self._explainer.shap_values(X[self.feature_names])
        if isinstance(values, list):  # eski shap sürümleri: sınıf başına liste
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[..., -1]
        return values

    def explain(
        self,
        X: pd.DataFrame,
        risk_scores: Sequence[float],
        transaction_ids: Sequence[str] | None = None,
        top_k: int = 3,
    ) -> list[Explanation]:
        if len(X) == 0:
            return []
        values = self.shap_values(X)
        raw = X[self.feature_names].to_numpy()
        ids = list(transaction_ids) if transaction_ids is not None else [None] * len(X)
        out = []
        for i in range(len(X)):
            row = values[i]
            positive = np.clip(row, 0, None)
            total_pos = positive.sum()
            order = [j for j in np.argsort(-row)[:top_k] if row[j] > 0]
            factors = [
                FactorContribution(
                    feature=self.feature_names[j],
                    value=None if math.isnan(raw[i, j]) else float(raw[i, j]),
                    shap_value=round(float(row[j]), 4),
                    family="graph" if self._is_graph[j] else "tabular",
                    description=describe(self.feature_names[j], float(raw[i, j])),
                )
                for j in order
            ]
            out.append(Explanation(
                transaction_id=ids[i],
                risk_score=float(risk_scores[i]),
                base_value=self.base_value,
                top_factors=factors,
                graph_contribution_share=float(positive[self._is_graph].sum() / total_pos) if total_pos else 0.0,
            ))
        return out
