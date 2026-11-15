"""Davranışsal, hız (velocity), anomali ve cihaz/konum yeniliği özellikleri — point-in-time.

`features(...)` yalnızca daha önce `update(...)` edilmiş işlemleri görür. Pencere sayımları
mevcut işlemi **hariç** tutar ("bu işlemden önceki son 10 dakikada kaç işlem vardı?").
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field

from src.database.models import Channel, DeviceType, MerchantCategory, TransactionType

DAY_SECONDS = 86_400

CHANNEL_CODES = {c.value: i for i, c in enumerate(Channel)}
TX_TYPE_CODES = {t.value: i for i, t in enumerate(TransactionType)}
DEVICE_TYPE_CODES = {d.value: i for i, d in enumerate(DeviceType)}
MERCHANT_CATEGORY_CODES = {m.value: i for i, m in enumerate(MerchantCategory)}

TABULAR_FEATURES: tuple[str, ...] = (
    "amount",
    "log_amount",
    "hour",
    "is_night_tx",
    "account_age_days",
    "receiver_account_age_days",
    "sender_kyc_level",
    "receiver_kyc_level",
    "receiver_is_merchant",
    "receiver_merchant_category",
    "channel_code",
    "tx_type_code",
    "device_type_code",
    "device_is_emulator",
    "tx_count_10m",
    "tx_count_1h",
    "tx_count_24h",
    "amount_sum_10m",
    "amount_sum_1h",
    "amount_sum_24h",
    "sender_unique_receivers_24h",
    "sender_tx_count_total",
    "seconds_since_last_tx",
    "amount_to_user_avg_ratio",
    "device_use_count_for_account",
    "is_new_device_for_account",
    "ip_use_count_for_account",
    "is_foreign_ip",
    "is_foreign_or_new_ip",
)

NAN = float("nan")


@dataclass(frozen=True, slots=True)
class AccountProfile:
    opened_ts: float
    kyc_level: int
    is_merchant: bool
    merchant_category_code: float  # NaN → üye işyeri değil


@dataclass(frozen=True, slots=True)
class DeviceProfile:
    type_code: int
    is_emulator: bool


@dataclass(slots=True)
class _SenderHistory:
    recent: deque[tuple[float, float, str]] = field(default_factory=deque)  # (ts, tutar, alıcı)
    total_count: int = 0
    total_amount: float = 0.0
    last_ts: float = NAN


class TabularFeatureState:
    def __init__(
        self,
        *,
        accounts: Mapping[str, AccountProfile],
        devices: Mapping[str, DeviceProfile],
        home_country: str = "TR",
        night_hours: tuple[int, int] = (1, 5),
        velocity_windows: tuple[int, int, int] = (600, 3_600, DAY_SECONDS),
    ) -> None:
        self._accounts = accounts
        self._devices = devices
        self._home = home_country
        self._night = night_hours
        self._w10m, self._w1h, self._w24h = velocity_windows
        self._history: dict[str, _SenderHistory] = {}
        self._device_use: dict[tuple[str, str], int] = {}
        self._ip_use: dict[tuple[str, str], int] = {}

    def features(
        self,
        *,
        ts: float,
        hour: int,
        sender: str,
        receiver: str,
        device: str | None,
        ip: str | None,
        amount: float,
        channel: str,
        tx_type: str,
        geo_country: str | None,
    ) -> dict[str, float]:
        s_prof, r_prof = self._accounts.get(sender), self._accounts.get(receiver)
        d_prof = self._devices.get(device) if device else None
        hist = self._history.get(sender)

        c10 = c1h = c24 = 0
        a10 = a1h = a24 = 0.0
        receivers: set[str] = set()
        if hist is not None:
            cutoff = ts - self._w24h
            while hist.recent and hist.recent[0][0] < cutoff:
                hist.recent.popleft()
            for t, amt, rcv in hist.recent:
                age = ts - t
                c24 += 1
                a24 += amt
                receivers.add(rcv)
                if age <= self._w1h:
                    c1h += 1
                    a1h += amt
                    if age <= self._w10m:
                        c10 += 1
                        a10 += amt

        dev_uses = self._device_use.get((sender, device), 0) if device else NAN
        ip_uses = self._ip_use.get((sender, ip), 0) if ip else NAN
        is_foreign = float(bool(geo_country) and geo_country != self._home)
        start, end = self._night

        return {
            "amount": amount,
            "log_amount": math.log1p(amount),
            "hour": float(hour),
            "is_night_tx": float(start <= hour < end),
            "account_age_days": (ts - s_prof.opened_ts) / DAY_SECONDS if s_prof else NAN,
            "receiver_account_age_days": (ts - r_prof.opened_ts) / DAY_SECONDS if r_prof else NAN,
            "sender_kyc_level": float(s_prof.kyc_level) if s_prof else NAN,
            "receiver_kyc_level": float(r_prof.kyc_level) if r_prof else NAN,
            "receiver_is_merchant": float(r_prof.is_merchant) if r_prof else NAN,
            "receiver_merchant_category": r_prof.merchant_category_code if r_prof else NAN,
            "channel_code": float(CHANNEL_CODES.get(channel, -1)),
            "tx_type_code": float(TX_TYPE_CODES.get(tx_type, -1)),
            "device_type_code": float(d_prof.type_code) if d_prof else NAN,
            "device_is_emulator": float(d_prof.is_emulator) if d_prof else NAN,
            "tx_count_10m": float(c10),
            "tx_count_1h": float(c1h),
            "tx_count_24h": float(c24),
            "amount_sum_10m": a10,
            "amount_sum_1h": a1h,
            "amount_sum_24h": a24,
            "sender_unique_receivers_24h": float(len(receivers)),
            "sender_tx_count_total": float(hist.total_count) if hist else 0.0,
            "seconds_since_last_tx": ts - hist.last_ts if hist else NAN,
            "amount_to_user_avg_ratio": (
                amount / (hist.total_amount / hist.total_count) if hist and hist.total_count else NAN
            ),
            "device_use_count_for_account": float(dev_uses),
            "is_new_device_for_account": float(dev_uses == 0) if device else 0.0,
            "ip_use_count_for_account": float(ip_uses),
            "is_foreign_ip": is_foreign,
            "is_foreign_or_new_ip": float(bool(ip) and (is_foreign == 1.0 or ip_uses == 0)),
        }

    def update(
        self, *, ts: float, sender: str, receiver: str, device: str | None, ip: str | None, amount: float
    ) -> None:
        hist = self._history.setdefault(sender, _SenderHistory())
        hist.recent.append((ts, amount, receiver))
        hist.total_count += 1
        hist.total_amount += amount
        hist.last_ts = ts
        if device:
            key = (sender, device)
            self._device_use[key] = self._device_use.get(key, 0) + 1
        if ip:
            key = (sender, ip)
            self._ip_use[key] = self._ip_use.get(key, 0) + 1
