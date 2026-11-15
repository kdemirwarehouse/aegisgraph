"""AegisGraph sentetik bankacılık verisi üretim motoru.

Normal müşteri davranışının üzerine dört gerçekçi dolandırıcılık senaryosu enjekte eder:

1. FRAUD_RING        Yeni açılmış, aynı cihazı/IP alt ağını paylaşan hesaplar arasında
                     dakikalar içinde dönen dairesel para akışı (layering).
2. ACCOUNT_TAKEOVER  Kurbanın hesabına daha önce görülmemiş (ve başka kurbanlarda da
                     kullanılan) bir cihaz + yabancı IP ile girilip yüksek tutarlı boşaltma.
3. MONEY_MULE        Çok sayıda kurbandan (APP scam) tek hub hesaba fan-in, ardından
                     saatler içinde cash-out hesaplarına fan-out.
4. CARD_TESTING      Dijital işyerlerinde saniyeler arayla mikro tutarlı denemeler ve
                     ardından büyük tutarlı alışveriş.

Tasarım ilkeleri:
  * Deterministik: aynı `seed` → bit düzeyinde aynı çıktı.
  * Sızıntısız ID'ler: hesap/cihaz ID'leri üretim sırasından bağımsız rastgele permütasyondur
    (fraud hesapları ID aralığından tanınamaz). İşlem ID'leri kronolojiktir.
  * Etiketler (`is_fraud`, `fraud_scenario`, `fraud_case_id`, `is_fraudster`) yalnızca
    değerlendirme içindir; bkz. `GROUND_TRUTH_COLUMNS`.

Kullanım:
    python -m src.generator.synthetic_data                # varsayılan ayarlar
    python -m src.generator.synthetic_data --customers 5000 --days 60 --seed 7
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import numpy as np
import pandas as pd
from faker import Faker

from config.settings import GeneratorSettings, get_settings
from src.database.models import (
    AccountType,
    Channel,
    DeviceType,
    FraudScenario,
    MerchantCategory,
    TransactionType,
    record_fields,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

MINUTE = 60
HOUR = 3_600
DAY = 86_400

# --------------------------------------------------------------------------- #
# Domain sabitleri
# --------------------------------------------------------------------------- #
TR_CITIES: tuple[str, ...] = (
    "İstanbul", "Ankara", "İzmir", "Bursa", "Antalya", "Adana", "Konya", "Gaziantep",
    "Kocaeli", "Kayseri", "Eskişehir", "Trabzon", "Samsun", "Diyarbakır", "Mersin",
)
_CITY_WEIGHTS = np.array([30, 12, 9, 6, 5, 4, 4, 3.5, 3.5, 3, 2.5, 2.5, 2.5, 2.5, 2.5])
CITY_PROBS: np.ndarray = _CITY_WEIGHTS / _CITY_WEIGHTS.sum()

# Gerçek TR banka kodu formatı (5 hane). Veri tamamen sentetiktir.
TR_BANK_CODES: tuple[str, ...] = (
    "00010", "00012", "00015", "00046", "00062", "00064", "00067", "00111", "00134", "00205",
)
TR_ISP_PREFIXES: tuple[str, ...] = (
    "78.160", "78.180", "88.230", "88.240", "85.96", "85.105", "176.40", "176.88", "31.140", "95.10",
)
# (IP öneki, ülke) — VPN / hosting / proxy çıkışlarını temsil eder
FOREIGN_IP_PREFIXES: tuple[tuple[str, str], ...] = (
    ("185.220", "DE"), ("45.155", "NL"), ("91.219", "RU"),
    ("194.165", "RO"), ("5.188", "UA"), ("102.89", "NG"),
)
# Meşru yurt dışı seyahat: otel / roaming / yerel operatör blokları
TRAVEL_IP_PREFIXES: tuple[tuple[str, str], ...] = (
    ("88.198", "DE"), ("62.210", "FR"), ("81.2", "GB"), ("151.0", "IT"),
    ("83.44", "ES"), ("94.66", "GR"), ("72.21", "US"), ("145.53", "NL"),
)

# kategori -> (katalogdaki ağırlık, tipik sepet tutarı TRY)
MERCHANT_PROFILE: dict[MerchantCategory, tuple[float, float]] = {
    MerchantCategory.GROCERY: (0.22, 420.0),
    MerchantCategory.RESTAURANT: (0.15, 380.0),
    MerchantCategory.FUEL: (0.10, 1_100.0),
    MerchantCategory.FASHION: (0.10, 900.0),
    MerchantCategory.ELECTRONICS: (0.07, 4_200.0),
    MerchantCategory.ONLINE_MARKETPLACE: (0.12, 650.0),
    MerchantCategory.PHARMACY: (0.06, 300.0),
    MerchantCategory.UTILITIES: (0.06, 700.0),
    MerchantCategory.TRAVEL: (0.05, 3_200.0),
    MerchantCategory.DIGITAL_GOODS: (0.07, 180.0),
}
BUSINESS_SECTORS: tuple[str, ...] = (
    "Gıda", "İnşaat", "Tekstil", "Lojistik", "Bilişim", "Turizm", "Otomotiv", "Danışmanlık",
    "Mobilya", "Enerji", "Ticaret", "Medikal", "Kırtasiye", "Yapı Market", "Elektronik",
)
CARD_TESTING_CATEGORIES = (MerchantCategory.DIGITAL_GOODS, MerchantCategory.ONLINE_MARKETPLACE)
CASHOUT_CATEGORIES = (MerchantCategory.ELECTRONICS, MerchantCategory.ONLINE_MARKETPLACE)

# Gün içi aktivite dağılımı (00:00 → 23:00)
_HOUR_WEIGHTS = np.array([
    0.6, 0.3, 0.2, 0.15, 0.15, 0.3, 0.8, 1.6, 2.6, 3.4, 3.8, 4.2,
    4.8, 4.6, 4.2, 4.0, 4.2, 4.6, 5.2, 5.6, 5.4, 4.4, 3.0, 1.6,
])
HOUR_PROBS: np.ndarray = _HOUR_WEIGHTS / _HOUR_WEIGHTS.sum()

# Normal işlem türü karışımı: [kart alışverişi, P2P havale, fatura]
_KIND_P_PERSONAL = np.array([0.60, 0.30, 0.10])
_KIND_P_BUSINESS = np.array([0.35, 0.50, 0.15])
_KIND_CARD, _KIND_P2P, _KIND_BILL = 0, 1, 2

_TX_FIELDS: tuple[str, ...] = (
    "ts", "sender", "receiver", "device", "amount_minor", "channel", "tx_type",
    "ip", "geo_city", "geo_country", "scenario", "case_id",
)


# --------------------------------------------------------------------------- #
# IBAN yardımcıları (ISO 13616, mod-97)
# --------------------------------------------------------------------------- #
def make_tr_iban(bank_code: str, account_number: str) -> str:
    """5 haneli banka kodu + 16 haneli hesap no'dan geçerli kontrol haneli TR IBAN üretir."""
    bban = f"{bank_code}0{account_number}"
    if len(bban) != 22 or not bban.isdigit():
        raise ValueError("TR BBAN 22 haneli sayısal olmalı")
    check = 98 - int(f"{bban}292700") % 97  # "TR00" → T=29, R=27
    return f"TR{check:02d}{bban}"


def is_valid_tr_iban(iban: str) -> bool:
    if len(iban) != 26 or not iban.startswith("TR") or not iban[2:].isdigit():
        return False
    return int(f"{iban[4:]}2927{iban[2:4]}") % 97 == 1


# --------------------------------------------------------------------------- #
# Çıktı
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SyntheticDataset:
    accounts: pd.DataFrame
    devices: pd.DataFrame
    account_devices: pd.DataFrame
    transactions: pd.DataFrame
    config: GeneratorSettings

    TABLES: ClassVar[tuple[str, ...]] = ("accounts", "devices", "account_devices", "transactions")

    def summary(self) -> dict[str, Any]:
        tx = self.transactions
        fraud = tx[tx["is_fraud"]]
        return {
            "accounts": len(self.accounts),
            "fraudster_accounts": int(self.accounts["is_fraudster"].sum()),
            "devices": len(self.devices),
            "account_device_links": len(self.account_devices),
            "transactions": len(tx),
            "fraud_transactions": len(fraud),
            "fraud_rate": round(len(fraud) / max(len(tx), 1), 5),
            "fraud_cases": int(fraud["fraud_case_id"].nunique()),
            "fraud_by_scenario": fraud["fraud_scenario"].value_counts().sort_index().to_dict(),
            "total_volume": round(tx["amount_minor"].sum() / 100, 2),
            "fraud_volume": round(fraud["amount_minor"].sum() / 100, 2),
            "period": f"{tx['timestamp'].min()} → {tx['timestamp'].max()}",
        }

    def to_csv(self, out_dir: Path) -> dict[str, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        for name in self.TABLES:
            path = out_dir / f"{name}.csv"
            getattr(self, name).to_csv(path, index=False)
            paths[name] = path
        return paths


# --------------------------------------------------------------------------- #
# Motor
# --------------------------------------------------------------------------- #
class SyntheticDataGenerator:
    """Tek kullanımlık üretici: `SyntheticDataGenerator(cfg).generate()`."""

    def __init__(self, config: GeneratorSettings) -> None:
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self.faker = Faker("tr_TR")
        self.faker.seed_instance(config.seed)
        self.horizon = config.simulation_days * DAY

        self._accounts: list[dict[str, Any]] = []
        self._devices: list[dict[str, Any]] = []
        self._owner_links: list[tuple[int, int, int]] = []  # (hesap, cihaz, kayıt zamanı)
        self._tx: dict[str, list[Any]] = {f: [] for f in _TX_FIELDS}
        self._ibans: set[str] = set()

        self._customers = np.empty(0, dtype=np.int64)
        self._merchants = np.empty(0, dtype=np.int64)
        self._merchant_probs = np.empty(0)
        self._merchant_ticket = np.empty(0)
        self._merchants_by_cat: dict[MerchantCategory, np.ndarray] = {}
        self._customer_devices: dict[int, list[int]] = {}
        self._contacts: dict[int, np.ndarray] = {}
        self._victim_queue: list[int] = []
        self._median_amount: dict[int, float] = {}
        # Meşru ama "şüpheli görünen" davranışlar (hard negatives)
        self._trips: dict[int, tuple[int, int, str, str]] = {}  # müşteri → (başla, bitir, ülke, ip öneki)
        self._vpn_users: set[int] = set()
        self._device_switch: dict[int, tuple[int, int]] = {}  # müşteri → (yeni cihaz, geçiş anı)
        self._generated = False

    # ------------------------------------------------------------------ API
    def generate(self) -> SyntheticDataset:
        if self._generated:
            raise RuntimeError("Her generator örneği yalnızca bir kez generate() çağırabilir")
        self._generated = True

        self._build_merchants()
        self._build_customers()
        self._build_contacts()
        self._build_legit_behaviours()
        self._simulate_legit_activity()
        self._simulate_legit_bursts()
        self._prepare_victim_pool()

        self._inject_fraud_rings()
        self._inject_account_takeovers()
        self._inject_mule_networks()
        self._inject_card_testing()

        dataset = self._finalize()
        logger.info("Üretim tamamlandı: %s", dataset.summary())
        return dataset

    # ------------------------------------------------------------ yardımcılar
    def _pick(self, seq: Sequence[T], p: np.ndarray | None = None) -> T:
        return seq[int(self.rng.choice(len(seq), p=p))]

    def _offset(self, start_margin: int, end_margin: int) -> int:
        """Simülasyon penceresinde [start_margin, horizon - end_margin) aralığında bir an."""
        high = self.horizon - end_margin
        if high <= start_margin:
            raise ValueError("simulation_days senaryo süreleri için çok kısa")
        return int(self.rng.integers(start_margin, high))

    def _tr_ip(self) -> str:
        return f"{self._pick(TR_ISP_PREFIXES)}.{self.rng.integers(0, 256)}.{self.rng.integers(1, 255)}"

    def _foreign_ip(self) -> tuple[str, str]:
        prefix, country = self._pick(FOREIGN_IP_PREFIXES)
        return f"{prefix}.{self.rng.integers(0, 256)}.{self.rng.integers(1, 255)}", country

    def _attack_origin(self) -> tuple[str, str, str | None]:
        """Saldırgan bağlantısı: yurt içi residential proxy veya yurt dışı VPN/hosting → (ip, ülke, şehir)."""
        if self.rng.random() < self.cfg.domestic_attack_share:
            return self._tr_ip(), "TR", self._pick(TR_CITIES, CITY_PROBS)
        ip, country = self._foreign_ip()
        return ip, country, None

    def _mule_profile(self, t_ref: int, new_min: int, new_max: int) -> tuple[int, int]:
        """Mule/drop hesap (açılış anı, KYC): yeni açılmış ya da devşirilmiş eski meşru hesap."""
        if self.rng.random() < self.cfg.recruited_mule_share:
            return -int(self.rng.integers(365 * DAY, 6 * 365 * DAY)), int(self.rng.choice([2, 3]))
        return t_ref - int(self.rng.integers(new_min, new_max)), 1

    def _create_account(
        self,
        account_type: AccountType,
        *,
        opened_at: int,
        city: str,
        kyc_level: int,
        is_fraudster: bool = False,
        case_id: str | None = None,
        merchant_category: MerchantCategory | None = None,
    ) -> int:
        while True:
            account_no = "".join(map(str, self.rng.integers(0, 10, 16)))
            iban = make_tr_iban(self._pick(TR_BANK_CODES), account_no)
            if iban not in self._ibans:
                self._ibans.add(iban)
                break
        name = self.faker.name() if account_type is AccountType.PERSONAL else self._business_name()
        self._accounts.append({
            "iban": iban,
            "holder_name": name[:128],
            "account_type": account_type.value,
            "kyc_level": int(kyc_level),
            "city": city,
            "merchant_category": merchant_category.value if merchant_category else None,
            "opened_at": int(opened_at),
            "is_fraudster": is_fraudster,
            "fraud_case_id": case_id,
        })
        return len(self._accounts) - 1

    def _business_name(self) -> str:
        """Kurgusal işletme adı (Faker tr_TR şirket listesi gerçek firmalar içerdiği için kullanılmaz)."""
        sector = self._pick(BUSINESS_SECTORS)
        suffix = self._pick(("Ltd. Şti.", "A.Ş.", "Tic. Ltd. Şti.", "San. ve Tic. A.Ş."))
        return f"{self.faker.last_name()} {sector} {suffix}"

    def _create_device(
        self, device_type: DeviceType, registered_at: int | None, *, is_emulator: bool = False
    ) -> int:
        idx = len(self._devices)
        salt = int(self.rng.integers(0, 2**62))
        fingerprint = hashlib.sha256(f"{self.cfg.seed}:{idx}:{salt}".encode()).hexdigest()[:32]
        self._devices.append({
            "fingerprint": fingerprint,
            "device_type": device_type,
            "is_emulator": bool(is_emulator),
            "registered_at": registered_at,
            "home_ip": self._tr_ip(),
        })
        return idx

    def _register(self, account: int, device: int) -> None:
        reg = self._devices[device]["registered_at"]
        self._owner_links.append((account, device, int(reg)))

    def _digital_channel(self, device: int) -> Channel:
        dtype: DeviceType = self._devices[device]["device_type"]
        return Channel.MOBILE_APP if dtype.is_mobile else Channel.INTERNET_BANKING

    def _primary_device(self, account: int) -> int:
        return self._customer_devices[account][0]

    def _add_tx(
        self,
        ts: int,
        sender: int,
        receiver: int,
        amount: float,
        channel: Channel,
        tx_type: TransactionType,
        *,
        device: int = -1,
        ip: str | None = None,
        geo_city: str | None = None,
        geo_country: str = "TR",
        scenario: FraudScenario = FraudScenario.NONE,
        case_id: str | None = None,
    ) -> None:
        row = (
            int(ts), int(sender), int(receiver), int(device), max(1, round(amount * 100)),
            channel.value, tx_type.value, ip, geo_city, geo_country, scenario.value, case_id,
        )
        for field, value in zip(_TX_FIELDS, row, strict=True):
            self._tx[field].append(value)

    def _next_victim(self) -> int:
        if not self._victim_queue:
            raise ValueError(
                "Fraud senaryoları için yeterli müşteri yok; n_customers'ı artırın "
                "veya senaryo sayılarını azaltın."
            )
        return self._victim_queue.pop()

    # ------------------------------------------------------------ popülasyon
    def _build_merchants(self) -> None:
        rng, n = self.rng, self.cfg.n_merchants
        cats = list(MERCHANT_PROFILE)
        weights = np.array([MERCHANT_PROFILE[c][0] for c in cats])
        weights = weights / weights.sum()
        assigned = cats + [cats[i] for i in rng.choice(len(cats), n - len(cats), p=weights)]

        idxs, tickets = [], []
        for cat in assigned:
            idx = self._create_account(
                AccountType.MERCHANT,
                opened_at=-int(rng.integers(365 * DAY, 10 * 365 * DAY)),
                city=self._pick(TR_CITIES, CITY_PROBS),
                kyc_level=3,
                merchant_category=cat,
            )
            idxs.append(idx)
            tickets.append(MERCHANT_PROFILE[cat][1] * rng.lognormal(0, 0.25))

        self._merchants = np.array(idxs, dtype=np.int64)
        self._merchant_ticket = np.array(tickets)  # merchant idx == 0..n-1
        popularity = 1.0 / (rng.permutation(n) + 1) ** 1.1  # Zipf benzeri
        self._merchant_probs = popularity / popularity.sum()
        for cat in cats:
            self._merchants_by_cat[cat] = np.array(
                [m for m, c in zip(idxs, assigned, strict=True) if c is cat], dtype=np.int64
            )

    def _build_customers(self) -> None:
        rng = self.rng
        idxs = []
        for _ in range(self.cfg.n_customers):
            is_business = rng.random() < 0.12
            recently_opened = rng.random() < 0.10
            opened = -int(rng.integers(DAY, 60 * DAY) if recently_opened
                          else rng.integers(90 * DAY, 8 * 365 * DAY))
            idx = self._create_account(
                AccountType.BUSINESS if is_business else AccountType.PERSONAL,
                opened_at=opened,
                city=self._pick(TR_CITIES, CITY_PROBS),
                kyc_level=3 if is_business else int(rng.choice([1, 2, 3], p=[0.10, 0.35, 0.55])),
            )
            devices = []
            for d in range(int(rng.choice([1, 2, 3], p=[0.62, 0.32, 0.06]))):
                if d > 0 and rng.random() < 0.5:
                    dtype = DeviceType.DESKTOP_WEB
                else:
                    dtype = self._pick([DeviceType.MOBILE_ANDROID, DeviceType.MOBILE_IOS],
                                       np.array([0.68, 0.32]))
                dev = self._create_device(dtype, int(rng.integers(opened, 0)))
                self._register(idx, dev)
                devices.append(dev)
            self._customer_devices[idx] = devices
            idxs.append(idx)
        self._customers = np.array(idxs, dtype=np.int64)

    def _build_contacts(self) -> None:
        """Her müşteriye çoğunlukla aynı şehirden 3-8 kişilik sosyal çevre (topluluk yapısı)."""
        rng, customers = self.rng, self._customers
        cities = np.array([self._accounts[c]["city"] for c in customers])
        by_city = {city: customers[cities == city] for city in np.unique(cities)}
        for c, city in zip(customers, cities, strict=True):
            k = int(rng.integers(3, 9))
            same = by_city[city][by_city[city] != c]
            n_same = min(int(rng.binomial(k, 0.6)), len(same))
            picks = [rng.choice(same, n_same, replace=False)] if n_same else []
            picks.append(rng.choice(customers, k - n_same))
            contacts = np.unique(np.concatenate(picks))
            contacts = contacts[contacts != c]
            if contacts.size == 0:
                contacts = np.array([customers[(np.searchsorted(customers, c) + 1) % len(customers)]])
            self._contacts[int(c)] = contacts

    def _build_legit_behaviours(self) -> None:
        """Fraud'a benzeyen meşru davranışlar: seyahat, VPN, telefon değişimi, ortak hane cihazı."""
        rng, cfg = self.rng, self.cfg
        for c in map(int, self._customers):
            if rng.random() < cfg.legit_travel_rate:
                start = self._offset(0, 3 * DAY)
                prefix, country = self._pick(TRAVEL_IP_PREFIXES)
                self._trips[c] = (start, start + int(rng.integers(2 * DAY, 7 * DAY)), country, prefix)
            if rng.random() < cfg.legit_vpn_rate:
                self._vpn_users.add(c)
            if rng.random() < cfg.legit_device_change_rate:
                t_switch = self._offset(DAY, DAY)
                dtype = self._pick([DeviceType.MOBILE_ANDROID, DeviceType.MOBILE_IOS])
                dev = self._create_device(dtype, t_switch)
                self._register(c, dev)
                self._customer_devices[c].append(dev)
                self._device_switch[c] = (dev, t_switch)
            if rng.random() < cfg.legit_shared_device_rate:
                partner = int(rng.choice(self._contacts[c]))
                opened = max(self._accounts[c]["opened_at"], self._accounts[partner]["opened_at"])
                dev = self._create_device(DeviceType.MOBILE_ANDROID, int(rng.integers(opened, 1)))
                for member in (c, partner):  # aile tableti: iki hesap aynı cihazda
                    self._register(member, dev)
                    self._customer_devices[member].append(dev)

    def _legit_digital_context(self, c: int, ts: int, dev: int, from_home: bool) -> tuple[str, str]:
        """Meşru dijital işlemin (ip, ülke) bağlamı: seyahat > VPN > ev/operatör IP'si."""
        trip = self._trips.get(c)
        if trip and trip[0] <= ts < trip[1]:
            return f"{trip[3]}.{self.rng.integers(0, 256)}.{self.rng.integers(1, 255)}", trip[2]
        if c in self._vpn_users and self.rng.random() < 0.35:
            return self._foreign_ip()
        return (self._devices[dev]["home_ip"] if from_home else self._tr_ip()), "TR"

    def _simulate_legit_bursts(self) -> None:
        """Oyuncu / dijital içerik alışverişi: dakikalar içinde çok sayıda küçük ödeme (meşru)."""
        rng = self.rng
        digital = self._merchants_by_cat[MerchantCategory.DIGITAL_GOODS]
        for c in map(int, self._customers):
            if rng.random() >= self.cfg.legit_microburst_rate:
                continue
            for _ in range(int(rng.integers(1, 4))):
                t = self._offset(0, HOUR)
                dev = self._primary_device(c)
                if c in self._device_switch and t >= self._device_switch[c][1]:
                    dev = self._device_switch[c][0]
                merchant = int(rng.choice(digital))
                for _ in range(int(rng.integers(4, 16))):
                    t += int(rng.integers(20, 300))
                    if t >= self.horizon:
                        break
                    ip, country = self._legit_digital_context(c, t, dev, True)
                    self._add_tx(t, c, merchant if rng.random() < 0.7 else int(rng.choice(digital)),
                                 rng.uniform(5, 60), Channel.ECOMMERCE, TransactionType.CARD_PURCHASE,
                                 device=dev, ip=ip, geo_city=None if country != "TR" else self._accounts[c]["city"],
                                 geo_country=country)

    # ------------------------------------------------------------ normal akış
    def _simulate_legit_activity(self) -> None:
        rng, cfg = self.rng, self.cfg
        days = cfg.simulation_days
        n_m = len(self._merchants)
        utilities = self._merchants_by_cat[MerchantCategory.UTILITIES]

        for c in self._customers:
            c = int(c)
            acc = self._accounts[c]
            business = acc["account_type"] == AccountType.BUSINESS
            lam = rng.gamma(2.0, cfg.avg_daily_tx_per_customer / 2.0) * (1.6 if business else 1.0)
            n = int(rng.poisson(lam * days))
            if n == 0:
                continue

            ts = (rng.integers(0, days, n) * DAY + rng.choice(24, n, p=HOUR_PROBS) * HOUR
                  + rng.integers(0, HOUR, n))
            kinds = rng.choice(3, n, p=_KIND_P_BUSINESS if business else _KIND_P_PERSONAL)
            receivers = np.where(
                kinds == _KIND_CARD, rng.choice(self._merchants, n, p=self._merchant_probs),
                np.where(kinds == _KIND_P2P, rng.choice(self._contacts[c], n), rng.choice(utilities, n)),
            )
            ticket = self._merchant_ticket[np.minimum(receivers, n_m - 1)]
            amounts = np.where(
                kinds == _KIND_CARD, ticket * rng.lognormal(0, 0.6, n),
                np.where(kinds == _KIND_P2P, rng.lognormal(np.log(1_500), 1.0, n),
                         ticket * rng.lognormal(0, 0.45, n)),
            )
            amounts = np.clip(amounts * (3.5 if business else 1.0), 5, 500_000)

            devices = self._customer_devices[c]
            dev_p = np.array([1.0]) if len(devices) == 1 else np.array(
                [0.85] + [0.15 / (len(devices) - 1)] * (len(devices) - 1))
            dev_choice = rng.choice(len(devices), n, p=dev_p)
            is_pos = rng.random(n) < 0.55
            from_home_ip = rng.random(n) < 0.88
            travelling = rng.random(n) < 0.08

            switch = self._device_switch.get(c)
            trip = self._trips.get(c)

            for i in range(n):
                kind = int(kinds[i])
                t = int(ts[i])
                abroad = trip is not None and trip[0] <= t < trip[1]
                city = None if abroad else (self._pick(TR_CITIES, CITY_PROBS) if travelling[i] else acc["city"])
                if kind == _KIND_CARD and is_pos[i]:
                    self._add_tx(t, c, receivers[i], amounts[i], Channel.POS, TransactionType.CARD_PURCHASE,
                                 geo_city=city, geo_country=trip[2] if abroad else "TR")
                    continue
                dev = devices[int(dev_choice[i])]
                if switch is not None:  # telefon değişimi: geçişten sonra çoğunlukla yeni cihaz
                    if t >= switch[1] and rng.random() < 0.9:
                        dev = switch[0]
                    elif dev == switch[0] and t < switch[1]:
                        dev = devices[0]
                ip, country = self._legit_digital_context(c, t, dev, bool(from_home_ip[i]))
                if kind == _KIND_CARD:
                    channel, tx_type = Channel.ECOMMERCE, TransactionType.CARD_PURCHASE
                else:
                    channel = self._digital_channel(dev)
                    tx_type = TransactionType.TRANSFER if kind == _KIND_P2P else TransactionType.BILL_PAYMENT
                self._add_tx(t, c, receivers[i], amounts[i], channel, tx_type, device=dev, ip=ip,
                             geo_city=city if country == "TR" else None, geo_country=country)

    def _prepare_victim_pool(self) -> None:
        senders = np.array(self._tx["sender"], dtype=np.int64)
        amounts = np.array(self._tx["amount_minor"], dtype=np.int64) / 100
        if senders.size:
            medians = pd.Series(amounts).groupby(senders).median()
            self._median_amount = {int(k): float(v) for k, v in medians.items()}
        self._victim_queue = [int(v) for v in self.rng.permutation(self._customers)]

    # ------------------------------------------------------------ senaryolar
    def _inject_fraud_rings(self) -> None:
        rng, cfg = self.rng, self.cfg
        for r in range(cfg.n_fraud_rings):
            case = f"RING-{r + 1:03d}"
            k = int(rng.integers(cfg.ring_size_min, cfg.ring_size_max + 1))
            city = self._pick(TR_CITIES, CITY_PROBS)
            t0 = self._offset(2 * DAY, 2 * DAY)

            members = [
                self._create_account(
                    AccountType.PERSONAL,
                    opened_at=t0 - int(rng.integers(3 * DAY, 60 * DAY)),
                    city=city,
                    kyc_level=int(rng.choice([1, 2], p=[0.7, 0.3])),
                    is_fraudster=True,
                    case_id=case,
                )
                for _ in range(k)
            ]
            subnet = f"{self._pick(TR_ISP_PREFIXES)}.{rng.integers(0, 256)}"

            def ring_ip(_subnet: str = subnet) -> str:
                return f"{_subnet}.{rng.integers(1, 255)}"

            devices = []
            for _ in range(int(rng.integers(1, 3))):
                dev = self._create_device(
                    self._pick([DeviceType.MOBILE_ANDROID, DeviceType.DESKTOP_WEB]),
                    t0 - int(rng.integers(DAY, 3 * DAY)),
                    is_emulator=rng.random() < 0.6,
                )
                self._devices[dev]["home_ip"] = ring_ip()
                for m in members:  # aynı cihaz tüm halka üyelerine kayıtlı
                    self._register(m, dev)
                devices.append(dev)

            order = [members[i] for i in rng.permutation(k)]
            principal = float(rng.uniform(20_000, 150_000))

            # Placement: dış kurbanlardan halkaya giriş
            t = t0
            n_inj = int(rng.integers(1, 4))
            for share in rng.dirichlet(np.ones(n_inj)) * principal:
                victim = self._next_victim()
                vdev = self._primary_device(victim)
                self._add_tx(t, victim, order[0], share, self._digital_channel(vdev),
                             TransactionType.TRANSFER, device=vdev,
                             ip=self._devices[vdev]["home_ip"],
                             geo_city=self._accounts[victim]["city"],
                             scenario=FraudScenario.FRAUD_RING, case_id=case)
                t += int(rng.integers(5 * MINUTE, 40 * MINUTE))

            # Layering: A → B → ... → A, birkaç tur, her adımda küçük "komisyon" kaybı
            amount = principal * rng.uniform(0.97, 0.995)
            for _ in range(int(rng.integers(cfg.ring_rounds_min, cfg.ring_rounds_max + 1))):
                for j in range(k):
                    t += int(rng.integers(3 * MINUTE, 45 * MINUTE))
                    dev = self._pick(devices)
                    self._add_tx(t, order[j], order[(j + 1) % k], amount, self._digital_channel(dev),
                                 TransactionType.TRANSFER, device=dev, ip=ring_ip(), geo_city=city,
                                 scenario=FraudScenario.FRAUD_RING, case_id=case)
                    amount *= rng.uniform(0.975, 0.995)

            # Kamuflaj: üyelerin etiketsiz, masum görünen küçük alışverişleri
            for m in members:
                for _ in range(int(rng.integers(2, 6))):
                    merchant = int(rng.choice(self._merchants, p=self._merchant_probs))
                    self._add_tx(
                        self._offset(max(0, t0 - 3 * DAY), HOUR), m, merchant,
                        self._merchant_ticket[merchant] * rng.lognormal(0, 0.5),
                        Channel.ECOMMERCE, TransactionType.CARD_PURCHASE,
                        device=self._pick(devices), ip=ring_ip(), geo_city=city,
                    )

    def _inject_account_takeovers(self) -> None:
        rng, n = self.rng, self.cfg.n_account_takeovers
        if n == 0:
            return
        crew = "ATO-CREW"
        drops = []
        for _ in range(max(2, n // 5)):
            opened, kyc = self._mule_profile(0, 10 * DAY, 120 * DAY)
            drops.append(self._create_account(
                AccountType.PERSONAL, opened_at=opened, city=self._pick(TR_CITIES, CITY_PROBS),
                kyc_level=kyc, is_fraudster=True, case_id=crew,
            ))
        # Saldırgan cihaz havuzu: aynı cihaz birden fazla kurbanda görünür (graf sinyali)
        attacker_devices = [
            self._create_device(self._pick([DeviceType.DESKTOP_WEB, DeviceType.MOBILE_ANDROID]),
                                None, is_emulator=rng.random() < 0.4)
            for _ in range(max(1, n // 4))
        ]
        for e in range(n):
            case = f"ATO-{e + 1:03d}"
            victim = self._next_victim()
            dev = attacker_devices[e % len(attacker_devices)]
            ip, country, geo_city = self._attack_origin()
            base = max(self._median_amount.get(victim, 1_500.0), 500.0)
            t = self._offset(DAY, 6 * HOUR)
            for _ in range(int(rng.integers(2, 6))):
                t += int(rng.integers(MINUTE, 12 * MINUTE))
                amount = max(base * rng.uniform(8, 30), rng.uniform(10_000, 25_000))
                self._add_tx(t, victim, self._pick(drops), amount, Channel.INTERNET_BANKING,
                             TransactionType.TRANSFER, device=dev, ip=ip, geo_city=geo_city,
                             geo_country=country, scenario=FraudScenario.ACCOUNT_TAKEOVER, case_id=case)

    def _inject_mule_networks(self) -> None:
        rng, cfg = self.rng, self.cfg
        for e in range(cfg.n_mule_networks):
            case = f"MULE-{e + 1:03d}"
            city = self._pick(TR_CITIES, CITY_PROBS)
            fan_in_window = int(rng.integers(DAY // 2, 3 * DAY))
            # Bitiş payı olaya göre: fan-in penceresi + fan-out (≤ 6 sa + 6 × 20 dk) + tampon
            t_start = self._offset(DAY, fan_in_window + 9 * HOUR)

            def new_mule(_t: int = t_start, _city: str = city, _case: str = case) -> int:
                opened, kyc = self._mule_profile(_t, 5 * DAY, 45 * DAY)
                return self._create_account(
                    AccountType.PERSONAL, opened_at=opened, city=_city, kyc_level=kyc,
                    is_fraudster=True, case_id=_case,
                )

            hub = new_mule()
            cash_outs = [new_mule() for _ in range(int(rng.integers(3, 7)))]
            shared = self._create_device(DeviceType.MOBILE_ANDROID, t_start - DAY)
            for acc in (hub, *cash_outs):
                self._register(acc, shared)
            hub_ip = self._devices[shared]["home_ip"]

            # Fan-in: kurbanlar kendi cihazlarından "yetkili" ödeme yapar (APP scam)
            n_victims = int(rng.integers(cfg.mule_victims_min, cfg.mule_victims_max + 1))
            in_times = np.sort(t_start + rng.integers(0, fan_in_window, n_victims))
            total_in = 0.0
            for t in in_times:
                victim = self._next_victim()
                vdev = self._primary_device(victim)
                amount = float(rng.uniform(1_000, 15_000))
                total_in += amount
                self._add_tx(int(t), victim, hub, amount, self._digital_channel(vdev),
                             TransactionType.TRANSFER, device=vdev,
                             ip=self._devices[vdev]["home_ip"],
                             geo_city=self._accounts[victim]["city"],
                             scenario=FraudScenario.MONEY_MULE, case_id=case)

            # Fan-out: son girişten 1-6 saat sonra cash-out hesaplarına dağıtım
            t = int(in_times[-1]) + int(rng.integers(HOUR, 6 * HOUR))
            out_total = total_in * rng.uniform(0.90, 0.97)
            for share, target in zip(rng.dirichlet(np.ones(len(cash_outs))), cash_outs, strict=True):
                t += int(rng.integers(2 * MINUTE, 20 * MINUTE))
                self._add_tx(t, hub, target, out_total * share, Channel.MOBILE_APP,
                             TransactionType.TRANSFER, device=shared, ip=hub_ip, geo_city=city,
                             scenario=FraudScenario.MONEY_MULE, case_id=case)

    def _inject_card_testing(self) -> None:
        rng = self.rng
        test_merchants = np.concatenate([self._merchants_by_cat[c] for c in CARD_TESTING_CATEGORIES])
        cashout_merchants = np.concatenate([self._merchants_by_cat[c] for c in CASHOUT_CATEGORIES])
        for e in range(self.cfg.n_card_testing):
            case = f"CT-{e + 1:03d}"
            victim = self._next_victim()
            dev = self._create_device(DeviceType.DESKTOP_WEB, None, is_emulator=rng.random() < 0.5)
            ip, country, geo_city = self._attack_origin()
            t = self._offset(DAY, 2 * HOUR)
            common = dict(device=dev, ip=ip, geo_city=geo_city, geo_country=country,
                          scenario=FraudScenario.CARD_TESTING, case_id=case)
            for _ in range(int(rng.integers(10, 31))):
                t += int(rng.integers(5, 90))
                self._add_tx(t, victim, int(rng.choice(test_merchants)), rng.uniform(1, 20),
                             Channel.ECOMMERCE, TransactionType.CARD_PURCHASE, **common)
            t += int(rng.integers(10 * MINUTE, 30 * MINUTE))
            for _ in range(int(rng.integers(1, 3))):
                self._add_tx(t, victim, int(rng.choice(cashout_merchants)), rng.uniform(5_000, 40_000),
                             Channel.ECOMMERCE, TransactionType.CARD_PURCHASE, **common)
                t += int(rng.integers(MINUTE, 5 * MINUTE))

    # ------------------------------------------------------------ çıktı
    def _finalize(self) -> SyntheticDataset:
        rng = self.rng
        start = pd.Timestamp(self.cfg.start_date)

        def to_ts(seconds: Any) -> pd.Series:
            return start + pd.to_timedelta(pd.Series(seconds, dtype="int64"), unit="s")

        # Sızıntısız ID ataması (üretim sırası ≠ ID sırası)
        acc_ids = np.array([f"ACC-{i + 1:06d}" for i in rng.permutation(len(self._accounts))])
        dev_ids = np.array([f"DEV-{i + 1:06d}" for i in rng.permutation(len(self._devices))])

        raw = pd.DataFrame(self._tx).sort_values("ts", kind="stable").reset_index(drop=True)
        if raw["ts"].min() < 0 or raw["ts"].max() >= self.horizon:
            raise RuntimeError("Simülasyon penceresi dışında işlem üretildi")
        device_idx = raw["device"].to_numpy()

        transactions = pd.DataFrame({
            "transaction_id": [f"TX-{i + 1:08d}" for i in range(len(raw))],
            "timestamp": to_ts(raw["ts"]),
            "sender_account_id": acc_ids[raw["sender"].to_numpy()],
            "receiver_account_id": acc_ids[raw["receiver"].to_numpy()],
            "device_id": pd.Series(np.where(device_idx >= 0, dev_ids[np.maximum(device_idx, 0)], None),
                                   dtype=object),
            "amount_minor": raw["amount_minor"].astype("int64"),
            "currency": self.cfg.currency,
            "channel": raw["channel"],
            "transaction_type": raw["tx_type"],
            "ip_address": raw["ip"],
            "geo_city": raw["geo_city"],
            "geo_country": raw["geo_country"],
            "is_fraud": raw["scenario"] != FraudScenario.NONE.value,
            "fraud_scenario": raw["scenario"],
            "fraud_case_id": raw["case_id"],
        })

        acc_raw = pd.DataFrame(self._accounts)
        accounts = acc_raw.assign(
            account_id=acc_ids, opened_at=to_ts(acc_raw["opened_at"])
        )[record_fields("accounts")].sort_values("account_id").reset_index(drop=True)

        # Cihaz–hesap kenarları: kayıtlı sahiplik ∪ işlemlerde gözlenen kullanım
        used = raw[raw["device"] >= 0]
        observed = (used.groupby(["sender", "device"])["ts"].agg(tx_first="min", tx_last="max")
                    .reset_index().rename(columns={"sender": "account"}))
        owners = pd.DataFrame(self._owner_links, columns=["account", "device", "registered"])
        links = owners.merge(observed, on=["account", "device"], how="outer")
        first = np.fmin(links["registered"], links["tx_first"]).astype("int64")
        last = np.fmax(links["registered"], links["tx_last"]).astype("int64")
        account_devices = pd.DataFrame({
            "account_id": acc_ids[links["account"].astype("int64").to_numpy()],
            "device_id": dev_ids[links["device"].astype("int64").to_numpy()],
            "first_seen": to_ts(first),
            "last_seen": to_ts(last),
        }).sort_values(["account_id", "device_id"]).reset_index(drop=True)

        dev_raw = pd.DataFrame(self._devices)
        tx_first = used.groupby("device")["ts"].min().reindex(range(len(dev_raw)))
        registered = pd.to_numeric(dev_raw["registered_at"], errors="coerce")
        dev_first = np.fmin(registered, tx_first.to_numpy()).fillna(0).astype("int64")
        devices = pd.DataFrame({
            "device_id": dev_ids,
            "fingerprint": dev_raw["fingerprint"],
            "device_type": dev_raw["device_type"].map(lambda d: d.value),
            "is_emulator": dev_raw["is_emulator"].astype(bool),
            "first_seen": to_ts(dev_first),
        }).sort_values("device_id").reset_index(drop=True)

        return SyntheticDataset(
            accounts=accounts,
            devices=devices[record_fields("devices")],
            account_devices=account_devices[record_fields("account_devices")],
            transactions=transactions[record_fields("transactions")],
            config=self.cfg,
        )


def generate_dataset(config: GeneratorSettings | None = None) -> SyntheticDataset:
    return SyntheticDataGenerator(config or get_settings().generator).generate()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AegisGraph sentetik veri üretici")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--customers", type=int, dest="n_customers")
    parser.add_argument("--merchants", type=int, dest="n_merchants")
    parser.add_argument("--days", type=int, dest="simulation_days")
    parser.add_argument("--out", type=Path, dest="output_dir", help="CSV çıktı klasörü")
    parser.add_argument("--no-db", action="store_true", help="SQLite'a yazma, sadece CSV üret")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    from src.database.db import create_db_engine, init_db, persist_dataset

    args = _parse_args(argv)
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    overrides = {k: v for k, v in vars(args).items() if k != "no_db" and v is not None}
    cfg = GeneratorSettings.model_validate({**settings.generator.model_dump(), **overrides})

    dataset = SyntheticDataGenerator(cfg).generate()
    for name, path in dataset.to_csv(cfg.output_dir).items():
        logger.info("CSV yazıldı: %s → %s", name, path)

    if not args.no_db:
        engine = create_db_engine()
        init_db(engine, drop_existing=True)
        persist_dataset(dataset, engine)
        engine.dispose()
        logger.info("Veritabanı: %s", settings.database.sqlalchemy_url)

    print("\n=== AegisGraph sentetik veri özeti ===")
    for key, value in dataset.summary().items():
        print(f"{key:>22}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
