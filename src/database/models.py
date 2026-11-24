"""AegisGraph veri modelleri.

İki katman vardır:
  * SQLAlchemy ORM tabloları (kalıcı depolama): Account, Device, AccountDevice, Transaction, Alert
  * Pydantic kayıt sözleşmeleri (*Record): üretilen / API'ye giren verinin şema doğrulaması

Para birimi kuralı: tutarlar her zaman **kuruş cinsinden tamsayı** (`amount_minor`) saklanır.
Float para tutarı veritabanına yazılmaz; gösterim için `Transaction.amount` (Decimal) kullanılır.

Zaman kuralı: tüm zaman damgaları UTC ve timezone-naive saklanır.
"""

from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, IPvAnyAddress, model_validator
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    func,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from src.compat import StrEnum


# --------------------------------------------------------------------------- #
# Enum'lar (isim == değer; DB'de string olarak saklanır)
# --------------------------------------------------------------------------- #
class AccountType(StrEnum):
    PERSONAL = "PERSONAL"
    BUSINESS = "BUSINESS"
    MERCHANT = "MERCHANT"


class MerchantCategory(StrEnum):
    GROCERY = "GROCERY"
    RESTAURANT = "RESTAURANT"
    FUEL = "FUEL"
    FASHION = "FASHION"
    ELECTRONICS = "ELECTRONICS"
    ONLINE_MARKETPLACE = "ONLINE_MARKETPLACE"
    PHARMACY = "PHARMACY"
    UTILITIES = "UTILITIES"
    TRAVEL = "TRAVEL"
    DIGITAL_GOODS = "DIGITAL_GOODS"


class DeviceType(StrEnum):
    MOBILE_ANDROID = "MOBILE_ANDROID"
    MOBILE_IOS = "MOBILE_IOS"
    DESKTOP_WEB = "DESKTOP_WEB"

    @property
    def is_mobile(self) -> bool:
        return self is not DeviceType.DESKTOP_WEB


class Channel(StrEnum):
    MOBILE_APP = "MOBILE_APP"
    INTERNET_BANKING = "INTERNET_BANKING"
    POS = "POS"
    ECOMMERCE = "ECOMMERCE"


class TransactionType(StrEnum):
    TRANSFER = "TRANSFER"
    CARD_PURCHASE = "CARD_PURCHASE"
    BILL_PAYMENT = "BILL_PAYMENT"


class FraudScenario(StrEnum):
    NONE = "NONE"
    FRAUD_RING = "FRAUD_RING"  # paylaşılan cihazlı, dairesel para akışı (layering)
    ACCOUNT_TAKEOVER = "ACCOUNT_TAKEOVER"  # yeni cihaz + yabancı IP + yüksek tutarlı boşaltma
    MONEY_MULE = "MONEY_MULE"  # çok kurbandan fan-in, hızlı fan-out
    CARD_TESTING = "CARD_TESTING"  # mikro tutarlı hız saldırısı + büyük alışveriş


class AlertStatus(StrEnum):
    OPEN = "OPEN"
    INVESTIGATING = "INVESTIGATING"
    ESCALATED = "ESCALATED"
    CONFIRMED_FRAUD = "CONFIRMED_FRAUD"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    CLOSED = "CLOSED"


class SuspectedTypology(StrEnum):
    """Otonom soruşturma ajanının tespit ettiği şüpheli tipoloji sınıfı."""

    FRAUD_RING = "FRAUD_RING"
    ACCOUNT_TAKEOVER = "ACCOUNT_TAKEOVER"
    MONEY_MULE = "MONEY_MULE"
    CARD_TESTING = "CARD_TESTING"
    ANOMALOUS_ACTIVITY = "ANOMALOUS_ACTIVITY"


class RecommendedAction(StrEnum):
    """Human-in-the-loop onayı için sunulan aksiyon önerisi."""

    BLOCK_ACCOUNT = "BLOCK_ACCOUNT"
    FREEZE_TX = "FREEZE_TX"
    ENHANCED_DUE_DILIGENCE = "ENHANCED_DUE_DILIGENCE"
    FALSE_POSITIVE_DISMISS = "FALSE_POSITIVE_DISMISS"


# Eğitimde ÖZELLİK OLARAK KULLANILMAMASI gereken etiket (ground truth) kolonları.
GROUND_TRUTH_COLUMNS: dict[str, tuple[str, ...]] = {
    "accounts": ("is_fraudster", "fraud_case_id"),
    "transactions": ("is_fraud", "fraud_scenario", "fraud_case_id"),
}


def _enum(enum_cls: type[StrEnum]) -> SAEnum:
    return SAEnum(
        enum_cls,
        native_enum=False,
        length=32,
        validate_strings=True,
        name=enum_cls.__name__.lower(),
    )


# --------------------------------------------------------------------------- #
# ORM tabloları
# --------------------------------------------------------------------------- #
class Base(DeclarativeBase):
    pass


class Account(Base):
    __tablename__ = "accounts"

    account_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    iban: Mapped[str] = mapped_column(String(34), unique=True, nullable=False)
    holder_name: Mapped[str] = mapped_column(String(128), nullable=False)
    account_type: Mapped[AccountType] = mapped_column(_enum(AccountType), nullable=False, index=True)
    kyc_level: Mapped[int] = mapped_column(Integer, nullable=False)
    city: Mapped[str] = mapped_column(String(64), nullable=False)
    merchant_category: Mapped[MerchantCategory | None] = mapped_column(_enum(MerchantCategory))
    opened_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    # Ground truth (sadece değerlendirme için)
    is_fraudster: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    fraud_case_id: Mapped[str | None] = mapped_column(String(32), index=True)

    device_links: Mapped[list[AccountDevice]] = relationship(back_populates="account")

    __table_args__ = (CheckConstraint("kyc_level BETWEEN 1 AND 3", name="ck_accounts_kyc_level"),)


class Device(Base):
    __tablename__ = "devices"

    device_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    fingerprint: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    device_type: Mapped[DeviceType] = mapped_column(_enum(DeviceType), nullable=False)
    is_emulator: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    first_seen: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    account_links: Mapped[list[AccountDevice]] = relationship(back_populates="device")


class AccountDevice(Base):
    """Hesap ↔ cihaz kenarı (graf analizinin temel bipartit ilişkisi)."""

    __tablename__ = "account_devices"

    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.account_id"), primary_key=True)
    device_id: Mapped[str] = mapped_column(ForeignKey("devices.device_id"), primary_key=True, index=True)
    first_seen: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    last_seen: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    account: Mapped[Account] = relationship(back_populates="device_links")
    device: Mapped[Device] = relationship(back_populates="account_links")

    __table_args__ = (CheckConstraint("first_seen <= last_seen", name="ck_account_devices_seen_order"),)


class Transaction(Base):
    __tablename__ = "transactions"

    transaction_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    sender_account_id: Mapped[str] = mapped_column(ForeignKey("accounts.account_id"), nullable=False)
    receiver_account_id: Mapped[str] = mapped_column(ForeignKey("accounts.account_id"), nullable=False)
    device_id: Mapped[str | None] = mapped_column(ForeignKey("devices.device_id"))
    amount_minor: Mapped[int] = mapped_column(BigInteger, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="TRY")
    channel: Mapped[Channel] = mapped_column(_enum(Channel), nullable=False)
    transaction_type: Mapped[TransactionType] = mapped_column(_enum(TransactionType), nullable=False)
    ip_address: Mapped[str | None] = mapped_column(String(45))
    geo_city: Mapped[str | None] = mapped_column(String(64))
    geo_country: Mapped[str] = mapped_column(String(2), nullable=False, default="TR")
    # Ground truth (sadece değerlendirme için)
    is_fraud: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    fraud_scenario: Mapped[FraudScenario] = mapped_column(
        _enum(FraudScenario), nullable=False, default=FraudScenario.NONE
    )
    fraud_case_id: Mapped[str | None] = mapped_column(String(32), index=True)

    sender: Mapped[Account] = relationship(foreign_keys=[sender_account_id])
    receiver: Mapped[Account] = relationship(foreign_keys=[receiver_account_id])
    device: Mapped[Device | None] = relationship()
    alerts: Mapped[list[Alert]] = relationship(back_populates="transaction")

    __table_args__ = (
        CheckConstraint("amount_minor > 0", name="ck_transactions_amount_positive"),
        CheckConstraint("sender_account_id <> receiver_account_id", name="ck_transactions_no_self_transfer"),
        Index("ix_transactions_sender_ts", "sender_account_id", "timestamp"),
        Index("ix_transactions_receiver_ts", "receiver_account_id", "timestamp"),
        Index("ix_transactions_device_ts", "device_id", "timestamp"),
    )

    @property
    def amount(self) -> Decimal:
        """Ana para birimi cinsinden tutar (örn. TRY)."""
        return Decimal(self.amount_minor) / Decimal(100)


class Alert(Base):
    """Skorlama motorunun ve soruşturma ajanının ürettiği alarm."""

    __tablename__ = "alerts"

    alert_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transaction_id: Mapped[str] = mapped_column(
        ForeignKey("transactions.transaction_id"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    ml_score: Mapped[float | None] = mapped_column(Float)
    graph_score: Mapped[float | None] = mapped_column(Float)
    status: Mapped[AlertStatus] = mapped_column(
        _enum(AlertStatus), nullable=False, default=AlertStatus.OPEN, index=True
    )
    reasons: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)  # SHAP faktörleri
    assigned_to: Mapped[str | None] = mapped_column(String(64))
    resolution_note: Mapped[str | None] = mapped_column(String(2000))

    transaction: Mapped[Transaction] = relationship(back_populates="alerts")
    investigation: Mapped[Investigation | None] = relationship(back_populates="alert", uselist=False)

    __table_args__ = (CheckConstraint("risk_score BETWEEN 0 AND 1", name="ck_alerts_risk_score_range"),)


class Investigation(Base):
    """Otonom soruşturma ajanının bir alarm için ürettiği yapılandırılmış SAR raporu."""

    __tablename__ = "investigations"

    investigation_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    alert_id: Mapped[int] = mapped_column(
        ForeignKey("alerts.alert_id"), unique=True, nullable=False, index=True
    )
    tx_id: Mapped[str] = mapped_column(
        ForeignKey("transactions.transaction_id"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())
    suspected_typology: Mapped[SuspectedTypology] = mapped_column(
        _enum(SuspectedTypology), nullable=False
    )
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    executive_summary_tr: Mapped[str] = mapped_column(String(4000), nullable=False)
    evidence_chain: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    related_accounts: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    recommended_action: Mapped[RecommendedAction] = mapped_column(
        _enum(RecommendedAction), nullable=False
    )
    requires_human_approval: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    tool_trace: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)

    alert: Mapped[Alert] = relationship(back_populates="investigation")

    __table_args__ = (
        CheckConstraint("confidence_score BETWEEN 0 AND 1", name="ck_investigations_confidence_range"),
    )


# --------------------------------------------------------------------------- #
# Pydantic kayıt sözleşmeleri (şema doğrulama / API I/O)
# --------------------------------------------------------------------------- #
class _Record(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid", frozen=True)


class AccountRecord(_Record):
    account_id: str = Field(pattern=r"^ACC-\d{6}$")
    iban: str = Field(pattern=r"^TR\d{24}$")
    holder_name: str = Field(min_length=1, max_length=128)
    account_type: AccountType
    kyc_level: int = Field(ge=1, le=3)
    city: str = Field(min_length=1)
    merchant_category: MerchantCategory | None
    opened_at: datetime
    is_fraudster: bool
    fraud_case_id: str | None

    @model_validator(mode="after")
    def _consistency(self) -> AccountRecord:
        if (self.account_type is AccountType.MERCHANT) != (self.merchant_category is not None):
            raise ValueError("merchant_category yalnızca MERCHANT hesaplarda dolu olmalı")
        if self.is_fraudster and self.fraud_case_id is None:
            raise ValueError("Fraudster hesabın fraud_case_id değeri olmalı")
        return self


class DeviceRecord(_Record):
    device_id: str = Field(pattern=r"^DEV-\d{6}$")
    fingerprint: str = Field(pattern=r"^[0-9a-f]{32}$")
    device_type: DeviceType
    is_emulator: bool
    first_seen: datetime


class AccountDeviceRecord(_Record):
    account_id: str = Field(pattern=r"^ACC-\d{6}$")
    device_id: str = Field(pattern=r"^DEV-\d{6}$")
    first_seen: datetime
    last_seen: datetime

    @model_validator(mode="after")
    def _order(self) -> AccountDeviceRecord:
        if self.first_seen > self.last_seen:
            raise ValueError("first_seen, last_seen'den sonra olamaz")
        return self


class TransactionRecord(_Record):
    transaction_id: str = Field(pattern=r"^TX-\d{8}$")
    timestamp: datetime
    sender_account_id: str = Field(pattern=r"^ACC-\d{6}$")
    receiver_account_id: str = Field(pattern=r"^ACC-\d{6}$")
    device_id: str | None = Field(default=None, pattern=r"^DEV-\d{6}$")
    amount_minor: int = Field(gt=0)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    channel: Channel
    transaction_type: TransactionType
    ip_address: IPvAnyAddress | None
    geo_city: str | None
    geo_country: str = Field(pattern=r"^[A-Z]{2}$")
    is_fraud: bool
    fraud_scenario: FraudScenario
    fraud_case_id: str | None

    @model_validator(mode="after")
    def _consistency(self) -> TransactionRecord:
        if self.sender_account_id == self.receiver_account_id:
            raise ValueError("Gönderen ve alıcı aynı hesap olamaz")
        labelled = self.fraud_scenario is not FraudScenario.NONE
        if not (self.is_fraud == labelled == (self.fraud_case_id is not None)):
            raise ValueError("is_fraud / fraud_scenario / fraud_case_id tutarsız")
        if (self.channel is Channel.POS) != (self.device_id is None):
            raise ValueError("POS işlemlerinde cihaz olmaz; dijital kanallarda cihaz zorunludur")
        return self


class InvestigationRecord(_Record):
    """Soruşturma raporu sözleşmesi; insan onayı (Human-in-the-Loop) daima zorunludur.

    Kolon sırası `investigations` ORM tablosuyla birebir aynıdır (`record_fields("investigations")`).
    """

    investigation_id: int | None = Field(default=None, gt=0)
    alert_id: int = Field(gt=0)
    tx_id: str = Field(min_length=1, max_length=20)
    created_at: datetime
    suspected_typology: SuspectedTypology
    confidence_score: float = Field(ge=0.0, le=1.0)
    executive_summary_tr: str = Field(min_length=1, max_length=4000)
    evidence_chain: list[str] = Field(min_length=1)
    related_accounts: list[str]
    recommended_action: RecommendedAction
    requires_human_approval: bool = True
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _require_human_approval(self) -> InvestigationRecord:
        if not self.requires_human_approval:
            raise ValueError("Human-in-the-Loop ilkesi gereği requires_human_approval daima True olmalıdır")
        return self


RECORD_SCHEMAS: dict[str, type[_Record]] = {
    "accounts": AccountRecord,
    "devices": DeviceRecord,
    "account_devices": AccountDeviceRecord,
    "transactions": TransactionRecord,
    "investigations": InvestigationRecord,
}


def mask_iban(iban: str | None) -> str:
    """TR IBAN bilgisini log ve API/ajan çıktıları için maskeler (`TR** **** **** **** **** **12 34`)."""
    if not iban:
        return "TR** **** **** **** **** **** **"
    clean = "".join(iban.split()).upper()
    if len(clean) >= 6:
        country = clean[:2]
        return f"{country}** **** **** **** **** **{clean[-4:-2]} {clean[-2:]}"
    return f"{clean[:2]}** ****"


_RAW_IBAN_RE = re.compile(r"\bTR\d{2}(?: ?\d{4}){5} ?\d{2}\b", re.IGNORECASE)


def redact_ibans(value: Any) -> Any:
    """Metin, liste ve sözlüklerdeki maskelenmemiş TR IBAN'ları özyinelemeli olarak maskeler.

    Ajan/LLM çıktıları için derinlemesine savunma katmanıdır: ham IBAN hiçbir koşulda dışarı sızmaz.
    """
    if isinstance(value, str):
        return _RAW_IBAN_RE.sub(lambda m: mask_iban(m.group(0)), value)
    if isinstance(value, dict):
        return {k: redact_ibans(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [redact_ibans(v) for v in value]
    return value


def mask_holder_name(name: str | None) -> str:
    """Hesap sahibi adını KVKK gereği maskeler (`Ahmet Yılmaz` → `A**** Y*****`)."""
    if not name:
        return "***"
    return " ".join(f"{part[0]}{'*' * max(len(part) - 1, 2)}" for part in name.split())


def record_fields(table: str) -> list[str]:
    """Bir tablonun beklenen kolon sırası (DataFrame ↔ DB ↔ Pydantic tek kaynak)."""
    return list(RECORD_SCHEMAS[table].model_fields)


__all__: list[Any] = [
    "GROUND_TRUTH_COLUMNS",
    "RECORD_SCHEMAS",
    "Account",
    "AccountDevice",
    "AccountDeviceRecord",
    "AccountRecord",
    "AccountType",
    "Alert",
    "AlertStatus",
    "Base",
    "Channel",
    "Device",
    "DeviceRecord",
    "DeviceType",
    "FraudScenario",
    "Investigation",
    "InvestigationRecord",
    "MerchantCategory",
    "RecommendedAction",
    "SuspectedTypology",
    "Transaction",
    "TransactionRecord",
    "TransactionType",
    "mask_holder_name",
    "mask_iban",
    "record_fields",
    "redact_ibans",
]
