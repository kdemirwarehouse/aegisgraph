"""AegisGraph merkezi konfigürasyonu.

Tüm eşik değerleri, yollar ve üretici parametreleri tek bir yerden yönetilir.
Değerler ortam değişkenleri veya `.env` dosyası ile ezilebilir:

    AEGIS_ENVIRONMENT=prod
    AEGIS_DATABASE__URL=sqlite:///C:/data/aegis.db
    AEGIS_GENERATOR__N_CUSTOMERS=5000
    AEGIS_RISK__ALERT_SCORE=0.65

Kod içinde her zaman `get_settings()` kullanın; modül seviyesinde sabit eşik yazmayın.
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"


class DatabaseSettings(BaseModel):
    """Veritabanı bağlantısı. `url` verilmezse `path` üzerinden SQLite kullanılır."""

    url: str | None = None
    path: Path = DATA_DIR / "aegisgraph.db"
    echo: bool = False

    @property
    def sqlalchemy_url(self) -> str:
        if self.url:
            return self.url
        return f"sqlite:///{self.path.resolve().as_posix()}"


class GeneratorSettings(BaseModel):
    """Sentetik veri motoru parametreleri. Aynı `seed` her zaman aynı veriyi üretir."""

    seed: int = 42
    n_customers: int = Field(2_000, ge=50, description="Bireysel + kurumsal müşteri sayısı")
    n_merchants: int = Field(150, ge=10, description="Üye işyeri sayısı (her kategoriden en az bir)")
    simulation_days: int = Field(30, ge=7, le=365)
    start_date: datetime = datetime(2026, 10, 1)
    avg_daily_tx_per_customer: float = Field(1.2, gt=0, le=20)
    currency: str = Field("TRY", pattern=r"^[A-Z]{3}$")

    # --- Fraud senaryoları ---
    n_fraud_rings: int = Field(8, ge=0)
    ring_size_min: int = Field(4, ge=3)
    ring_size_max: int = Field(8, ge=3)
    ring_rounds_min: int = Field(2, ge=1)
    ring_rounds_max: int = Field(4, ge=1)
    n_account_takeovers: int = Field(25, ge=0)
    n_mule_networks: int = Field(12, ge=0)
    mule_victims_min: int = Field(15, ge=3)
    mule_victims_max: int = Field(40, ge=3)
    n_card_testing: int = Field(10, ge=0)
    domestic_attack_share: float = Field(0.4, ge=0, le=1, description="ATO/card testing için yurt içi proxy payı")
    recruited_mule_share: float = Field(0.4, ge=0, le=1, description="Devşirilmiş eski hesap olan mule payı")

    # --- Fraud'a benzeyen meşru davranışlar (hard negatives) ---
    legit_travel_rate: float = Field(0.04, ge=0, le=1, description="Pencerede yurt dışına çıkan müşteri payı")
    legit_vpn_rate: float = Field(0.03, ge=0, le=1)
    legit_device_change_rate: float = Field(0.04, ge=0, le=1, description="Telefon değiştiren müşteri payı")
    legit_shared_device_rate: float = Field(0.04, ge=0, le=1, description="Aile cihazı paylaşan müşteri payı")
    legit_microburst_rate: float = Field(0.04, ge=0, le=1, description="Dijital mikro-ödeme seansı yapan pay")

    output_dir: Path = DATA_DIR / "raw"

    @model_validator(mode="after")
    def _check_ranges(self) -> GeneratorSettings:
        if self.ring_size_min > self.ring_size_max:
            raise ValueError("ring_size_min, ring_size_max değerinden büyük olamaz")
        if self.ring_rounds_min > self.ring_rounds_max:
            raise ValueError("ring_rounds_min, ring_rounds_max değerinden büyük olamaz")
        if self.mule_victims_min > self.mule_victims_max:
            raise ValueError("mule_victims_min, mule_victims_max değerinden büyük olamaz")
        return self


class RiskThresholds(BaseModel):
    """Karar eşikleri. Model ve API katmanı bu değerleri tüketir."""

    review_score: float = Field(0.50, ge=0, le=1, description="Bu skorun üstü manuel inceleme kuyruğuna")
    alert_score: float = Field(0.70, ge=0, le=1, description="Bu skorun üstü Alert üretir")
    block_score: float = Field(0.90, ge=0, le=1, description="Bu skorun üstü işlem bloklanır")
    high_value_amount_try: float = Field(50_000, gt=0)
    velocity_window_seconds: int = Field(600, gt=0)
    velocity_max_tx: int = Field(10, gt=0)

    @model_validator(mode="after")
    def _check_order(self) -> RiskThresholds:
        if not self.review_score <= self.alert_score <= self.block_score:
            raise ValueError("Eşikler review_score <= alert_score <= block_score sırasında olmalı")
        return self


class FeatureSettings(BaseModel):
    """Özellik mühendisliği parametreleri. Batch ve online yol aynı değerleri kullanır."""

    home_country: str = Field("TR", pattern=r"^[A-Z]{2}$")
    night_start_hour: int = Field(1, ge=0, le=23)
    night_end_hour: int = Field(5, ge=0, le=23, description="Hariç (01:00 ≤ saat < 05:00)")
    velocity_windows_seconds: tuple[int, int, int] = (600, 3_600, 86_400)
    flow_window_seconds: int = Field(86_400, gt=0, description="Döngü/karşılıklı akış ve fan-in penceresi")
    max_cycle_hops: int = Field(7, ge=1, le=12, description="Alıcıdan göndericiye dönüş yolu arama derinliği")


class ModelSettings(BaseModel):
    """Eğitim, temporal split ve artefakt ayarları."""

    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    train_days: int = Field(21, ge=1)
    val_days: int = Field(4, ge=1)
    target_fpr: float = Field(0.01, gt=0, lt=1)
    n_search_trials: int = Field(8, ge=1)
    num_boost_round: int = Field(2_000, ge=10)
    early_stopping_rounds: int = Field(100, ge=5)
    random_state: int = 42
    explain_top_k: int = Field(3, ge=1, le=10)


class ApiSettings(BaseModel):
    """FastAPI servisi, SSE akışı ve trafik simülatörü parametreleri."""

    host: str = "127.0.0.1"
    port: int = Field(8000, ge=1, le=65535)
    stream_buffer_size: int = Field(200, ge=10, le=10_000, description="SSE geçmiş olay tampon boyutu")
    stream_heartbeat_seconds: float = Field(15.0, gt=0)
    default_alerts_limit: int = Field(50, ge=1, le=500)
    max_alerts_limit: int = Field(500, ge=1, le=5000)
    graph_default_hops: int = Field(2, ge=1, le=4)
    graph_max_hops: int = Field(3, ge=1, le=5)
    graph_max_nodes: int = Field(150, ge=10, le=1000)
    simulator_tps: float = Field(5.0, gt=0, le=200.0)
    simulator_fraud_ratio: float = Field(0.25, ge=0, le=1)


class AgentSettings(BaseModel):
    """Otonom finansal soruşturma ajanı ve SAR raporlama ayarları."""

    auto_investigate_min_score: float = Field(
        0.85,
        ge=0.0,
        le=1.0,
        description="Bu skor ve üzerindeki alarmlarda otonom soruşturma otomatik tetiklenir",
    )
    max_tool_steps: int = Field(6, ge=1, le=20, description="Soruşturma başına maksimum araç çağrısı adımı")
    history_lookback_days: int = Field(30, ge=1, le=365, description="Geçmiş işlem ve ağ analizi penceresi (gün)")
    recent_tx_limit: int = Field(20, ge=1, le=200, description="İncelenecek son işlem sayısı sınırı")
    max_evidence_ids: int = Field(25, ge=1, le=200, description="Araç çıktısında listelenecek kanıt ID üst sınırı")
    max_cycle_paths: int = Field(3, ge=1, le=20, description="Raporlanacak maksimum döngü yolu sayısı")

    # Davranışsal kanıt eşikleri (kart deneme / tutar sapması)
    micro_tx_amount_try: float = Field(25.0, gt=0, description="Mikro işlem (kart deneme) üst tutarı (TRY)")
    micro_burst_window_seconds: int = Field(3_600, ge=60, description="Mikro işlem patlaması penceresi (sn)")
    card_testing_min_burst: int = Field(3, ge=2, description="Kart deneme sinyali için min. mikro işlem sayısı")
    amount_deviation_ratio: float = Field(5.0, gt=1, description="Baz medyana göre şüpheli tutar sapma katı")

    # Ağ kanıt eşikleri (katır / halka)
    mule_min_fan_in: int = Field(4, ge=2, description="Katır hub sinyali için 24 saatlik min. farklı gönderici")
    ring_min_shared_accounts: int = Field(2, ge=1, description="Halka sinyali için ortak cihaz/IP min. hesap sayısı")

    # Tipoloji ve aksiyon karar eşikleri (öneri — insan onayı daima zorunlu)
    min_typology_signal: float = Field(
        0.35, ge=0.0, le=1.0, description="Bu sinyal gücünün altında tipoloji ANOMALOUS_ACTIVITY sayılır"
    )
    block_min_confidence: float = Field(
        0.85, ge=0.0, le=1.0, description="Tipolojiden bağımsız BLOCK_ACCOUNT önerisi için min. güven skoru"
    )

    # Opsiyonel LLM sağlayıcı (OpenAI uyumlu Chat Completions API)
    provider: Literal["local", "openai"] = "local"
    model_name: str = Field("aegis-sar-synthesizer-v1", min_length=1)
    api_key: SecretStr | None = None
    llm_base_url: str = "https://api.openai.com/v1"
    llm_timeout_seconds: float = Field(10.0, gt=0, le=120)
    temperature: float = Field(0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_provider(self) -> AgentSettings:
        if self.provider != "local" and (self.api_key is None or not self.api_key.get_secret_value()):
            raise ValueError("LLM sağlayıcı modu için agent.api_key zorunludur (veya provider='local' kullanın)")
        return self


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AEGIS_",
        env_nested_delimiter="__",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "AegisGraph"
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    generator: GeneratorSettings = Field(default_factory=GeneratorSettings)
    risk: RiskThresholds = Field(default_factory=RiskThresholds)
    features: FeatureSettings = Field(default_factory=FeatureSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)

    @property
    def FRAUD_THRESHOLD(self) -> float:  # spesifikasyondaki isim (büyük harf bilinçli)
        """Alert üretim eşiği (`risk.alert_score` için takma ad)."""
        return self.risk.alert_score


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Uygulama genelinde tek Settings örneği (testlerde `get_settings.cache_clear()`)."""
    return Settings()
