# CLAUDE.md — AegisGraph

Hibrit (ML + Graf) gerçek zamanlı dolandırıcılık tespit sistemi. 7 günlük plan; her oturum bu
dosyayı okuyarak başlar. Mimari kararlar burada; ayrıntılı domain kuralları `.antigravity/rules.md`.

## Yol haritası
| Gün | Kapsam | Modül |
|---|---|---|
| 1 ✅ | Konfigürasyon, veri modeli, SQLite, sentetik veri motoru, testler | `config/`, `src/database/`, `src/generator/` |
| 2 ✅ | NetworkX graf özellikleri + LightGBM, temporal değerlendirme, SHAP, Alert yazımı | `src/features/`, `src/models/` |
| 3 | FastAPI skorlama + SSE ile canlı işlem/alert akışı | `src/api/` |
| 4 | Otonom soruşturma ajanı (Alert → kanıt toplama → rapor) | `src/agent/` |
| 5 | Dashboard | — |

## Komutlar (Windows, proje kökünden)
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.generator.synthetic_data          # CSV + SQLite üret
.\.venv\Scripts\python.exe -m src.models.train                    # özellik + eğitim + artifacts/ + alerts
.\.venv\Scripts\python.exe -m pytest                                # tüm testler (~60 sn)
.\.venv\Scripts\python.exe -m ruff check .
```

## Mimari kurallar
- **Konfigürasyon tek kaynak:** tüm eşikler/yollar `config/settings.py` → `get_settings()`.
  Kod içine sabit eşik (0.7, 50_000 vb.) yazılmaz. Yeni parametre = yeni alan + varsayılan + doğrulayıcı.
- **Engine yalnızca `src/database/db.py` içinde oluşturulur** (`create_db_engine` / `get_engine`).
- **Şema tek kaynak:** kolon sırası ve tipleri Pydantic `*Record` modellerinden gelir
  (`record_fields(table)`). ORM tablosu, DataFrame ve Pydantic modeli birlikte güncellenir.
- **Para:** `amount_minor` (kuruş, int64/BigInteger). Float tutar DB'ye yazılmaz.
- **Zaman:** UTC, timezone-naive `datetime`. İşlem ID'leri kronolojiktir (`TX-00000001`).
- **Etiket sızıntısı yasak:** `GROUND_TRUTH_COLUMNS` (`is_fraud`, `fraud_scenario`, `fraud_case_id`,
  `is_fraudster`) feature üretiminde ASLA kullanılmaz; sadece hedef/değerlendirme içindir.
  Hesap/cihaz ID'leri bilinçli olarak rastgele permütasyondur — ID'den özellik türetilmez.
- **Zaman bazlı split:** model değerlendirmesi rastgele değil, zamana göre (train < val < test)
  yapılır; graf özellikleri yalnızca işlem anından *önceki* veriden hesaplanır (point-in-time).
- **Determinizm:** tüm rastgelelik `np.random.default_rng(seed)` üzerinden; global `random`/`np.random.*` yok.
- Enum'lar `StrEnum`, isim == değer, DB'de string (`native_enum=False`).

## Özellik ve model mimarisi (Gün 2)
- `FeatureExtractor` (src/features/pipeline.py) **durumlu ve kronolojiktir**: her işlem için önce
  `features()` (yalnızca geçmiş), sonra `update()`. Batch `fit_transform` satır satır `transform_one`
  çağırır → train/serve aynı kod yolu. Geriye giden zaman damgası `ValueError`.
- Girdi `TX_INPUT_COLUMNS` ile süzülür; etiketler extractor'a fiziksel olarak ulaşmaz.
  Yeni özellik = `TABULAR_FEATURES`/`GRAPH_FEATURES` + `explainer._VALUE/_BINARY` açıklaması + test.
- PageRank/clustering günlük snapshot (gün içi değer = önceki gün sonu); dereceler, ortak cihaz/IP,
  döngü (BFS, 24 sa pencere, `max_cycle_hops`) işlem anında artımlı.
- Split: `ModelSettings.train_days/val_days` (21/4/5). Hiperparametre ve operating-point eşiği
  (hedef FPR) **validasyonda** seçilir, test yalnızca raporlanır.
- Artefaktlar `artifacts/`: `model.txt`, `feature_columns.json`, `metrics.json`, `report.md`,
  `feature_importance.csv`, `feature_extractor.joblib` (Gün 3 online skorlama için sıcak durum).
  Model yüklenirken özellik listesi `FEATURE_COLUMNS` ile birebir doğrulanır (şema kayması koruması).
- Alert: `risk_score >= settings.FRAUD_THRESHOLD` (`risk.alert_score`); `reasons` = SHAP top-3
  (feature, value, shap_value, family, description); `graph_score` = pozitif SHAP içindeki graf payı.
- Sentetik veride "hard negative"lar (meşru seyahat/VPN, telefon değişimi, aile cihazı, mikro-ödeme
  seansı, yeni meşru hesaplar) ve zor fraud (yurt içi saldırgan, devşirilmiş eski mule hesapları) vardır;
  bunlar kaldırılırsa model tek-özellik kısayolları öğrenir.

## Sentetik veri senaryoları (Gün 1)
| Senaryo | İmza (modelin yakalaması beklenen sinyal) |
|---|---|
| `FRAUD_RING` | Yeni + düşük KYC hesaplar, ortak cihaz & /24 IP alt ağı, dakikalar içinde dairesel transfer, her adımda azalan tutar |
| `ACCOUNT_TAKEOVER` | Kurbanda daha önce görülmemiş cihaz, yabancı IP/ülke, medyanın 8-30 katı transferler; saldırgan cihazı birden çok kurbanda |
| `MONEY_MULE` | 15-40 farklı kurbandan tek hub'a fan-in, 1-6 saat sonra cash-out hesaplarına %90-97 fan-out |
| `CARD_TESTING` | Dijital işyerlerinde saniyeler arayla 1-20 TRY denemeler → büyük e-ticaret alışverişi |

Fraud oranı varsayılan ayarlarla ~%1.0 (gerçekçi dengesizlik). Halka üyeleri kamuflaj için etiketsiz
normal alışveriş de yapar.

## Kod standartları
- Python ≥ 3.11, tam tip ipucu, `from __future__ import annotations`.
- Her yeni modül için `tests/` altında pytest; yeni fraud senaryosu = topolojisini doğrulayan test.
- Docstring ve log mesajları Türkçe, tanımlayıcılar İngilizce.
- `print` yalnızca CLI özetinde; diğer her yerde `logging`.
- Gün N iskeleti boş `__init__.py`'dır; önceki günlerin public API'si (`SyntheticDataset`,
  `persist_dataset`, `load_table`, `get_settings`) kırılmadan genişletilir.
