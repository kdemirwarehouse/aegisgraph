# 🛡️ AegisGraph

**Hibrit (ML + Graf Analizi) ve otonom gerçek zamanlı dolandırıcılık tespit sistemi.**

Klasik kural motorları ve tekil işlem modelleri, *koordineli* dolandırıcılığı (fraud halkaları, mule ağları,
paylaşılan cihazlar) göremez. AegisGraph her işlemi iki açıdan değerlendirir:

1. **ML skoru**: işlem ve hesap davranışı özellikleri üzerinde LightGBM
2. **Graf skoru**: hesap–cihaz–işlem ağında döngüler, ortak cihazlar, fan-in/fan-out ve bileşen yapısı

Yüksek riskli işlemler **Alert** üretir. Otonom bir **soruşturma ajanı** kanıt toplayıp analiste
açıklanabilir bir rapor sunar.

```
 İşlem akışı ──► Feature store ──► LightGBM ─┐
       │                                     ├──► Risk skoru ──► Alert ──► Soruşturma ajanı ──► Analist
       └──────► Graf (NetworkX) ──► Graf ────┘                       (SSE)                     (Dashboard)
```

## 7 günlük yol haritası
| Gün | Teslimat | Durum |
|---|---|---|
| 1 | Mimari, konfigürasyon, veri modeli, SQLite, sentetik veri motoru, testler | ✅ |
| 2 | Graf özellik mühendisliği (NetworkX) + LightGBM + SHAP açıklanabilirlik | ✅ |
| 3 | FastAPI gerçek zamanlı skorlama + SSE streaming | ⏳ |
| 4 | Otonom soruşturma ajanı | ⏳ |
| 5 | Dashboard | ⏳ |
| 6-7 | Sertleştirme, değerlendirme, demo | ⏳ |

## Kurulum (Gün 1)
Gereksinim: Python ≥ 3.11

```powershell
cd aegisgraph
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env                 # isteğe bağlı
```

### Sentetik veri üretimi
```powershell
python -m src.generator.synthetic_data                          # varsayılan: 2.000 müşteri, 30 gün
python -m src.generator.synthetic_data --customers 5000 --days 60 --seed 7
python -m src.generator.synthetic_data --no-db                   # sadece CSV
```
Çıktılar:
- `data/raw/{accounts,devices,account_devices,transactions}.csv`
- `data/aegisgraph.db` (SQLite: `accounts`, `devices`, `account_devices`, `transactions`, `alerts`)

### Model eğitimi (Gün 2)
```powershell
python -m src.models.train            # özellik çıkarımı → eğitim → artifacts/ → alerts tablosu
```
Sonuçlar `artifacts/report.md` dosyasında. Test seti = son 5 gün (temporal hold-out):

| Metrik | LightGBM (hibrit) | LightGBM (graf yok) | LogReg baseline |
|---|---|---|---|
| PR-AUC | 0.9992 | 0.7112 | 0.9089 |
| ROC-AUC | 1.0000 | 0.9482 | 0.9984 |
| Recall @ %1 FPR | 1.000 | 0.693 | 0.986 |

Graf özellikleri çıkarıldığında Money Mule recall'u **1.00 → 0.25**'e düşüyor: hibrit mimarinin
katkısı en net bu senaryoda görülür. Not: Veri sentetik olduğundan metrikler gerçek bankacılık
verisine göre iyimser bir üst sınırdır.

Her alert için SHAP ile en çok riski artıran 3 faktör Türkçe olarak saklanır:
```
TX-00067511  MONEY_MULE  risk=0.962  graf payı=%43
  +3.63  Alıcı bir üye işyeri değil (kişi/kurum hesabına transfer)
  +2.60  İşlem tutarı 7,964.76 TRY
  +2.55  Alıcı şimdiye kadar 0 farklı hesaba para gönderdi
```

### Testler
```powershell
pytest
```

## Veri modeli
| Tablo | Açıklama |
|---|---|
| `accounts` | Bireysel / kurumsal / üye işyeri hesapları, geçerli TR IBAN, KYC seviyesi |
| `devices` | Cihaz parmak izi, tip, emülatör bayrağı |
| `account_devices` | Hesap ↔ cihaz kenarları (graf için bipartit ilişki) |
| `transactions` | Tutar (kuruş), kanal, IP, konum + ground-truth etiketleri |
| `alerts` | Risk skoru, graf katkı payı ve SHAP gerekçeleri (Gün 2) |

## Fraud senaryoları
| Senaryo | Nasıl modellendi |
|---|---|
| **Fraud Ring** | 4-8 yeni hesap, ortak cihaz ve IP alt ağı; kurbandan giren para dakikalar içinde A→B→…→A döngüsünde birkaç tur döner |
| **Account Takeover** | Kurbanın hesabına yeni cihaz + yabancı IP ile girilir, normal tutarın 8-30 katı transferler; saldırgan cihazı birden fazla kurbanda tekrar görülür |
| **Money Mule** | 15-40 kurbandan tek hub hesaba fan-in, saatler içinde 3-6 cash-out hesabına fan-out |
| **Card Testing** | Saniyeler arayla 1-20 TRY'lik denemeler, ardından büyük e-ticaret alışverişi |

Varsayılan ayarlarla ~81 bin işlemin yaklaşık **%1.0'ı** fraud (gerçekçi sınıf dengesizliği).
Meşru trafikte de fraud'a benzeyen davranışlar (seyahat, VPN, telefon değişimi, aile cihazı,
mikro-ödeme seansları) bulunur; model tek bir kısayol özelliğe yaslanamaz.
Hesap ve cihaz ID'leri rastgele permüte edilir; fraud hesapları ID sırasından tanınamaz.

## Konfigürasyon
Tüm parametreler `config/settings.py` içinde (Pydantic `BaseSettings`), `AEGIS_` önekli ortam
değişkenleriyle ezilebilir (iç içe alanlar için `__`):
```
AEGIS_GENERATOR__N_CUSTOMERS=5000
AEGIS_RISK__ALERT_SCORE=0.65
```

## Proje yapısı
```
aegisgraph/
├── .antigravity/rules.md      # Domain + kod kuralları (ajanlar için bağlayıcı)
├── CLAUDE.md                  # Mimari ve teknik kurallar
├── config/settings.py         # Merkezi konfigürasyon
├── data/raw/                  # Üretilen veri (git'e girmez)
├── src/
│   ├── database/              # ORM + Pydantic şemalar, engine, toplu yükleme
│   ├── generator/             # Sentetik veri motoru
│   ├── features/              # Gün 2: tabular + graf özellikleri, FeatureExtractor
│   ├── models/                # Gün 2: train, explainer (SHAP), scorer (alert)
│   ├── api/                   # Gün 3
│   └── agent/                 # Gün 4
├── artifacts/                 # report.md, metrics.json (model ikilileri git dışı)
└── tests/                     # test_generator.py, test_features_and_model.py
```

> Tüm veriler sentetiktir; gerçek kişi veya hesap bilgisi içermez.
