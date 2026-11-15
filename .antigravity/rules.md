# AegisGraph — Domain ve Kod Kuralları

Bu dosya, projede çalışan her IDE ajanı / SDK için bağlayıcıdır. Mimari özet: `CLAUDE.md`.

## 1. Domain sözlüğü
- **Account**: `PERSONAL`, `BUSINESS`, `MERCHANT`. TR IBAN (mod-97 geçerli), KYC seviyesi 1-3.
- **Device**: parmak izi tekil; bir cihaz birden çok hesaba bağlanabilir (`account_devices`).
- **Transaction**: gönderen → alıcı, `amount_minor` (kuruş), kanal (`MOBILE_APP`, `INTERNET_BANKING`,
  `POS`, `ECOMMERCE`), tür (`TRANSFER`, `CARD_PURCHASE`, `BILL_PAYMENT`). POS işleminde cihaz/IP yoktur.
- **Alert**: skor ∈ [0,1], durum akışı `OPEN → INVESTIGATING → (ESCALATED) → CONFIRMED_FRAUD | FALSE_POSITIVE → CLOSED`.
- **Fraud case**: koordineli olay kimliği (`RING-001`, `ATO-003`, `MULE-002`, `CT-005`, `ATO-CREW`).

## 2. Değişmezler (invariant) — ihlal eden kod birleştirilmez
1. `amount_minor > 0`, tamsayı. Float para hesabı yalnızca gösterim/özellik katmanında.
2. `sender_account_id != receiver_account_id`.
3. `is_fraud == (fraud_scenario != NONE) == (fraud_case_id is not None)`.
4. Hesap açılış tarihinden önce işlem olamaz.
5. `account_devices.first_seen <= last_seen`; işlemde kullanılan her (hesap, cihaz) çifti kayıtlıdır.
6. Aynı seed → aynı veri (bit düzeyinde).

## 3. ML / graf kuralları
- Etiket kolonları (`GROUND_TRUTH_COLUMNS`) özellik olarak kullanılamaz.
- Hesap/cihaz ID'leri, ID sırası veya `fraud_case_id` önekinden özellik türetilmez.
- Point-in-time doğruluğu: t anındaki işlemin özellikleri yalnızca `< t` verisiyle hesaplanır.
- Değerlendirme zaman bazlı split ile; birincil metrikler PR-AUC, Recall@FPR≤%1, tutar ağırlıklı yakalama.
- Graf özellik adayları: hesap–cihaz bipartit derece, ortak cihaz komşuluğu, döngü tespiti,
  fan-in/fan-out oranı, PageRank, bağlı bileşen boyutu, işlem hızı (velocity) pencereleri.

## 4. Kod kuralları
- Eşikler yalnızca `config/settings.py` (`RiskThresholds`) içinde.
- DB erişimi `src/database/db.py` üzerinden; ham SQL yalnızca okuma/teşhis amaçlı.
- Tüm rastgelelik seed'li `numpy.random.Generator` ile.
- Yeni senaryo/özellik ⇒ pytest testi zorunlu. `pytest` yeşil olmadan iş bitmiş sayılmaz.
- PII: veri tamamen sentetik; gerçek müşteri verisi repoya ve loglara girmez. Loglarda IBAN maskelenir
  (Gün 3+ API katmanında `TR** **** **** **** **** **12 34` formatı).

## 5. Ajan (Gün 4) kuralları
- Ajan karar vermez, **öneri** üretir; `CONFIRMED_FRAUD` / blok kararı insan analist onayı gerektirir.
- Her bulgu kaynak işlem/kenar ID'leri ile kanıtlanır (açıklanabilirlik).
- Ajan yalnızca okuma araçlarına sahiptir; DB'ye yalnızca `alerts` tablosundaki not/durum alanlarını yazar.
