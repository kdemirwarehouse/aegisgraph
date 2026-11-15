# AegisGraph — Model Raporu

Oluşturulma: 2026-11-15 21:52 UTC

## Test seti (temporal hold-out)

| Metrik | LightGBM (hibrit) | LightGBM (graf yok) | LogReg baseline |
|---|---|---|---|
| PR-AUC | 0.9992 | 0.7112 | 0.9089 |
| ROC-AUC | 1.0000 | 0.9482 | 0.9984 |
| recall_at_1%_fpr | 1.000 | 0.693 | 0.986 |
| Operating point recall (val eşiği) | 1.000 | 0.693 | 0.979 |
| Operating point FPR | 0.0103 | 0.0100 | 0.0085 |
| Precision @ alert eşiği (0.7) | 1.000 | 0.963 | 0.826 |

## Senaryo bazlı recall (operating point)

| Senaryo | n | Recall | Recall @ alert eşiği |
|---|---|---|---|
| FRAUD_RING | 32 | 1.000 | 0.938 |
| ACCOUNT_TAKEOVER | 26 | 1.000 | 1.000 |
| MONEY_MULE | 57 | 1.000 | 0.754 |
| CARD_TESTING | 25 | 1.000 | 1.000 |

## Senaryo bazlı recall — graf özellikleri olmadan

| Senaryo | Recall (hibrit) | Recall (graf yok) |
|---|---|---|
| FRAUD_RING | 1.000 | 1.000 |
| ACCOUNT_TAKEOVER | 1.000 | 1.000 |
| MONEY_MULE | 1.000 | 0.246 |
| CARD_TESTING | 1.000 | 1.000 |

## En önemli 10 özellik (gain)

| # | Özellik | Aile | Gain payı |
|---|---|---|---|
| 1 | `amount` | tabular | 0.657 |
| 2 | `receiver_out_degree` | graph | 0.171 |
| 3 | `receiver_kyc_level` | tabular | 0.074 |
| 4 | `receiver_account_age_days` | tabular | 0.031 |
| 5 | `receiver_merchant_category` | tabular | 0.011 |
| 6 | `tx_count_10m` | tabular | 0.006 |
| 7 | `receiver_pagerank` | graph | 0.005 |
| 8 | `receiver_in_out_ratio` | graph | 0.004 |
| 9 | `device_type_code` | tabular | 0.004 |
| 10 | `device_use_count_for_account` | tabular | 0.004 |
