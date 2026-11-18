"""Temporal split + dengesiz veri için LightGBM eğitimi ve bankacılık odaklı değerlendirme.

Akış:
    DB → FeatureExtractor.fit_transform (point-in-time) → kronolojik train/val/test
    → Logistic Regression baseline + LightGBM rastgele hiperparametre araması (val PR-AUC)
    → test metrikleri (PR-AUC, ROC-AUC, Recall@FPR, senaryo bazlı recall) → artifacts/
    → test penceresindeki yüksek riskli işlemler için SHAP açıklamalı Alert kayıtları

Kullanım:
    python -m src.models.train                # eğit + artefaktları kaydet + alert yaz
    python -m src.models.train --no-alerts
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config.settings import ModelSettings, get_settings
from src.database.models import FraudScenario
from src.features.graph_features import GRAPH_FEATURES
from src.features.pipeline import FEATURE_COLUMNS, FeatureExtractor

logger = logging.getLogger(__name__)

SCENARIOS: tuple[str, ...] = tuple(s.value for s in FraudScenario if s is not FraudScenario.NONE)
META_COLUMNS = ("timestamp", "is_fraud", "fraud_scenario", "amount_minor")

SEARCH_SPACE: dict[str, list[Any]] = {
    "num_leaves": [15, 31, 63],
    "min_child_samples": [10, 30, 80],
    "learning_rate": [0.03, 0.05, 0.1],
    "feature_fraction": [0.7, 0.9],
    "reg_lambda": [0.0, 1.0, 5.0],
    "spw_power": [0.5, 1.0],  # scale_pos_weight = (neg/pos) ** spw_power
}
DEFAULT_TRIAL: dict[str, Any] = {
    "num_leaves": 31, "min_child_samples": 30, "learning_rate": 0.05,
    "feature_fraction": 0.9, "reg_lambda": 1.0, "spw_power": 0.5,
}


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #
@dataclass
class DataSplit:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    meta: pd.DataFrame  # sadece değerlendirme: timestamp, is_fraud, fraud_scenario, amount_minor

    @property
    def n_fraud(self) -> int:
        return int(self.y.sum())


def temporal_split(
    features: pd.DataFrame,
    transactions: pd.DataFrame,
    *,
    start: datetime,
    train_days: int,
    val_days: int,
) -> dict[str, DataSplit]:
    """Kronolojik bölme: [start, +train) → train, [+train, +train+val) → val, sonrası → test."""
    meta = transactions.set_index("transaction_id").loc[features.index, list(META_COLUMNS)]
    ts = meta["timestamp"]
    train_end = pd.Timestamp(start) + timedelta(days=train_days)
    val_end = train_end + timedelta(days=val_days)
    masks = {
        "train": (ts < train_end).to_numpy(),
        "val": ((ts >= train_end) & (ts < val_end)).to_numpy(),
        "test": (ts >= val_end).to_numpy(),
    }
    splits: dict[str, DataSplit] = {}
    for name, mask in masks.items():
        if not mask.any():
            raise ValueError(f"'{name}' bölmesi boş; train_days/val_days değerlerini kontrol edin")
        splits[name] = DataSplit(
            name=name,
            X=features.loc[mask, list(FEATURE_COLUMNS)],
            y=meta["is_fraud"].to_numpy()[mask].astype(np.int8),
            meta=meta.loc[mask],
        )
    return splits


# --------------------------------------------------------------------------- #
# Metrikler
# --------------------------------------------------------------------------- #
def threshold_at_fpr(y: np.ndarray, scores: np.ndarray, fpr: float) -> float:
    """`score > eşik` kuralıyla yanlış pozitif oranı ≤ fpr olacak en düşük eşik."""
    neg = np.sort(scores[y == 0])
    allowed_fp = int(np.floor(fpr * len(neg)))
    if allowed_fp >= len(neg):
        return -np.inf
    return float(neg[len(neg) - allowed_fp - 1])


def _confusion(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    return {
        "flagged": int(pred.sum()),
        "recall": tp / pos if pos else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "fpr": fp / neg if neg else float("nan"),
    }


def evaluate(
    y: np.ndarray,
    scores: np.ndarray,
    meta: pd.DataFrame,
    *,
    target_fpr: float,
    operating_threshold: float,
    alert_threshold: float,
) -> dict[str, Any]:
    """Bankacılık metrikleri. `operating_threshold` validasyonda seçilir (dürüst test)."""
    oracle_thr = threshold_at_fpr(y, scores, target_fpr)
    at_op = (scores > operating_threshold).astype(int)
    at_alert = (scores >= alert_threshold).astype(int)
    amounts = meta["amount_minor"].to_numpy()
    fraud_amount = amounts[y == 1].sum()

    by_scenario: dict[str, dict[str, float]] = {}
    scen = meta["fraud_scenario"].to_numpy()
    for s in SCENARIOS:
        mask = scen == s
        n = int(mask.sum())
        by_scenario[s] = {
            "n": n,
            "recall_at_operating": float(at_op[mask].mean()) if n else float("nan"),
            "recall_at_alert": float(at_alert[mask].mean()) if n else float("nan"),
        }

    return {
        "n": len(y),
        "n_fraud": int(y.sum()),
        "fraud_rate": float(y.mean()),
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        f"recall_at_{target_fpr:.0%}_fpr": _confusion(y, (scores > oracle_thr).astype(int))["recall"],
        "operating_point": {"threshold": operating_threshold, "target_fpr": target_fpr,
                            **_confusion(y, at_op)},
        "alert_point": {"threshold": alert_threshold, **_confusion(y, at_alert)},
        "amount_weighted_recall_at_operating": (
            float(amounts[(y == 1) & (at_op == 1)].sum() / fraud_amount) if fraud_amount else float("nan")
        ),
        "recall_by_scenario": by_scenario,
    }


# --------------------------------------------------------------------------- #
# Modeller
# --------------------------------------------------------------------------- #
def train_baseline(train: DataSplit) -> Any:
    """Karşılaştırma için sınıf ağırlıklı lojistik regresyon."""
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=3_000),
    )
    model.fit(train.X.to_numpy(), train.y)
    return model


def _sample_trials(n: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    trials = [dict(DEFAULT_TRIAL)]
    while len(trials) < n:
        trial = {k: v[int(rng.integers(len(v)))] for k, v in SEARCH_SPACE.items()}
        if trial not in trials:
            trials.append(trial)
    return trials[:n]


def _lgb_params(trial: dict[str, Any], base_spw: float, seed: int) -> dict[str, Any]:
    params = {k: v for k, v in trial.items() if k != "spw_power"}
    params.update(
        objective="binary",
        metric="average_precision",
        scale_pos_weight=base_spw ** trial["spw_power"],
        bagging_fraction=0.8,
        bagging_freq=1,
        seed=seed,
        deterministic=True,
        force_col_wise=True,
        num_threads=4,
        verbose=-1,
    )
    return params


def train_lightgbm(
    train: DataSplit, val: DataSplit, cfg: ModelSettings
) -> tuple[lgb.Booster, dict[str, Any], list[dict[str, Any]]]:
    """Val PR-AUC'ye göre rastgele arama + early stopping. En iyi modeli döndürür."""
    pos = max(train.n_fraud, 1)
    base_spw = (len(train.y) - pos) / pos
    dtrain = lgb.Dataset(train.X, label=train.y, free_raw_data=False, params={"feature_pre_filter": False})
    dval = lgb.Dataset(val.X, label=val.y, reference=dtrain, free_raw_data=False,
                       params={"feature_pre_filter": False})

    best: tuple[float, lgb.Booster, dict[str, Any]] | None = None
    history = []
    for i, trial in enumerate(_sample_trials(cfg.n_search_trials, cfg.random_state)):
        params = _lgb_params(trial, base_spw, cfg.random_state)
        booster = lgb.train(
            params, dtrain, num_boost_round=cfg.num_boost_round, valid_sets=[dval],
            callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
        )
        val_ap = float(average_precision_score(val.y, booster.predict(val.X, num_iteration=booster.best_iteration)))
        history.append({**trial, "best_iteration": booster.best_iteration, "val_pr_auc": val_ap})
        logger.info("Deneme %d/%d: val PR-AUC=%.4f (%d ağaç) %s",
                    i + 1, cfg.n_search_trials, val_ap, booster.best_iteration, trial)
        if best is None or val_ap > best[0]:
            best = (val_ap, booster, params | {"best_iteration": booster.best_iteration})

    assert best is not None
    _, booster, params = best
    # Yalnızca en iyi iterasyona kadar olan ağaçları içeren bağımsız model
    final = lgb.Booster(model_str=booster.model_to_string(num_iteration=booster.best_iteration))
    return final, params, history


def train_tabular_only_ablation(train: DataSplit, val: DataSplit, cfg: ModelSettings) -> lgb.Booster:
    """Graf özellikleri olmadan aynı LightGBM (hibrit mimarinin katkısını ölçmek için)."""
    cols = [c for c in FEATURE_COLUMNS if c not in GRAPH_FEATURES]
    pos = max(train.n_fraud, 1)
    params = _lgb_params(DEFAULT_TRIAL, (len(train.y) - pos) / pos, cfg.random_state)
    dtrain = lgb.Dataset(train.X[cols], label=train.y, params={"feature_pre_filter": False})
    dval = lgb.Dataset(val.X[cols], label=val.y, reference=dtrain, params={"feature_pre_filter": False})
    return lgb.train(params, dtrain, num_boost_round=cfg.num_boost_round, valid_sets=[dval],
                     callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)])


# --------------------------------------------------------------------------- #
# Orkestrasyon
# --------------------------------------------------------------------------- #
@dataclass
class TrainingResult:
    booster: lgb.Booster
    params: dict[str, Any]
    search_history: list[dict[str, Any]]
    operating_threshold: float
    alert_threshold: float
    metrics: dict[str, Any]
    importance: pd.DataFrame
    splits: dict[str, DataSplit] = field(repr=False)
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))


def run_training(
    features: pd.DataFrame,
    transactions: pd.DataFrame,
    *,
    start: datetime,
    cfg: ModelSettings,
    alert_threshold: float,
) -> TrainingResult:
    splits = temporal_split(features, transactions, start=start,
                            train_days=cfg.train_days, val_days=cfg.val_days)
    train, val, test = splits["train"], splits["val"], splits["test"]
    if train.meta["timestamp"].max() >= val.meta["timestamp"].min():
        raise RuntimeError("Temporal split ihlali")
    for s in (train, val, test):
        logger.info("%-5s: %6d işlem, %4d fraud (%.3f%%)", s.name, len(s.y), s.n_fraud, 100 * s.y.mean())

    baseline = train_baseline(train)
    booster, params, history = train_lightgbm(train, val, cfg)

    val_scores = booster.predict(val.X)
    operating = threshold_at_fpr(val.y, val_scores, cfg.target_fpr)
    eval_kw = dict(target_fpr=cfg.target_fpr, operating_threshold=operating, alert_threshold=alert_threshold)

    base_val = baseline.predict_proba(val.X.to_numpy())[:, 1]
    base_op = threshold_at_fpr(val.y, base_val, cfg.target_fpr)
    base_test = baseline.predict_proba(test.X.to_numpy())[:, 1]

    ablation = train_tabular_only_ablation(train, val, cfg)
    tab_cols = ablation.feature_name()
    abl_val = ablation.predict(val.X[tab_cols], num_iteration=ablation.best_iteration)
    abl_test = ablation.predict(test.X[tab_cols], num_iteration=ablation.best_iteration)

    metrics = {
        "lightgbm": {
            "val": evaluate(val.y, val_scores, val.meta, **eval_kw),
            "test": evaluate(test.y, booster.predict(test.X), test.meta, **eval_kw),
        },
        "baseline_logreg": {
            "test": evaluate(test.y, base_test, test.meta, target_fpr=cfg.target_fpr,
                             operating_threshold=base_op, alert_threshold=alert_threshold),
        },
        "lightgbm_tabular_only": {
            "test": evaluate(test.y, abl_test, test.meta, target_fpr=cfg.target_fpr,
                             operating_threshold=threshold_at_fpr(val.y, abl_val, cfg.target_fpr),
                             alert_threshold=alert_threshold),
        },
        "split": {
            s.name: {"start": str(s.meta["timestamp"].min()), "end": str(s.meta["timestamp"].max()),
                     "n": len(s.y), "n_fraud": s.n_fraud}
            for s in (train, val, test)
        },
    }

    gain = booster.feature_importance(importance_type="gain")
    importance = pd.DataFrame({
        "feature": booster.feature_name(),
        "gain": gain,
        "gain_share": gain / gain.sum(),
        "splits": booster.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False).reset_index(drop=True)
    importance["family"] = np.where(importance["feature"].isin(GRAPH_FEATURES), "graph", "tabular")

    return TrainingResult(booster, params, history, operating, alert_threshold, metrics, importance, splits)


def _report_markdown(result: TrainingResult) -> str:
    m = result.metrics["lightgbm"]["test"]
    b = result.metrics["baseline_logreg"]["test"]
    t = result.metrics["lightgbm_tabular_only"]["test"]
    fpr_key = next(k for k in m if k.startswith("recall_at_"))
    lines = [
        "# AegisGraph — Model Raporu", "",
        f"Oluşturulma: {datetime.now(UTC):%Y-%m-%d %H:%M} UTC", "",
        "## Test seti (temporal hold-out)", "",
        "| Metrik | LightGBM (hibrit) | LightGBM (graf yok) | LogReg baseline |", "|---|---|---|---|",
        f"| PR-AUC | {m['pr_auc']:.4f} | {t['pr_auc']:.4f} | {b['pr_auc']:.4f} |",
        f"| ROC-AUC | {m['roc_auc']:.4f} | {t['roc_auc']:.4f} | {b['roc_auc']:.4f} |",
        f"| {fpr_key} | {m[fpr_key]:.3f} | {t[fpr_key]:.3f} | {b[fpr_key]:.3f} |",
        f"| Operating point recall (val eşiği) | {m['operating_point']['recall']:.3f} "
        f"| {t['operating_point']['recall']:.3f} | {b['operating_point']['recall']:.3f} |",
        f"| Operating point FPR | {m['operating_point']['fpr']:.4f} | {t['operating_point']['fpr']:.4f} "
        f"| {b['operating_point']['fpr']:.4f} |",
        f"| Precision @ alert eşiği ({m['alert_point']['threshold']}) | {m['alert_point']['precision']:.3f} "
        f"| {t['alert_point']['precision']:.3f} | {b['alert_point']['precision']:.3f} |",
        "", "## Senaryo bazlı recall (operating point)", "",
        "| Senaryo | n | Recall | Recall @ alert eşiği |", "|---|---|---|---|",
    ]
    for s, v in m["recall_by_scenario"].items():
        lines.append(f"| {s} | {v['n']} | {v['recall_at_operating']:.3f} | {v['recall_at_alert']:.3f} |")
    lines += ["", "## Senaryo bazlı recall — graf özellikleri olmadan", "",
              "| Senaryo | Recall (hibrit) | Recall (graf yok) |", "|---|---|---|"]
    for s, v in m["recall_by_scenario"].items():
        lines.append(f"| {s} | {v['recall_at_operating']:.3f} | "
                     f"{t['recall_by_scenario'][s]['recall_at_operating']:.3f} |")
    lines += ["", "## En önemli 10 özellik (gain)", "", "| # | Özellik | Aile | Gain payı |", "|---|---|---|---|"]
    for i, row in result.importance.head(10).iterrows():
        lines.append(f"| {i + 1} | `{row.feature}` | {row.family} | {row.gain_share:.3f} |")
    return "\n".join(lines) + "\n"


def save_artifacts(result: TrainingResult, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "model": out_dir / "model.txt",
        "features": out_dir / "feature_columns.json",
        "metrics": out_dir / "metrics.json",
        "importance": out_dir / "feature_importance.csv",
        "report": out_dir / "report.md",
    }
    result.booster.save_model(str(paths["model"]))
    paths["features"].write_text(json.dumps({
        "feature_columns": result.feature_columns,
        "operating_threshold": result.operating_threshold,
        "alert_threshold": result.alert_threshold,
        "params": result.params,
        "trained_at": datetime.now(UTC).isoformat(),
    }, indent=2, default=str), encoding="utf-8")
    paths["metrics"].write_text(json.dumps(
        {**result.metrics, "search_history": result.search_history}, indent=2, default=float
    ), encoding="utf-8")
    result.importance.to_csv(paths["importance"], index=False)
    paths["report"].write_text(_report_markdown(result), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    from src.database.db import get_engine, load_table
    from src.models.scorer import RiskScorer, write_alerts

    parser = argparse.ArgumentParser(description="AegisGraph model eğitimi")
    parser.add_argument("--no-alerts", action="store_true", help="alerts tablosuna yazma")
    args = parser.parse_args(argv)

    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    engine = get_engine()
    try:
        accounts, devices, transactions = (load_table(t, engine) for t in ("accounts", "devices", "transactions"))
    except Exception as exc:
        raise SystemExit("Veritabanı okunamadı. Önce: python -m src.generator.synthetic_data") from exc

    t0 = time.perf_counter()
    extractor = FeatureExtractor(accounts, devices, settings.features)
    features = extractor.fit_transform(transactions)
    logger.info("Özellik çıkarımı: %d işlem × %d özellik, %.1fs",
                *features.shape, time.perf_counter() - t0)

    result = run_training(features, transactions, start=settings.generator.start_date,
                          cfg=settings.model, alert_threshold=settings.FRAUD_THRESHOLD)
    out_dir = settings.model.artifacts_dir
    save_artifacts(result, out_dir)
    extractor.save(out_dir / "feature_extractor.joblib")  # online skorlama için sıcak durum

    print((out_dir / "report.md").read_text(encoding="utf-8"))
    if not args.no_alerts:
        scorer = RiskScorer.from_artifacts(out_dir, settings)
        n = write_alerts(scorer, result.splits["test"].X, engine)
        print(f"alerts tablosuna {n} kayıt yazıldı (eşik {scorer.alert_threshold}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
