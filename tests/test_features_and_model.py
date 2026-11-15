"""Gün 2 testleri: sızıntısız özellikler, train/serve tutarlılığı, model performansı, SHAP, alert yazımı."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import func, select

from config.settings import FeatureSettings, GeneratorSettings, ModelSettings, Settings
from src.database.db import create_db_engine, init_db, persist_dataset, session_scope
from src.database.models import GROUND_TRUTH_COLUMNS, Alert
from src.features.graph_features import GRAPH_FEATURES
from src.features.pipeline import FEATURE_COLUMNS, FORBIDDEN_FEATURES, FeatureExtractor
from src.generator.synthetic_data import SyntheticDataGenerator, SyntheticDataset
from src.models.explainer import has_description
from src.models.scorer import RiskScorer, write_alerts
from src.models.train import TrainingResult, run_training, save_artifacts, temporal_split, threshold_at_fpr

GEN_CFG = GeneratorSettings(
    seed=11, n_customers=700, n_merchants=50, simulation_days=21,
    n_fraud_rings=6, n_account_takeovers=14, n_mule_networks=8, n_card_testing=8,
)
MODEL_CFG = ModelSettings(train_days=13, val_days=3, n_search_trials=2, num_boost_round=400,
                          early_stopping_rounds=30)
FEAT_CFG = FeatureSettings()
ALERT_THRESHOLD = 0.7


@pytest.fixture(scope="module")
def ds() -> SyntheticDataset:
    return SyntheticDataGenerator(GEN_CFG).generate()


@pytest.fixture(scope="module")
def features(ds: SyntheticDataset) -> pd.DataFrame:
    return FeatureExtractor(ds.accounts, ds.devices, FEAT_CFG).fit_transform(ds.transactions)


@pytest.fixture(scope="module")
def result(ds: SyntheticDataset, features: pd.DataFrame) -> TrainingResult:
    return run_training(features, ds.transactions, start=GEN_CFG.start_date, cfg=MODEL_CFG,
                        alert_threshold=ALERT_THRESHOLD)


@pytest.fixture(scope="module")
def scorer(result: TrainingResult) -> RiskScorer:
    return RiskScorer(result.booster, result.feature_columns, alert_threshold=ALERT_THRESHOLD)


# --------------------------------------------------------------------------- #
# Sızıntı
# --------------------------------------------------------------------------- #
def test_feature_matrix_has_no_forbidden_columns(features: pd.DataFrame) -> None:
    assert list(features.columns) == list(FEATURE_COLUMNS)
    labels = {c for cols in GROUND_TRUTH_COLUMNS.values() for c in cols}
    assert labels <= FORBIDDEN_FEATURES
    assert not FORBIDDEN_FEATURES & set(features.columns)
    assert features.index.name == "transaction_id"


def test_features_are_invariant_to_labels(ds: SyntheticDataset) -> None:
    """Etiketler karıştırılsa da özellikler bit düzeyinde aynı kalmalı."""
    tx = ds.transactions.head(4_000)
    poisoned = tx.copy()
    rng = np.random.default_rng(0)
    poisoned["is_fraud"] = rng.random(len(tx)) < 0.5
    poisoned["fraud_scenario"] = "FRAUD_RING"
    poisoned["fraud_case_id"] = "RING-999"
    a = FeatureExtractor(ds.accounts, ds.devices, FEAT_CFG).fit_transform(tx)
    b = FeatureExtractor(ds.accounts, ds.devices, FEAT_CFG).fit_transform(poisoned)
    pd.testing.assert_frame_equal(a, b)


def test_no_future_leakage_prefix_invariance(ds: SyntheticDataset, features: pd.DataFrame) -> None:
    """Bir işlemin özellikleri, sonrasındaki işlemler hiç var olmasa da aynı olmalı."""
    cutoff = pd.Timestamp(GEN_CFG.start_date) + timedelta(days=9, hours=13)
    prefix = ds.transactions[ds.transactions["timestamp"] <= cutoff]
    prefix_feats = FeatureExtractor(ds.accounts, ds.devices, FEAT_CFG).fit_transform(prefix)
    pd.testing.assert_frame_equal(prefix_feats, features.loc[prefix_feats.index])


def test_out_of_order_transaction_is_rejected(ds: SyntheticDataset) -> None:
    ex = FeatureExtractor(ds.accounts, ds.devices, FEAT_CFG)
    rows = ds.transactions.iloc[[100, 50]].to_dict("records")
    ex.transform_one(rows[0])
    with pytest.raises(ValueError, match="kronolojik"):
        ex.transform_one(rows[1])


def test_temporal_split_is_chronological(ds: SyntheticDataset, features: pd.DataFrame) -> None:
    splits = temporal_split(features, ds.transactions, start=GEN_CFG.start_date,
                            train_days=MODEL_CFG.train_days, val_days=MODEL_CFG.val_days)
    tr, va, te = (splits[k].meta["timestamp"] for k in ("train", "val", "test"))
    assert tr.max() < va.min() and va.max() < te.min()
    assert sum(len(splits[k].y) for k in splits) == len(features)
    assert all(splits[k].n_fraud > 0 for k in splits)


# --------------------------------------------------------------------------- #
# Özellik doğruluğu (elle kurulmuş mini akış)
# --------------------------------------------------------------------------- #
def _mini_extractor() -> FeatureExtractor:
    opened = datetime(2026, 1, 1)
    accounts = pd.DataFrame({
        "account_id": ["A", "B", "C", "D", "M"],
        "opened_at": [opened] * 5,
        "kyc_level": [3, 3, 1, 2, 3],
        "account_type": ["PERSONAL"] * 4 + ["MERCHANT"],
        "merchant_category": [None] * 4 + ["DIGITAL_GOODS"],
    })
    devices = pd.DataFrame({"device_id": ["d1", "d2"], "device_type": ["MOBILE_ANDROID", "DESKTOP_WEB"],
                            "is_emulator": [False, True]})
    return FeatureExtractor(accounts, devices, FEAT_CFG)


def _tx(minute: float, sender: str, receiver: str, amount: float, device: str | None = "d1",
        ip: str | None = "85.105.1.10", country: str = "TR", tid: str = "T") -> dict:
    return {
        "transaction_id": tid, "timestamp": datetime(2026, 2, 1, 12) + timedelta(minutes=minute),
        "sender_account_id": sender, "receiver_account_id": receiver, "device_id": device,
        "amount_minor": int(amount * 100), "channel": "MOBILE_APP", "transaction_type": "TRANSFER",
        "ip_address": ip, "geo_country": country,
    }


def test_velocity_and_novelty_features() -> None:
    ex = _mini_extractor()
    f1 = ex.transform_one(_tx(0, "A", "M", 10))
    f2 = ex.transform_one(_tx(3, "A", "M", 20))
    f3 = ex.transform_one(_tx(30, "A", "M", 90, device="d2", ip="45.155.3.3", country="NL"))
    assert f1["tx_count_10m"] == 0 and f1["is_new_device_for_account"] == 1
    assert math.isnan(f1["amount_to_user_avg_ratio"])
    assert f2["tx_count_10m"] == 1 and f2["amount_sum_10m"] == 10 and f2["is_new_device_for_account"] == 0
    assert f3["tx_count_10m"] == 0 and f3["tx_count_1h"] == 2 and f3["amount_sum_1h"] == 30
    assert f3["amount_to_user_avg_ratio"] == pytest.approx(90 / 15)
    assert f3["is_new_device_for_account"] == 1 and f3["device_is_emulator"] == 1
    assert f3["is_foreign_ip"] == 1 and f3["is_foreign_or_new_ip"] == 1
    assert f3["seconds_since_last_tx"] == 27 * 60
    assert f1["account_age_days"] == pytest.approx(31.5)


def test_reciprocal_cycle_and_shared_device_features() -> None:
    ex = _mini_extractor()
    ex.transform_one(_tx(0, "A", "B", 100))
    back = ex.transform_one(_tx(5, "B", "A", 100))
    assert back["has_reciprocal_flow"] == 1 and back["short_cycle_risk"] == 1.0
    assert back["shared_device_account_count"] == 2  # A ve B aynı d1 cihazını kullanıyor
    ex.transform_one(_tx(10, "B", "C", 100))
    close = ex.transform_one(_tx(15, "C", "A", 100))  # A→B→C→A
    assert close["has_reciprocal_flow"] == 0
    assert close["short_cycle_risk"] == pytest.approx(2 / 3)
    assert close["shared_device_account_count"] == 3 and close["shared_ip_account_count"] == 3
    assert close["sender_in_degree"] == 1 and close["receiver_out_degree"] == 1
    # 24 saat sonra pencere dışı → döngü sinyali kaybolur
    late = ex.transform_one(_tx(15 + 25 * 60, "B", "A", 100))
    assert late["short_cycle_risk"] == 0.0


def test_graph_features_separate_fraud_patterns(ds: SyntheticDataset, features: pd.DataFrame) -> None:
    tx = ds.transactions.set_index("transaction_id").loc[features.index]
    legit = features[~tx["is_fraud"].to_numpy()]
    ring = features[(tx["fraud_scenario"] == "FRAUD_RING").to_numpy()]
    mule = features[(tx["fraud_scenario"] == "MONEY_MULE").to_numpy()]
    assert (ring["short_cycle_risk"] > 0).mean() > 0.5
    assert (legit["short_cycle_risk"] > 0).mean() < 0.05
    assert ring["shared_device_account_count"].median() >= GEN_CFG.ring_size_min
    assert mule["receiver_unique_senders_24h"].quantile(0.75) > legit.loc[
        legit["receiver_is_merchant"] == 0, "receiver_unique_senders_24h"].quantile(0.99)


# --------------------------------------------------------------------------- #
# Model performansı
# --------------------------------------------------------------------------- #
def test_threshold_at_fpr_respects_target() -> None:
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(10_000), np.ones(100)].astype(int)
    scores = np.r_[rng.random(10_000) * 0.9, 0.5 + rng.random(100) * 0.5]
    thr = threshold_at_fpr(y, scores, 0.01)
    assert ((scores > thr) & (y == 0)).sum() <= 100


def test_model_performance_above_floor(result: TrainingResult) -> None:
    m = result.metrics["lightgbm"]["test"]
    assert m["pr_auc"] >= 0.80
    assert m["roc_auc"] >= 0.97
    assert m["recall_at_1%_fpr"] >= 0.80
    for scenario, v in m["recall_by_scenario"].items():
        if v["n"] >= 5:
            assert v["recall_at_operating"] >= 0.6, (scenario, v)


def test_lightgbm_not_worse_than_baseline(result: TrainingResult) -> None:
    assert (result.metrics["lightgbm"]["test"]["pr_auc"]
            >= result.metrics["baseline_logreg"]["test"]["pr_auc"] - 0.02)


def test_feature_importance_covers_both_families(result: TrainingResult) -> None:
    imp = result.importance
    assert set(imp["feature"]) == set(FEATURE_COLUMNS)
    assert imp["gain_share"].sum() == pytest.approx(1.0)
    top = imp.head(15)
    assert (top["family"] == "graph").any() and (top["family"] == "tabular").any()


# --------------------------------------------------------------------------- #
# SHAP + online skorlama
# --------------------------------------------------------------------------- #
def test_every_feature_has_human_readable_description() -> None:
    missing = [f for f in FEATURE_COLUMNS if not has_description(f)]
    assert not missing


def test_missing_values_have_meaningful_descriptions() -> None:
    from src.models.explainer import describe
    assert "üye işyeri değil" in describe("receiver_merchant_category", float("nan"))
    assert "ilk işlem" in describe("amount_to_user_avg_ratio", float("nan"))
    assert describe("short_cycle_risk", 2 / 3).startswith("Para, alıcıdan göndericiye 3 hesaplık")


def test_shap_values_are_additive(result: TrainingResult, scorer: RiskScorer) -> None:
    X = result.splits["test"].X.head(200)
    shap_vals = scorer.explainer.shap_values(X)
    margin = result.booster.predict(X, raw_score=True)
    np.testing.assert_allclose(shap_vals.sum(axis=1) + scorer.explainer.base_value, margin, atol=1e-4)


def test_alert_explanations_top3(result: TrainingResult, scorer: RiskScorer) -> None:
    X = result.splits["test"].X
    scores = scorer.predict(X)
    top = np.argsort(-scores)[:20]
    exps = scorer.explainer.explain(X.iloc[top], scores[top], X.index[top].tolist(), top_k=3)
    for exp in exps:
        assert 1 <= len(exp.top_factors) <= 3
        vals = [f.shap_value for f in exp.top_factors]
        assert vals == sorted(vals, reverse=True) and all(v > 0 for v in vals)
        assert all(f.description and f.family in {"graph", "tabular"} for f in exp.top_factors)
        assert 0.0 <= exp.graph_contribution_share <= 1.0
        assert exp.to_dict()["transaction_id"] == exp.transaction_id


def test_online_inference_matches_batch_and_is_fast(
    ds: SyntheticDataset, features: pd.DataFrame, scorer: RiskScorer
) -> None:
    tx = ds.transactions.sort_values(["timestamp", "transaction_id"], kind="stable")
    split_at = len(tx) - 300
    ex = FeatureExtractor(ds.accounts, ds.devices, FEAT_CFG)
    ex.fit_transform(tx.iloc[:split_at])  # geçmişi ısıt

    live = tx.iloc[split_at:split_at + 100].to_dict("records")
    results = [scorer.score_transaction(ex, rec) for rec in live]
    ids = [r["transaction_id"] for r in live]
    batch_scores = scorer.predict(features.loc[ids])
    np.testing.assert_allclose([r.risk_score for r in results], batch_scores, rtol=1e-9)

    latencies = np.array([r.latency_ms for r in results])
    assert np.median(latencies) < 50, f"medyan gecikme {np.median(latencies):.1f} ms"
    assert all(r.explanation is not None and len(r.explanation.top_factors) <= 3 for r in results)
    assert set(results[0].to_dict()) == {"transaction_id", "risk_score", "is_alert", "latency_ms", "explanation"}


def test_what_if_scoring_does_not_mutate_state(ds: SyntheticDataset) -> None:
    ex = FeatureExtractor(ds.accounts, ds.devices, FEAT_CFG)
    rows = ds.transactions.head(500).to_dict("records")
    for r in rows[:-1]:
        ex.transform_one(r)
    n = ex.n_processed
    a = ex.transform_one(rows[-1], update_state=False)
    b = ex.transform_one(rows[-1], update_state=False)
    assert a == b and ex.n_processed == n


# --------------------------------------------------------------------------- #
# Artefakt + DB
# --------------------------------------------------------------------------- #
def test_artifacts_roundtrip(result: TrainingResult, tmp_path) -> None:
    paths = save_artifacts(result, tmp_path)
    assert all(p.exists() for p in paths.values())
    loaded = RiskScorer.from_artifacts(tmp_path, Settings())
    X = result.splits["test"].X.head(50)
    np.testing.assert_allclose(loaded.predict(X), result.booster.predict(X), rtol=1e-12)
    assert "PR-AUC" in paths["report"].read_text(encoding="utf-8")


def test_extractor_state_roundtrip(ds: SyntheticDataset, tmp_path) -> None:
    rows = ds.transactions.head(2_001).to_dict("records")
    ex = FeatureExtractor(ds.accounts, ds.devices, FEAT_CFG)
    for r in rows[:-1]:
        ex.transform_one(r)
    restored = FeatureExtractor.load(ex.save(tmp_path / "fx.joblib"))
    assert restored.transform_one(rows[-1]) == ex.transform_one(rows[-1])


def test_write_alerts_persists_shap_reasons(ds: SyntheticDataset, result: TrainingResult, scorer: RiskScorer,
                                            tmp_path) -> None:
    engine = create_db_engine(f"sqlite:///{(tmp_path / 'alerts.db').as_posix()}")
    try:
        init_db(engine)
        persist_dataset(ds, engine)
        X = result.splits["test"].X
        expected = int((scorer.predict(X) >= ALERT_THRESHOLD).sum())
        assert expected > 0
        assert write_alerts(scorer, X, engine) == expected
        assert write_alerts(scorer, X, engine) == expected  # idempotent
        with session_scope(engine) as session:
            assert session.scalar(select(func.count()).select_from(Alert)) == expected
            alert = session.scalars(select(Alert).order_by(Alert.risk_score.desc())).first()
            assert alert.risk_score >= ALERT_THRESHOLD and alert.status.value == "OPEN"
            assert 1 <= len(alert.reasons) <= 3
            assert {"feature", "value", "shap_value", "family", "description"} <= set(alert.reasons[0])
            assert 0 <= alert.graph_score <= 1
    finally:
        engine.dispose()


def test_graph_feature_names_are_registered() -> None:
    assert set(GRAPH_FEATURES) <= set(FEATURE_COLUMNS)
