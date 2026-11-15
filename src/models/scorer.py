"""Risk skorlama ve SHAP açıklamalı Alert üretimi.

Batch:   `write_alerts(scorer, X, engine)` → eşik üstü işlemler `alerts` tablosuna
Online:  `scorer.score_transaction(extractor, tx)` → tek işlem, milisaniye mertebesinde (Gün 3 API)
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sqlalchemy import Engine, delete

from config.settings import Settings, get_settings
from src.database.db import session_scope
from src.database.models import Alert, AlertStatus
from src.features.pipeline import FEATURE_COLUMNS, FeatureExtractor
from src.models.explainer import Explanation, RiskExplainer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoreResult:
    transaction_id: str | None
    risk_score: float
    is_alert: bool
    explanation: Explanation | None
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "risk_score": self.risk_score,
            "is_alert": self.is_alert,
            "latency_ms": self.latency_ms,
            "explanation": self.explanation.to_dict() if self.explanation else None,
        }


class RiskScorer:
    def __init__(
        self,
        booster: lgb.Booster,
        feature_names: list[str],
        *,
        alert_threshold: float,
        top_k: int = 3,
    ) -> None:
        if list(feature_names) != list(FEATURE_COLUMNS):
            raise ValueError("Model özellik listesi mevcut FEATURE_COLUMNS ile uyuşmuyor (şema kayması)")
        if booster.num_feature() != len(feature_names):
            raise ValueError("Booster özellik sayısı uyuşmuyor")
        self.booster = booster
        self.feature_names = list(feature_names)
        self.alert_threshold = alert_threshold
        self.top_k = top_k
        self.explainer = RiskExplainer(booster, self.feature_names)

    @classmethod
    def from_artifacts(cls, artifacts_dir: Path | None = None, settings: Settings | None = None) -> RiskScorer:
        settings = settings or get_settings()
        artifacts_dir = artifacts_dir or settings.model.artifacts_dir
        meta = json.loads((artifacts_dir / "feature_columns.json").read_text(encoding="utf-8"))
        booster = lgb.Booster(model_file=str(artifacts_dir / "model.txt"))
        return cls(booster, meta["feature_columns"], alert_threshold=settings.FRAUD_THRESHOLD,
                   top_k=settings.model.explain_top_k)

    # ------------------------------------------------------------------ skor
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(X[self.feature_names])

    def score_features(
        self, features: Mapping[str, float], *, transaction_id: str | None = None, explain: bool = True
    ) -> ScoreResult:
        t0 = time.perf_counter()
        X = FeatureExtractor.to_frame(features)
        score = float(self.predict(X)[0])
        is_alert = score >= self.alert_threshold
        explanation = (
            self.explainer.explain(X, [score], [transaction_id], self.top_k)[0] if explain else None
        )
        return ScoreResult(transaction_id, score, is_alert, explanation, (time.perf_counter() - t0) * 1e3)

    def score_transaction(
        self, extractor: FeatureExtractor, tx: Mapping[str, Any], *, update_state: bool = True
    ) -> ScoreResult:
        """Canlı tek işlem: özellik çıkarımı + skor + SHAP (gecikme tüm adımları kapsar)."""
        t0 = time.perf_counter()
        feats = extractor.transform_one(tx, update_state=update_state)
        result = self.score_features(feats, transaction_id=tx.get("transaction_id"))
        return ScoreResult(result.transaction_id, result.risk_score, result.is_alert, result.explanation,
                           (time.perf_counter() - t0) * 1e3)


def write_alerts(
    scorer: RiskScorer,
    X: pd.DataFrame,
    engine: Engine | None = None,
    *,
    threshold: float | None = None,
) -> int:
    """`risk_score >= eşik` olan işlemleri SHAP gerekçeleriyle `alerts` tablosuna yazar (idempotent)."""
    threshold = scorer.alert_threshold if threshold is None else threshold
    scores = scorer.predict(X)
    mask = scores >= threshold
    if not mask.any():
        return 0
    flagged = X.loc[mask]
    ids = flagged.index.astype(str).tolist()
    explanations = scorer.explainer.explain(flagged, scores[mask], ids, scorer.top_k)

    with session_scope(engine) as session:
        for start in range(0, len(ids), 500):
            session.execute(delete(Alert).where(Alert.transaction_id.in_(ids[start : start + 500])))
        session.add_all(
            Alert(
                transaction_id=exp.transaction_id,
                risk_score=round(exp.risk_score, 6),
                ml_score=round(exp.risk_score, 6),
                graph_score=round(exp.graph_contribution_share, 4),
                status=AlertStatus.OPEN,
                reasons=[f.__dict__ for f in exp.top_factors],
            )
            for exp in explanations
        )
    logger.info("%d alert yazıldı (eşik %.2f)", len(explanations), threshold)
    return len(explanations)
