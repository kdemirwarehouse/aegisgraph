"""AegisGraph Hugging Face Spaces (Gradio SDK) uygulama giriş noktası.

Sunucu ayağa kalkmadan önce veritabanı ve eğitilmiş model artefaktlarının
hazır olduğundan emin olur; eksikse sentetik veriyi üretir, modeli eğitir
ve ardından FastAPI + SOC Dashboard uygulamasını Uvicorn üzerinden başlatır.
"""

from __future__ import annotations

import logging
import os

import uvicorn

from config.settings import Settings, get_settings

logger = logging.getLogger("aegisgraph.app")


def ensure_runtime_artifacts(settings: Settings | None = None) -> None:
    """Veritabanı veya eğitilmiş model eksikse başlangıç öncesi üretir."""
    resolved = settings or get_settings()
    db_path = resolved.database.path
    artifacts_dir = resolved.model.artifacts_dir
    model_path = artifacts_dir / "model.txt"
    extractor_path = artifacts_dir / "feature_extractor.joblib"

    db_missing = not db_path.is_file()
    if db_missing:
        logger.info("Veritabanı bulunamadı (%s); sentetik veri üretiliyor...", db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        from src.generator.synthetic_data import main as generate_main

        if generate_main([]) != 0:
            raise RuntimeError("Sentetik veri üretimi başarısız oldu")

    if db_missing or not model_path.is_file() or not extractor_path.is_file():
        logger.info("Model artefaktları eksik (%s); model eğitimi başlatılıyor...", artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        from src.models.train import main as train_main

        if train_main([]) != 0:
            raise RuntimeError("Model eğitimi başarısız oldu")

    logger.info("Çalışma zamanı veritabanı ve model artefaktları hazır.")


def main() -> int:
    """Artefaktları doğrular ve Uvicorn sunucusunu başlatır."""
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    ensure_runtime_artifacts(settings)

    port = int(os.getenv("PORT", "7860"))
    logger.info("AegisGraph sunucusu başlatılıyor: 0.0.0.0:%d", port)
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
