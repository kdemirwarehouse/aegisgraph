"""AegisGraph FastAPI uygulama fabrikası ve Uvicorn giriş noktası.

Kullanım:
    python -m src.api.main
    uvicorn src.api.main:app --reload
"""

from __future__ import annotations

import argparse
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import Settings, get_settings
from src.api.routes import router
from src.api.scoring_service import ScoringService

logger = logging.getLogger(__name__)


def create_app(
    service: ScoringService | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """AegisGraph FastAPI uygulamasını yapılandırır."""
    resolved_settings = settings or (service.settings if service is not None else get_settings())

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if service is not None:
            application.state.scoring_service = service
        elif getattr(application.state, "scoring_service", None) is None:
            application.state.scoring_service = ScoringService.from_artifacts(settings=resolved_settings)
        logger.info("%s API başlatıldı (ortam=%s)", resolved_settings.app_name, resolved_settings.environment)
        yield
        logger.info("%s API kapatılıyor", resolved_settings.app_name)

    application = FastAPI(
        title=f"{resolved_settings.app_name} API",
        description="Hibrit LightGBM + NetworkX gerçek zamanlı dolandırıcılık skorlama ve graf akışı servisi",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    if service is not None:
        application.state.scoring_service = service
    application.include_router(router)
    return application


app = create_app()


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="AegisGraph FastAPI sunucusunu başlatır")
    parser.add_argument("--host", default=settings.api.host)
    parser.add_argument("--port", type=int, default=settings.api.port)
    args = parser.parse_args(argv)
    uvicorn.run("src.api.main:app", host=args.host, port=args.port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
