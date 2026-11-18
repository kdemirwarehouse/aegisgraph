"""AegisGraph FastAPI uygulama fabrikası ve Uvicorn giriş noktası.

Kullanım:
    python -m src.api.main
    uvicorn src.api.main:app --reload
"""

from __future__ import annotations

import argparse
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config.settings import Settings, get_settings
from src.api.routes import router
from src.api.scoring_service import ScoringService

logger = logging.getLogger(__name__)
DASHBOARD_STATIC_DIR: Path = Path(__file__).resolve().parents[1] / "dashboard" / "static"


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
    if DASHBOARD_STATIC_DIR.is_dir():
        application.mount(
            "/static",
            StaticFiles(directory=str(DASHBOARD_STATIC_DIR)),
            name="dashboard-static",
        )

    @application.get("/", include_in_schema=False)
    @application.get("/dashboard", include_in_schema=False)
    def serve_dashboard() -> FileResponse:
        """SOC Dashboard ana arayüzünü servis eder."""
        index_file = DASHBOARD_STATIC_DIR / "index.html"
        if not index_file.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dashboard arayüz dosyaları bulunamadı",
            )
        return FileResponse(index_file)

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
