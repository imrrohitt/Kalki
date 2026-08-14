from fastapi import FastAPI

from app.api.routes import router
from app.config import settings


def create_app() -> FastAPI:
    settings.storage_path.mkdir(parents=True, exist_ok=True)
    (settings.storage_path / "uploads").mkdir(parents=True, exist_ok=True)
    (settings.storage_path / "jobs").mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="AI Reel Editor — Captions MVP", version="0.1.0")
    app.include_router(router)
    return app


app = create_app()
