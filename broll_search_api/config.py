from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PKG_DIR = Path(__file__).resolve().parent
ROOT_DIR = PKG_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_model: str = "groq/llama-3.3-70b-versatile"
    llm_api_key: str = ""
    llm_base_url: str = ""

    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    storage_dir: str = str(PKG_DIR / "storage")
    library_dir: str = str(PKG_DIR / "assets")

    playwright_headless: bool = True
    search_timeout_ms: int = 25000
    page_settle_ms: int = 1800
    max_asset_bytes: int = 80_000_000
    broll_clip_seconds: float = 4.0
    output_width: int = 1080
    output_height: int = 1920
    user_agent: str = (
        "KalkiBrollSearch/0.1 (local reusable-media research; +https://localhost)"
    )

    @property
    def storage_path(self) -> Path:
        path = Path(self.storage_dir)
        if not path.is_absolute():
            path = ROOT_DIR / path
        return path

    @property
    def library_path(self) -> Path:
        path = Path(self.library_dir)
        if not path.is_absolute():
            path = ROOT_DIR / path
        return path


settings = Settings()
