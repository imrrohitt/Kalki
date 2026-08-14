from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_model: str = "groq/llama-3.3-70b-versatile"
    llm_api_key: str = ""
    llm_base_url: str = ""

    whisper_model: str = "tiny"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_vad_enabled: bool = True
    whisper_min_silence_ms: int = 500
    whisper_beam_size: int = 5

    caption_font_path: str = "assets/fonts/Montserrat-Bold.ttf"
    # Max source length in seconds. 0 disables the duration check.
    max_video_duration_sec: float = 0.0
    caption_chunk_seconds: float = 0.0
    output_width: int = 1080
    output_height: int = 1920
    output_fps: int = 30
    storage_dir: str = "storage"
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"

    @property
    def font_path(self) -> Path:
        path = Path(self.caption_font_path)
        if not path.is_absolute():
            path = ROOT_DIR / path
        return path

    @property
    def storage_path(self) -> Path:
        path = Path(self.storage_dir)
        if not path.is_absolute():
            path = ROOT_DIR / path
        return path


settings = Settings()
