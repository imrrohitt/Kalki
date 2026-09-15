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
    # Plain-text DeepSeek key. When the file exists it wins over LLM_API_KEY.
    deepseek_key_file: str = "deepseek_key.txt"
    deepseek_base_url: str = "https://api.deepseek.com"
    # Caption director: writes + reviews on-screen copy. Pro reads Hinglish better.
    director_model: str = "deepseek-v4-pro"
    director_fast_model: str = "deepseek-flash"

    whisper_model: str = "tiny"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_vad_enabled: bool = True
    whisper_min_silence_ms: int = 500
    whisper_beam_size: int = 5
    # Decode target. "en" turns Hindi/Hinglish speech into timed English words.
    # Empty = transcribe in the detected spoken language.
    whisper_output_language: str = "en"
    whisper_initial_prompt: str = ""

    caption_font_path: str = "assets/fonts/Montserrat-Bold.ttf"
    # Max source length in seconds. 0 disables the duration check.
    max_video_duration_sec: float = 0.0
    # 0 = one LLM call for the whole video (often collapses the tail into one caption).
    caption_chunk_seconds: float = 18.0
    output_width: int = 1080
    output_height: int = 1920
    output_fps: int = 30
    split_layout_enabled: bool = True
    # Visual theme for split-layout motion graphics: paper|noir|tech|ivory.
    graphics_theme: str = "paper"
    caption_heuristic_only: bool = False
    editorial_llm_enabled: bool = True
    graphics_llm_enabled: bool = True
    sfx_llm_enabled: bool = True
    transcript_repair_llm_enabled: bool = True
    scenes_llm_enabled: bool = True
    sfx_enabled: bool = True
    sfx_dir: str = "Sound Effects V4"
    zoom_max_scale: float = 1.35
    zoom_score_threshold: float = 0.42
    zoom_min_gap_sec: float = 1.6
    zoom_min_ease_in_sec: float = 0.45
    zoom_min_ease_out_sec: float = 0.38
    zoom_max_duration_sec: float = 4.8
    storage_dir: str = "storage"
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    # Encode: CRF 16 + medium + 256k audio. Slow looks hung on long clips.
    x264_preset: str = "medium"
    x264_crf: int = 16
    audio_bitrate: str = "256k"
    # Full-frame talking head: rasterize captions at delivery width at least.
    overlay_min_width: int = 1080
    caption_director_enabled: bool = True
    music_enabled: bool = True
    # Drop licensed tracks here (mood words in the filename help the picker).
    music_dir: str = "assets/music"
    # Bed gain before ducking. -32 dB sits the synthesized bed ~18 dB under a normal voice.
    music_gain_db: float = -32.0

    @property
    def deepseek_api_key(self) -> str:
        path = Path(self.deepseek_key_file)
        if not path.is_absolute():
            path = ROOT_DIR / path
        if path.is_file():
            key = path.read_text(encoding="utf-8").strip()
            if key:
                return key
        return self.llm_api_key.replace("Bearer ", "").strip()

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else ROOT_DIR / path

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
