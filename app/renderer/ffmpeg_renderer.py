from __future__ import annotations

import subprocess
from pathlib import Path

from app.captions.models import CaptionTimeline
from app.config import settings
from app.editorial.models import ZoomDecision
from app.media.probe import MediaError
from app.renderer.ass import write_ass_file
from app.renderer.filters import vertical_scale_crop_filter
from app.renderer.zoom import build_zoom_filtergraph


class FFmpegRenderer:
    def __init__(
        self,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
        font_path: str | None = None,
    ) -> None:
        self.width = width or settings.output_width
        self.height = height or settings.output_height
        self.fps = fps or settings.output_fps
        self.font_path = font_path or str(settings.font_path)

    def render(
        self,
        source_video: str,
        caption_timeline: CaptionTimeline,
        output_path: str,
        zooms: list[ZoomDecision] | None = None,
    ) -> str:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        font = Path(self.font_path)
        if not font.exists():
            raise MediaError(f"Caption font not found: {self.font_path}")

        out = Path(output_path)
        ass_path = out.with_suffix(".ass")
        write_ass_file(
            caption_timeline,
            str(ass_path),
            width=self.width,
            height=self.height,
            font_name="Montserrat",
        )

        # Escape for filtergraph: path may contain spaces.
        ass_escaped = (
            str(ass_path.resolve())
            .replace("\\", "/")
            .replace(":", "\\:")
            .replace("'", "\\'")
            .replace(",", "\\,")
            .replace("[", "\\[")
            .replace("]", "\\]")
            .replace(" ", "\\ ")
        )
        fonts_dir = (
            str(font.parent.resolve())
            .replace("\\", "/")
            .replace(":", "\\:")
            .replace("'", "\\'")
            .replace(",", "\\,")
            .replace(" ", "\\ ")
        )

        finish = (
            f"fps={self.fps},format=yuv420p,"
            f"ass={ass_escaped}:fontsdir={fonts_dir}"
        )
        zoom_graph = build_zoom_filtergraph(zooms, self.width, self.height, self.fps)

        cmd = [
            settings.ffmpeg_path,
            "-y",
            "-i",
            source_video,
        ]
        if zoom_graph:
            cmd.extend(
                [
                    "-filter_complex",
                    f"{zoom_graph};[vzoom]{finish}[vout]",
                    "-map",
                    "[vout]",
                    "-map",
                    "0:a?",
                ]
            )
        else:
            vf = ",".join(
                [
                    vertical_scale_crop_filter(self.width, self.height),
                    finish,
                ]
            )
            cmd.extend(["-vf", vf])

        cmd.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-movflags",
                "+faststart",
                "-shortest",
                output_path,
            ]
        )

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except FileNotFoundError as exc:
            raise MediaError("ffmpeg not found. Install FFmpeg.") from exc
        except subprocess.CalledProcessError as exc:
            raise MediaError(f"render failed: {exc.stderr[-1000:]}") from exc
        return output_path
