from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from app.captions.models import CaptionTimeline
from app.config import settings
from app.editorial.models import GraphicBeat, SfxHit, ZoomDecision
from app.media.probe import MediaError, probe_video
from app.media.subject import detect_head_top
from app.renderer.ass import write_ass_file
from app.renderer.canvas import build_full_canvas_filtergraph
from app.renderer.sfx_mix import build_sfx_mix
from app.renderer.split import build_split_filtergraph

logger = logging.getLogger(__name__)


class FFmpegRenderer:
    def __init__(
        self,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
        font_path: str | None = None,
        split_layout: bool | None = None,
    ) -> None:
        self.width = width or settings.output_width
        self.height = height or settings.output_height
        self.fps = fps or settings.output_fps
        self.font_path = font_path or str(settings.font_path)
        self.split_layout = (
            settings.split_layout_enabled if split_layout is None else split_layout
        )

    def _prepare_fonts(self) -> None:
        font = Path(self.font_path)
        if not font.exists():
            raise MediaError(f"Caption font not found: {self.font_path}")
        emoji = Path("/System/Library/Fonts/Apple Color Emoji.ttc")
        if emoji.exists():
            link = font.parent / "Apple Color Emoji.ttc"
            if not link.exists():
                try:
                    link.symlink_to(emoji)
                except OSError:
                    pass

    def _escape_filter_path(self, path: Path) -> str:
        return (
            str(path.resolve())
            .replace("\\", "/")
            .replace(":", "\\:")
            .replace("'", "\\'")
            .replace(",", "\\,")
            .replace("[", "\\[")
            .replace("]", "\\]")
            .replace(" ", "\\ ")
        )

    def _encode_args(self) -> list[str]:
        return [
            "-c:v",
            "libx264",
            "-preset",
            settings.x264_preset,
            "-crf",
            str(settings.x264_crf),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            settings.audio_bitrate,
            "-movflags",
            "+faststart",
            "-shortest",
        ]

    def render(
        self,
        source_video: str,
        caption_timeline: CaptionTimeline,
        output_path: str,
        zooms: list[ZoomDecision] | None = None,
        graphics: list[GraphicBeat] | None = None,
        sfx: list[SfxHit] | None = None,
        split_layout: bool | None = None,
        video_duration: float = 0.0,
        theme: str | None = None,
    ) -> str:
        theme = theme or settings.graphics_theme
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self._prepare_fonts()
        font = Path(self.font_path)

        use_split = self.split_layout if split_layout is None else split_layout
        info = probe_video(source_video)
        if use_split:
            out_w, out_h = self.width, self.height
        else:
            out_w, out_h = info.display_width, info.display_height
        out = Path(output_path)
        ass_path = out.with_suffix(".ass")
        head_top = None
        if not use_split:
            head_top = detect_head_top(
                source_video, width=out_w, height=out_h
            )
        write_ass_file(
            caption_timeline,
            str(ass_path),
            width=out_w,
            height=out_h,
            font_name="Montserrat",
            graphics=graphics if use_split else [],
            split_layout=use_split,
            video_duration=video_duration,
            theme=theme,
            head_top=head_top,
        )

        _ = zooms
        ass_escaped = self._escape_filter_path(ass_path)
        fonts_dir = self._escape_filter_path(font.parent)

        cmd = [settings.ffmpeg_path, "-y", "-i", source_video]

        sfx_files, audio_filter = ([], "")
        if settings.sfx_enabled and sfx:
            sfx_files, audio_filter = build_sfx_mix(
                sfx,
                voice_has_audio=info.has_audio,
                video_duration=video_duration or info.duration,
            )
            for path in sfx_files:
                cmd.extend(["-i", str(path)])

        if use_split:
            graph = build_split_filtergraph(
                width=out_w,
                height=out_h,
                fps=self.fps,
                ass_escaped=ass_escaped,
                fonts_dir=fonts_dir,
                theme=theme,
            )
            if audio_filter:
                cmd.extend(
                    [
                        "-filter_complex",
                        f"{graph};{audio_filter}",
                        "-map",
                        "[vout]",
                        "-map",
                        "[aout]",
                    ]
                )
            else:
                cmd.extend(["-filter_complex", graph, "-map", "[vout]", "-map", "0:a?"])
        else:
            finish = (
                f"setsar=1,format=yuv420p,"
                f"ass={ass_escaped}:fontsdir={fonts_dir}"
            )
            vf = finish
            video_graph = f"[0:v]{vf}[vout]" if audio_filter else ""
            if audio_filter:
                video_graph = f"[0:v]{vf}[vout]"
                cmd.extend(
                    [
                        "-filter_complex",
                        f"{video_graph};{audio_filter}",
                        "-map",
                        "[vout]",
                        "-map",
                        "[aout]",
                    ]
                )
            else:
                cmd.extend(["-vf", vf])

        cmd.extend(self._encode_args())
        cmd.append(output_path)

        try:
            logger.info(
                "ffmpeg %s %sx%s sfx=%s captions=%s crf=%s preset=%s",
                "split" if use_split else "source-frame",
                out_w,
                out_h,
                len(sfx or []),
                len(caption_timeline.captions),
                settings.x264_crf,
                settings.x264_preset,
            )
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except FileNotFoundError as exc:
            raise MediaError("ffmpeg not found. Install FFmpeg.") from exc
        except subprocess.CalledProcessError as exc:
            raise MediaError(f"render failed: {exc.stderr[-1000:]}") from exc
        return output_path

    def render_audio_reel(
        self,
        source_audio: str,
        caption_timeline: CaptionTimeline,
        output_path: str,
        graphics: list[GraphicBeat] | None = None,
        sfx: list[SfxHit] | None = None,
        audio_duration: float = 0.0,
        theme: str | None = None,
    ) -> str:
        theme = theme or settings.graphics_theme
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self._prepare_fonts()
        font = Path(self.font_path)
        out = Path(output_path)
        ass_path = out.with_suffix(".ass")
        write_ass_file(
            caption_timeline,
            str(ass_path),
            width=self.width,
            height=self.height,
            font_name="Montserrat",
            graphics=graphics,
            layout="full",
            video_duration=audio_duration,
            theme=theme,
        )
        ass_escaped = self._escape_filter_path(ass_path)
        fonts_dir = self._escape_filter_path(font.parent)
        video_graph = build_full_canvas_filtergraph(
            width=self.width,
            height=self.height,
            fps=self.fps,
            ass_escaped=ass_escaped,
            fonts_dir=fonts_dir,
            theme=theme,
        )

        cmd = [settings.ffmpeg_path, "-y", "-i", source_audio]
        sfx_files, audio_filter = ([], "")
        if settings.sfx_enabled and sfx:
            sfx_files, audio_filter = build_sfx_mix(
                sfx,
                voice_has_audio=True,
                video_duration=audio_duration,
            )
            for path in sfx_files:
                cmd.extend(["-i", str(path)])

        if audio_filter:
            cmd.extend(
                [
                    "-filter_complex",
                    f"{video_graph};{audio_filter}",
                    "-map",
                    "[vout]",
                    "-map",
                    "[aout]",
                ]
            )
        else:
            cmd.extend(
                ["-filter_complex", video_graph, "-map", "[vout]", "-map", "0:a"]
            )

        if audio_duration > 0:
            cmd.extend(["-t", f"{audio_duration:.3f}"])
        cmd.extend(self._encode_args())
        cmd.append(output_path)
        try:
            logger.info(
                "ffmpeg audio-reel graphics=%s sfx=%s captions=%s duration=%.1fs crf=%s preset=%s",
                len(graphics or []),
                len(sfx or []),
                len(caption_timeline.captions),
                audio_duration,
                settings.x264_crf,
                settings.x264_preset,
            )
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except FileNotFoundError as exc:
            raise MediaError("ffmpeg not found. Install FFmpeg.") from exc
        except subprocess.CalledProcessError as exc:
            raise MediaError(f"render failed: {exc.stderr[-1000:]}") from exc
        return output_path
