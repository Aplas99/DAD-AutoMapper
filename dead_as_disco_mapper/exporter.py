from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

from .models import DEFAULT_SLICE_COUNT, SongProject, sanitize_song_name


def build_meta(project: SongProject) -> dict:
    return {
        "version": 1,
        "uniqueId": random.randint(100000000, 2147483647),
        "songName": project.song_name,
        "tempo": round(project.base_tempo, 3),
        "customTempoSections": [
            {
                "tempo": round(section.tempo, 3),
                "startTimestamp": {
                    "barNumber": 99,
                    "beatNumber": 1,
                    "sliceNumber": 0,
                    "sliceCount": DEFAULT_SLICE_COUNT,
                    "msOffset": 0,
                },
                "startAbsoluteTime": round(section.start_time, 6),
            }
            for section in sorted(project.tempo_sections, key=lambda item: item.start_time)
        ],
        "beatOffset": round(project.beat_offset, 4),
        "startSongOffset": round(project.start_song_offset, 4),
        "endSongOffset": round(project.end_song_offset, 4),
    }


def export_project(project: SongProject, export_root: str | Path) -> Path:
    if not project.audio_path:
        raise ValueError("No audio file loaded.")

    root = Path(export_root)
    song_dir = root / sanitize_song_name(project.song_name)
    song_dir.mkdir(parents=True, exist_ok=True)

    meta_path = song_dir / "meta.json"
    audio_path = song_dir / "audio.ogg"

    meta_path.write_text(json.dumps(build_meta(project), indent=4), encoding="utf-8")
    convert_audio_to_ogg(project.audio_path, audio_path)
    return song_dir


def convert_audio_to_ogg(source: str | Path, destination: str | Path) -> None:
    source_path = Path(source)
    destination_path = Path(destination)
    if source_path.suffix.lower() == ".ogg":
        destination_path.write_bytes(source_path.read_bytes())
        return

    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg was not found. Install ffmpeg and make sure it is on your PATH, "
            "then restart the application."
        )
    _convert_with_ffmpeg(ffmpeg, source_path, destination_path)


def _find_ffmpeg() -> str | None:
    # When running as a PyInstaller bundle, _MEIPASS points to the _internal folder
    # where --add-binary places files in one-directory mode.
    if getattr(sys, "frozen", False):
        bundled = Path(getattr(sys, "_MEIPASS", "")) / "ffmpeg.exe"
        if bundled.exists():
            return str(bundled)
    return shutil.which("ffmpeg")


def _convert_with_ffmpeg(ffmpeg: str, source_path: Path, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source_path),
        "-vn",
        "-map_metadata",
        "-1",
        "-acodec",
        "libvorbis",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-q:a",
        "5",
        str(destination_path),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        error_text = completed.stderr.strip() or completed.stdout.strip() or "Unknown ffmpeg error"
        raise RuntimeError(f"ffmpeg conversion failed: {error_text}")
