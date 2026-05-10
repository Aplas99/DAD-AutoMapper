from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_SLICE_COUNT = 24


@dataclass(slots=True)
class TempoSection:
    tempo: float
    start_time: float
    confidence: float = 0.0


@dataclass(slots=True)
class AnalysisResult:
    audio_path: Path
    sample_rate: int
    duration: float
    base_tempo: float
    beat_offset: float
    beat_times: list[float]
    tempo_sections: list[TempoSection]
    waveform_times: list[float]
    waveform_values: list[float]
    raw_waveform_values: list[float]
    start_song_offset: float = 0.0
    end_song_offset: float = 0.0


@dataclass(slots=True)
class SongProject:
    song_name: str
    audio_path: Path | None = None
    base_tempo: float = 120.0
    beat_offset: float = 0.0
    start_song_offset: float = 0.0
    end_song_offset: float = 0.0
    tempo_sections: list[TempoSection] = field(default_factory=list)
    beat_times: list[float] = field(default_factory=list)
    waveform_times: list[float] = field(default_factory=list)
    waveform_values: list[float] = field(default_factory=list)
    raw_waveform_values: list[float] = field(default_factory=list)
    duration: float = 0.0
    sample_rate: int = 0


def sanitize_song_name(name: str) -> str:
    clean = "".join(ch for ch in name if ch not in '<>:"/\\|?*').strip()
    return clean or "Imported Song"
