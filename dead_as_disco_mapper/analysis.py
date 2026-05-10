from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import median_filter

from .models import AnalysisResult, TempoSection


@dataclass(slots=True)
class DetectionConfig:
    sample_rate: int = 22050
    waveform_points: int = 6000
    section_min_length: float = 8.0
    section_change_threshold: float = 0.14
    confidence_window: int = 16


COMMON_TEMPI = np.array(
    [
        60.0,
        70.0,
        75.0,
        80.0,
        85.0,
        90.0,
        95.0,
        100.0,
        105.0,
        110.0,
        115.0,
        120.0,
        122.0,
        124.0,
        126.0,
        128.0,
        130.0,
        135.0,
        140.0,
        145.0,
        150.0,
        160.0,
        170.0,
        180.0,
        183.0,
        190.0,
        200.0,
    ]
)


def analyze_audio(audio_path: str | Path, config: DetectionConfig | None = None) -> AnalysisResult:
    cfg = config or DetectionConfig()
    audio_file = Path(audio_path)
    samples, sr = librosa.load(audio_file, sr=cfg.sample_rate, mono=True)
    duration = float(librosa.get_duration(y=samples, sr=sr))

    onset_env = librosa.onset.onset_strength(y=samples, sr=sr, aggregate=np.median)
    rms = librosa.feature.rms(y=samples)[0]
    tempo, beat_frames = librosa.beat.beat_track(y=samples, sr=sr, onset_envelope=onset_env, trim=False)
    tempo_value = _coerce_tempo(tempo)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    beat_offset = _estimate_beat_offset(beat_times, tempo_value)

    local_tempo = _estimate_local_tempo(onset_env, sr)
    smoothed_tempo = median_filter(local_tempo, size=9)
    base_tempo = _refine_base_tempo(_snap_tempo(tempo_value))
    sections = _detect_sections(
        smoothed_tempo,
        onset_env,
        rms,
        samples,
        sr,
        beat_times,
        base_tempo,
        cfg.section_min_length,
        cfg.section_change_threshold,
        cfg.confidence_window,
    )

    waveform_times, waveform_values, raw_waveform_values = _build_waveform(samples, sr, cfg.waveform_points)
    return AnalysisResult(
        audio_path=audio_file,
        sample_rate=sr,
        duration=duration,
        base_tempo=base_tempo,
        beat_offset=beat_offset,
        beat_times=beat_times,
        tempo_sections=sections,
        waveform_times=waveform_times,
        waveform_values=waveform_values,
        raw_waveform_values=raw_waveform_values,
    )


def recommended_tempo_mapping(base_tempo: float, sections: list[TempoSection]) -> tuple[float, list[TempoSection]]:
    multiplier = _recommended_tempo_multiplier(base_tempo)
    if multiplier == 1.0:
        return round(base_tempo, 2), list(sections)

    adjusted_sections = [
        TempoSection(
            tempo=_snap_tempo(section.tempo * multiplier),
            start_time=section.start_time,
            confidence=section.confidence,
        )
        for section in sections
    ]
    return _snap_tempo(base_tempo * multiplier), adjusted_sections


def sexy_tempo_mapping(base_tempo: float, sections: list[TempoSection]) -> tuple[float, list[TempoSection]]:
    new_base = _snap_tempo(base_tempo * 2.0) if base_tempo < 120.0 else base_tempo
    new_sections = [
        TempoSection(
            tempo=_snap_tempo(s.tempo * 2.0) if s.tempo < 120.0 else s.tempo,
            start_time=s.start_time,
            confidence=s.confidence,
        )
        for s in sections
    ]
    return new_base, new_sections


def recommended_section_tempo_mapping(sections: list[TempoSection], threshold: float = 120.0) -> list[TempoSection]:
    adjusted_sections: list[TempoSection] = []
    for section in sections:
        multiplier = _recommended_tempo_multiplier(section.tempo, threshold=threshold)
        adjusted_sections.append(
            TempoSection(
                tempo=_snap_tempo(section.tempo * multiplier),
                start_time=section.start_time,
                confidence=section.confidence,
            )
        )
    return adjusted_sections


def _estimate_local_tempo(onset_env: np.ndarray, sr: int) -> np.ndarray:
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
    bpms = librosa.tempo_frequencies(tempogram.shape[0], sr=sr)
    valid_rows = np.where(np.isfinite(bpms) & (bpms > 1.0) & (bpms < 300.0))[0]
    if len(valid_rows) == 0:
        return np.zeros(tempogram.shape[1], dtype=float)
    valid_tempogram = tempogram[valid_rows]
    tempo_idx = np.argmax(valid_tempogram, axis=0)
    valid_bpms = bpms[valid_rows]
    local_tempo = valid_bpms[tempo_idx]
    local_tempo = np.nan_to_num(local_tempo, nan=0.0, posinf=0.0, neginf=0.0)
    local_tempo = np.where(np.isfinite(local_tempo), local_tempo, 0.0)
    return np.where((local_tempo > 1.0) & (local_tempo < 300.0), local_tempo, 0.0)


def _recommended_tempo_multiplier(base_tempo: float, threshold: float = 120.0) -> float:
    tempo = max(float(base_tempo), 1.0)
    multiplier = 1.0
    while tempo < threshold and tempo * 2.0 <= 220.0:
        tempo *= 2.0
        multiplier *= 2.0
    return multiplier


def _estimate_beat_offset(beat_times: list[float], tempo: float) -> float:
    if len(beat_times) < 2 or tempo <= 0:
        return 0.0
    beat_length = 60.0 / float(tempo)
    reference = beat_times[0]
    return round(reference % beat_length, 4)


def _detect_sections(
    local_tempo: np.ndarray,
    onset_env: np.ndarray,
    rms: np.ndarray,
    samples: np.ndarray,
    sr: int,
    beat_times: list[float],
    base_tempo: float,
    min_length: float,
    change_threshold: float,
    confidence_window: int,
) -> list[TempoSection]:
    snapped = np.array([_snap_relative_tempo(value, base_tempo) for value in local_tempo], dtype=float)
    snapped = median_filter(snapped, size=11)

    segment_edges = _structural_boundaries(samples, sr, min_length)
    if len(segment_edges) < 2:
        return []

    alt_tempo = _snap_tempo(base_tempo * 1.5)
    energy_scores = []
    classified_segments: list[tuple[float, float, float, float]] = []

    for start_time, end_time in zip(segment_edges[:-1], segment_edges[1:]):
        if end_time - start_time < min_length * 0.55:
            continue
        start_frame = int(librosa.time_to_frames(start_time, sr=sr))
        end_frame = max(start_frame + 1, int(librosa.time_to_frames(end_time, sr=sr)))
        segment_rms = float(np.mean(rms[start_frame:end_frame])) if end_frame > start_frame else 0.0
        segment_onset = float(np.mean(onset_env[start_frame:end_frame])) if end_frame > start_frame else 0.0
        alt_score = _tempo_presence_score(snapped[start_frame:end_frame], alt_tempo)
        base_score = _tempo_presence_score(snapped[start_frame:end_frame], base_tempo)
        energy_scores.append(segment_rms)
        classified_segments.append((start_time, end_time, segment_rms + segment_onset * 0.08, alt_score - base_score))

    if not classified_segments:
        return []

    energy_threshold = float(np.percentile(np.array(energy_scores), 80))
    sections: list[TempoSection] = []
    previous_tempo = base_tempo

    for start_time, end_time, energy_score, pulse_bias in classified_segments:
        segment_length = end_time - start_time
        tempo = base_tempo
        if (
            segment_length >= max(min_length * 1.8, 18.0)
            and energy_score >= energy_threshold
            and start_time >= min_length * 2.0
        ):
            tempo = alt_tempo
        if abs(tempo - previous_tempo) / max(previous_tempo, 1.0) >= change_threshold:
            confidence = round(
                min(
                    0.98,
                    max(
                        0.25,
                        0.55
                        + min(0.25, max(0.0, energy_score - energy_threshold))
                        + min(0.18, max(0.0, pulse_bias + 0.18)),
                    ),
                ),
                3,
            )
            sections.append(
                TempoSection(
                    tempo=tempo,
                    start_time=_snap_time_to_beats(start_time, beat_times),
                    confidence=confidence,
                )
            )
            previous_tempo = tempo

    return _collapse_sections(sections, base_tempo, min_length)


def _collapse_sections(sections: list[TempoSection], base_tempo: float, min_length: float) -> list[TempoSection]:
    collapsed: list[TempoSection] = []
    for section in sorted(sections, key=lambda item: item.start_time):
        if section.start_time < 0.05:
            continue
        if collapsed and abs(collapsed[-1].tempo - section.tempo) < 0.5:
            continue
        if collapsed and section.start_time - collapsed[-1].start_time < min_length:
            if section.confidence > collapsed[-1].confidence:
                collapsed[-1] = section
            continue
        collapsed.append(section)
    if collapsed and abs(collapsed[0].tempo - base_tempo) < 0.5:
        collapsed.pop(0)
    return collapsed


def _snap_tempo(value: float) -> float:
    if value <= 0:
        return 120.0
    return round(float(COMMON_TEMPI[np.argmin(np.abs(COMMON_TEMPI - value))]), 2)


def _refine_base_tempo(value: float) -> float:
    nearby = [tempo for tempo in COMMON_TEMPI if abs(float(tempo) - value) <= 4.0]
    if not nearby:
        return value
    best = min(
        nearby,
        key=lambda tempo: abs(float(tempo) - value) + 0.75 * abs(_snap_tempo(float(tempo) * 1.5) - float(tempo) * 1.5),
    )
    return round(float(best), 2)


def _snap_relative_tempo(value: float, base_tempo: float) -> float:
    if value <= 0:
        return 0.0
    candidates = np.array(
        [
            base_tempo * 0.5,
            base_tempo * (2.0 / 3.0),
            base_tempo,
            base_tempo * 1.5,
            base_tempo * 2.0,
        ]
    )
    nearest = float(candidates[np.argmin(np.abs(candidates - value))])
    if abs(nearest - value) / max(nearest, 1.0) < 0.16:
        return _snap_tempo(nearest)
    return _snap_tempo(value)


def _snap_time_to_beats(value: float, beat_times: list[float]) -> float:
    if not beat_times:
        return round(value, 4)
    beat_array = np.array(beat_times)
    return round(float(beat_array[np.argmin(np.abs(beat_array - value))]), 4)


def _structural_boundaries(samples: np.ndarray, sr: int, min_length: float) -> list[float]:
    duration = float(librosa.get_duration(y=samples, sr=sr))
    target_segments = int(np.clip(round(duration / 24.0), 6, 10))
    mfcc = librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=13)
    boundaries = librosa.segment.agglomerative(mfcc, target_segments)
    times = librosa.frames_to_time(boundaries, sr=sr).tolist()
    cleaned = [0.0]
    for time_value in times[1:]:
        if time_value - cleaned[-1] >= min_length * 0.65:
            cleaned.append(float(time_value))
    if duration - cleaned[-1] >= min_length * 0.35:
        cleaned.append(duration)
    elif cleaned[-1] != duration:
        cleaned[-1] = duration
    return cleaned


def _tempo_presence_score(tempo_slice: np.ndarray, target_tempo: float) -> float:
    if len(tempo_slice) == 0:
        return 0.0
    return float(np.mean(np.exp(-np.abs(tempo_slice - target_tempo) / 6.0)))


def _build_waveform(samples: np.ndarray, sr: int, target_points: int) -> tuple[list[float], list[float], list[float]]:
    if len(samples) == 0:
        return [], [], []
    window = max(1, len(samples) // target_points)
    trimmed = samples[: len(samples) - (len(samples) % window)]
    chunks = trimmed.reshape(-1, window) if len(trimmed) else samples.reshape(1, -1)
    raw = np.mean(chunks, axis=1)
    peaks = np.max(np.abs(chunks), axis=1)
    times = np.linspace(0.0, len(samples) / sr, num=len(peaks), endpoint=False)
    return times.tolist(), peaks.tolist(), raw.tolist()


def _coerce_tempo(value: float | np.ndarray) -> float:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 120.0
        return float(value.reshape(-1)[0])
    return float(value)
