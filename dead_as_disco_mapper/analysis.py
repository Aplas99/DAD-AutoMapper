from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from .models import AnalysisResult, TempoSection


@dataclass(slots=True)
class DetectionConfig:
    sample_rate: int = 22050
    waveform_points: int = 6000
    section_min_length: float = 8.0
    section_change_threshold: float = 0.14


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

# librosa's default hop length — used when converting between frames and seconds.
_HOP_LENGTH = 512

def _switch_threshold(candidate: float, base_tempo: float) -> float:
    """Return the minimum score advantage needed to accept a candidate tempo.

    Harmonically related tempos (octaves, triplets) score artificially well
    because their beats happen to land on the base-tempo's strong beats.
    We require a stricter margin for those candidates to reduce false positives.
    """
    ratio = candidate / max(base_tempo, 1.0)
    if abs(ratio - 0.5) < 0.06 or abs(ratio - 2.0) < 0.06:
        return 0.15   # octave — needs strong evidence
    if abs(ratio - (2.0 / 3.0)) < 0.06 or abs(ratio - 1.5) < 0.06:
        return 0.13   # triplet-related — needs moderate evidence
    return 0.10        # unrelated tempo — standard threshold


def analyze_audio(audio_path: str | Path, config: DetectionConfig | None = None) -> AnalysisResult:
    # Inline import breaks the circular dependency: btt_adapter imports COMMON_TEMPI from here.
    from .btt_adapter import BTTUnavailableError, analyze_with_btt

    cfg = config or DetectionConfig()
    audio_file = Path(audio_path)

    try:
        # Default profile gives a reliable base BPM across all tempo ranges.
        btt = analyze_with_btt(audio_file, profile="default")

        # For high-BPM songs (≥150 BPM) BTT default's beat callbacks fire at the
        # half-tempo alias, so the beat grid is wrong even though base_tempo is right.
        # fast_tempo's histogram bias (weight mean 175) fixes the beat spacing for
        # those songs.  For songs below 150 BPM the bias causes the opposite error
        # (timeline votes for 200 BPM on a 115 BPM track), so we skip it there.
        if btt.base_tempo >= 150:
            btt = analyze_with_btt(audio_file, profile="fast_tempo")

        samples, sr = librosa.load(audio_file, sr=cfg.sample_rate, mono=True)
        waveform_times, waveform_values, raw_waveform_values = _build_waveform(samples, sr, cfg.waveform_points)

        boundaries = _structural_boundaries(samples, sr, cfg.section_min_length)
        sections = _sections_from_boundaries(
            samples,
            sr,
            btt.beat_times,
            btt.base_tempo,
            boundaries,
            cfg.section_min_length,
            cfg.section_change_threshold,
        )
        return AnalysisResult(
            audio_path=audio_file,
            sample_rate=btt.sample_rate,
            duration=btt.duration,
            base_tempo=btt.base_tempo,
            beat_offset=btt.beat_offset,
            beat_times=btt.beat_times,
            tempo_sections=sections,
            waveform_times=waveform_times,
            waveform_values=waveform_values,
            raw_waveform_values=raw_waveform_values,
        )
    except BTTUnavailableError:
        pass

    # Librosa fallback when the BTT native library cannot be built or loaded.
    samples, sr = librosa.load(audio_file, sr=cfg.sample_rate, mono=True)
    duration = float(librosa.get_duration(y=samples, sr=sr))

    onset_env = librosa.onset.onset_strength(y=samples, sr=sr, aggregate=np.median)
    tempo, beat_frames = librosa.beat.beat_track(y=samples, sr=sr, onset_envelope=onset_env, trim=False)
    tempo_value = _coerce_tempo(tempo)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()
    beat_offset = _first_noticeable_beat(beat_times, onset_times, tempo_value)
    base_tempo = _refine_base_tempo(_snap_tempo(tempo_value))

    sections = _detect_sections(
        samples,
        sr,
        beat_times,
        base_tempo,
        cfg.section_min_length,
        cfg.section_change_threshold,
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
    new_base = round(base_tempo * 2.0, 3) if base_tempo < 120.0 else base_tempo
    new_sections = [
        TempoSection(
            tempo=round(s.tempo * 2.0, 3) if s.tempo < 120.0 else s.tempo,
            start_time=s.start_time,
            confidence=s.confidence,
        )
        for s in sections
    ]
    return new_base, new_sections


def _feel_change_points(
    samples: np.ndarray,
    sr: int,
    base_tempo: float,
    duration: float,
    time_offset: float,
    window: float = 8.0,
    step: float = 4.0,
    threshold: float = 0.55,
) -> list[float]:
    if duration < window * 2:
        return []
    expected = base_tempo / 60.0
    densities: list[tuple[float, float]] = []
    t = 0.0
    while t + window <= duration:
        start_s = int(t * sr)
        end_s = min(int((t + window) * sr), len(samples))
        env = librosa.onset.onset_strength(y=samples[start_s:end_s], sr=sr, aggregate=np.median)
        n = len(librosa.onset.onset_detect(onset_envelope=env, sr=sr))
        densities.append((t + window / 2.0, n / window / max(expected, 0.001)))
        t += step
    if len(densities) < 3:
        return []
    changes: list[float] = []
    for i in range(1, len(densities)):
        prev = densities[i - 1][1]
        curr = densities[i][1]
        if abs(curr - prev) > threshold:
            changes.append(round(time_offset + densities[i][0] - window / 2.0, 3))
    deduped: list[float] = []
    for t_val in sorted(set(changes)):
        if not deduped or t_val - deduped[-1] >= window:
            deduped.append(t_val)
    return deduped


def analyze_section_feel(
    audio_path: str | Path,
    base_tempo: float,
    start_time: float,
    end_time: float,
    sample_rate: int = 22050,
) -> dict[str, object]:
    """Detect rhythmic pacing and feel within a time window.

    Compares onset density and beat-grid alignment at base, double, and half
    tempo to classify the region as normal, double-time, or half-time feel,
    and returns absolute timestamps where the pacing noticeably shifts.
    """
    samples, sr = librosa.load(
        Path(audio_path), sr=sample_rate, mono=True,
        offset=start_time, duration=end_time - start_time,
    )
    if len(samples) < sr * 3:
        return {
            "feel": "unknown", "confidence": 0.0, "density_ratio": 1.0,
            "base_score": 0.0, "double_score": 0.0, "half_score": 0.0,
            "change_points": [],
        }

    onset_env = librosa.onset.onset_strength(y=samples, sr=sr, aggregate=np.median)
    duration = len(samples) / sr

    base_score = _beat_consistency_score(onset_env, sr, base_tempo)
    double_score = _beat_consistency_score(onset_env, sr, base_tempo * 2.0)
    half_score = _beat_consistency_score(onset_env, sr, max(base_tempo * 0.5, 30.0))

    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_density = len(onset_frames) / max(duration, 0.001)
    expected_density = base_tempo / 60.0
    density_ratio = onset_density / max(expected_density, 0.001)

    feel = "normal"
    confidence = 0.0
    double_advantage = double_score - base_score
    if double_advantage > 0.06 or density_ratio > 1.7:
        feel = "double_time"
        confidence = round(min(0.95, max(double_advantage * 4.0, (density_ratio - 1.0) * 0.4)), 3)
    elif half_score >= base_score - 0.04 and density_ratio < 0.65:
        feel = "half_time"
        confidence = round(min(0.95, (1.0 - density_ratio) * 1.5), 3)

    change_points = _feel_change_points(samples, sr, base_tempo, duration, start_time)

    return {
        "feel": feel,
        "confidence": confidence,
        "density_ratio": round(density_ratio, 3),
        "base_score": round(base_score, 3),
        "double_score": round(double_score, 3),
        "half_score": round(half_score, 3),
        "change_points": change_points,
    }


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


def _recommended_tempo_multiplier(base_tempo: float, threshold: float = 120.0) -> float:
    tempo = max(float(base_tempo), 1.0)
    multiplier = 1.0
    while tempo < threshold and tempo * 2.0 <= 220.0:
        tempo *= 2.0
        multiplier *= 2.0
    return multiplier


def _first_noticeable_beat(beat_times: list[float], onset_times: list[float], tempo: float) -> float:
    """Return the time of the first beat that coincides with a detected onset.

    Scans beat_times in order and returns the earliest one where an onset lands
    within a quarter-beat window.  Falls back to beat_times[0] when no onset
    match is found so the grid still starts somewhere sensible.
    """
    if not beat_times:
        return 0.0
    if not onset_times or tempo <= 0:
        return round(beat_times[0], 4)
    beat_length = 60.0 / float(tempo)
    tolerance = beat_length * 0.25
    onset_array = np.array(onset_times)
    for bt in beat_times:
        if np.any(np.abs(onset_array - bt) <= tolerance):
            return round(bt, 4)
    return round(beat_times[0], 4)


def _sections_from_boundaries(
    samples: np.ndarray,
    sr: int,
    beat_times: list[float],
    base_tempo: float,
    boundaries: list[float],
    min_length: float,
    change_threshold: float,
    min_confidence: float = 0.35,
) -> list[TempoSection]:
    """Estimate per-segment tempos from pre-computed boundary list."""
    if len(boundaries) < 2:
        return []

    sections: list[TempoSection] = []
    previous_tempo = base_tempo

    for start_time, end_time in zip(boundaries[:-1], boundaries[1:]):
        if end_time - start_time < min_length * 0.55:
            continue
        tempo, confidence = _refine_section_tempo(samples, sr, start_time, end_time, base_tempo)
        if (
            abs(tempo - previous_tempo) / max(previous_tempo, 1.0) >= change_threshold
            and confidence >= min_confidence
        ):
            sections.append(
                TempoSection(
                    tempo=tempo,
                    start_time=_snap_time_to_beats(start_time, beat_times),
                    confidence=confidence,
                )
            )
            previous_tempo = tempo

    return _collapse_sections(sections, base_tempo, min_length)


def _detect_sections(
    samples: np.ndarray,
    sr: int,
    beat_times: list[float],
    base_tempo: float,
    min_length: float,
    change_threshold: float,
) -> list[TempoSection]:
    segment_edges = _structural_boundaries(samples, sr, min_length)
    if len(segment_edges) < 2:
        return []

    sections: list[TempoSection] = []
    previous_tempo = base_tempo

    for start_time, end_time in zip(segment_edges[:-1], segment_edges[1:]):
        if end_time - start_time < min_length * 0.55:
            continue
        tempo, confidence = _refine_section_tempo(samples, sr, start_time, end_time, base_tempo)
        if abs(tempo - previous_tempo) / max(previous_tempo, 1.0) >= change_threshold:
            sections.append(
                TempoSection(
                    tempo=tempo,
                    start_time=_snap_time_to_beats(start_time, beat_times),
                    confidence=confidence,
                )
            )
            previous_tempo = tempo

    return _collapse_sections(sections, base_tempo, min_length)


def _refine_section_tempo(
    samples: np.ndarray,
    sr: int,
    start_time: float,
    end_time: float,
    base_tempo: float,
) -> tuple[float, float]:
    """Return (snapped_tempo, confidence) for one structural section.

    Runs the beat tracker on the section's audio, builds a candidate set from
    that result and harmonic multiples of the base tempo, then picks the
    candidate whose synthetic beat grid best aligns with real onsets.  The
    alignment score is used directly as confidence.
    """
    start_sample = int(start_time * sr)
    end_sample = min(int(end_time * sr), len(samples))
    section = samples[start_sample:end_sample]

    if len(section) < sr:
        return base_tempo, 0.25

    section_onset = librosa.onset.onset_strength(y=section, sr=sr, aggregate=np.median)

    raw_tempo, _ = librosa.beat.beat_track(onset_envelope=section_onset, sr=sr, trim=False)
    section_tempo = _snap_tempo(_coerce_tempo(raw_tempo))

    candidates: set[float] = {
        _snap_tempo(base_tempo * 0.5),
        _snap_tempo(base_tempo * (2.0 / 3.0)),
        base_tempo,
        _snap_tempo(base_tempo * 1.5),
        _snap_tempo(base_tempo * 2.0),
        section_tempo,
        _snap_tempo(section_tempo * 0.5),
        _snap_tempo(section_tempo * 2.0),
    }
    candidates = {c for c in candidates if 40.0 <= c <= 260.0}

    base_score = _beat_consistency_score(section_onset, sr, base_tempo)
    best_tempo = base_tempo
    best_score = base_score

    for candidate in candidates:
        if abs(candidate - base_tempo) < 0.5:
            continue
        score = _beat_consistency_score(section_onset, sr, candidate)
        # Harmonically related candidates need a stricter advantage threshold to
        # prevent octave-ambiguity false positives (e.g. 85 winning in 170 BPM song).
        if score > base_score + _switch_threshold(candidate, base_tempo) and score > best_score:
            best_score = score
            best_tempo = candidate

    confidence = round(min(0.98, max(0.25, best_score)), 3)
    return best_tempo, confidence


def _beat_consistency_score(onset_env: np.ndarray, sr: int, tempo: float) -> float:
    """Measure how well a tempo hypothesis aligns with the onset envelope.

    Generates synthetic beat positions from the given tempo, then for each
    beat checks the peak onset strength within a ±15 % tolerance window.
    The mean of those normalised peak values is the score (0–1).
    """
    if tempo <= 0 or len(onset_env) == 0:
        return 0.0
    duration = len(onset_env) * _HOP_LENGTH / sr
    period = 60.0 / tempo
    beat_times = np.arange(0.0, duration, period)
    beat_frames = np.clip(
        (beat_times * sr / _HOP_LENGTH).astype(int), 0, len(onset_env) - 1
    )
    if len(beat_frames) < 2:
        return 0.0
    tolerance = max(1, int(period * sr / _HOP_LENGTH * 0.15))
    peak_global = float(np.max(onset_env)) or 1.0
    scores = []
    for bf in beat_frames:
        lo = max(0, bf - tolerance)
        hi = min(len(onset_env), bf + tolerance + 1)
        scores.append(float(np.max(onset_env[lo:hi])) / peak_global)
    return float(np.mean(scores))


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
