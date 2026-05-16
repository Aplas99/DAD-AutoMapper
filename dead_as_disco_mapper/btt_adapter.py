from __future__ import annotations

import ctypes
import json
import platform
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from scipy.ndimage import median_filter
from setuptools._distutils.ccompiler import new_compiler
from setuptools._distutils.sysconfig import customize_compiler

from .analysis import COMMON_TEMPI
from .models import TempoSection


class BTTUnavailableError(RuntimeError):
    pass


@dataclass(slots=True)
class BTTAnalysisResult:
    base_tempo: float
    beat_offset: float
    beat_times: list[float]
    tempo_sections: list[TempoSection]
    onset_times: list[float]
    tempo_updates: list[dict[str, float]]
    sample_rate: int
    duration: float
    debug: dict[str, Any]


BTT_PROFILE_DEFAULTS: dict[str, dict[str, float | int]] = {
    "default": {},
    "fast_adaptation": {
        "gaussian_tempo_histogram_decay": 0.995,
        "num_tempo_candidates": 12,
        "log_gaussian_tempo_weight_mean": 122.0,
        "log_gaussian_tempo_weight_width": 85.0,
        "onset_threshold": 0.08,
        "count_in_n": 1,
    },
    "fast_tempo": {
        # Raise the ceiling so 200 BPM sits well inside the range rather than on its edge,
        # and shift the histogram weight mean toward 175 so BTT favours 200 over the 100 alias.
        "max_tempo": 250.0,
        "log_gaussian_tempo_weight_mean": 175.0,
    },
}

_BTT_TRACKING_MODE_FULL = 3
_DEFAULT_SAMPLE_RATE = 44100
_DEFAULT_CHUNK_SIZE = 64


def available_profiles() -> tuple[str, ...]:
    return tuple(BTT_PROFILE_DEFAULTS)


def analyze_with_btt(
    audio_path: str | Path,
    profile: str = "default",
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    sample_rate: int = _DEFAULT_SAMPLE_RATE,
) -> BTTAnalysisResult:
    if profile not in BTT_PROFILE_DEFAULTS:
        raise ValueError(f"Unknown BTT profile '{profile}'. Available: {', '.join(available_profiles())}")

    library = _LoadedBTTLibrary.load()
    samples, sr = librosa.load(Path(audio_path), sr=sample_rate, mono=True)
    samples = np.asarray(samples, dtype=np.float32)
    duration = float(librosa.get_duration(y=samples, sr=sr))

    onset_times: list[float] = []
    beat_times: list[float] = []
    tempo_updates: list[dict[str, float]] = []

    btt = library.lib.btt_new_default()
    if not btt:
        raise BTTUnavailableError("btt_new_default() returned NULL.")

    onset_cb = library.onset_callback_type(lambda _self, sample_time: onset_times.append(sample_time / sr))
    beat_cb = library.beat_callback_type(lambda _self, sample_time: beat_times.append(sample_time / sr))
    library.lib.btt_set_tracking_mode(btt, _BTT_TRACKING_MODE_FULL)
    library.lib.btt_set_onset_tracking_callback(btt, onset_cb, None)
    library.lib.btt_set_beat_tracking_callback(btt, beat_cb, None)
    _apply_profile(library, btt, profile)

    last_bpm = -1.0
    last_certainty = -1.0
    try:
        for start in range(0, len(samples), chunk_size):
            chunk = np.ascontiguousarray(samples[start : start + chunk_size], dtype=np.float32)
            c_buffer = chunk.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            library.lib.btt_process(btt, c_buffer, len(chunk))
            bpm = float(library.lib.btt_get_tempo_bpm(btt))
            certainty = float(library.lib.btt_get_tempo_certainty(btt))
            current_time = min((start + len(chunk)) / sr, duration)
            if bpm > 0 and (
                not tempo_updates
                or abs(bpm - last_bpm) >= 0.25
                or abs(certainty - last_certainty) >= 0.02
            ):
                tempo_updates.append(
                    {
                        "time": round(current_time, 6),
                        "bpm": round(bpm, 6),
                        "certainty": round(certainty, 6),
                    }
                )
                last_bpm = bpm
                last_certainty = certainty
        final_bpm = float(library.lib.btt_get_tempo_bpm(btt))
        final_certainty = float(library.lib.btt_get_tempo_certainty(btt))
    finally:
        library.lib.btt_destroy(btt)

    base_tempo = _choose_base_tempo(tempo_updates, final_bpm)
    beat_offset = _estimate_beat_offset(beat_times, base_tempo)
    tempo_sections = _derive_tempo_sections(tempo_updates, beat_times, base_tempo, duration)

    return BTTAnalysisResult(
        base_tempo=base_tempo,
        beat_offset=beat_offset,
        beat_times=[round(value, 6) for value in beat_times],
        tempo_sections=tempo_sections,
        onset_times=[round(value, 6) for value in onset_times],
        tempo_updates=tempo_updates,
        sample_rate=sr,
        duration=duration,
        debug={
            "profile": profile,
            "chunk_size": chunk_size,
            "final_raw_tempo_bpm": round(final_bpm, 6),
            "final_tempo_certainty": round(final_certainty, 6),
            "tempo_update_count": len(tempo_updates),
            "onset_count": len(onset_times),
            "beat_count": len(beat_times),
            "library_path": str(library.library_path),
            "build_metadata_path": str(library.build_metadata_path),
        },
    )


class _LoadedBTTLibrary:
    def __init__(self, lib: ctypes.CDLL, library_path: Path, build_metadata_path: Path) -> None:
        self.lib = lib
        self.library_path = library_path
        self.build_metadata_path = build_metadata_path
        self.onset_callback_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_ulonglong)
        self.beat_callback_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_ulonglong)
        self._configure()

    @classmethod
    def load(cls) -> "_LoadedBTTLibrary":
        import sys
        frozen = getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")

        build_dir = _build_root()
        library_path = build_dir / _shared_library_name()
        metadata_path = build_dir / "build_metadata.json"

        if not frozen:
            source_root = _vendor_root()
            if not source_root.exists():
                raise BTTUnavailableError(
                    f"Vendored Beat-and-Tempo-Tracking source was not found at {source_root}."
                )
            build_dir.mkdir(parents=True, exist_ok=True)
            if _needs_rebuild(source_root, library_path):
                _build_shared_library(source_root, build_dir, library_path, metadata_path)

        if not library_path.exists():
            raise BTTUnavailableError(f"BTT shared library not found at {library_path}.")

        try:
            lib = ctypes.CDLL(str(library_path))
            return cls(lib, library_path, metadata_path)
        except AttributeError as exc:
            raise BTTUnavailableError(
                f"BTT library at {library_path} is missing required exports. "
                f"Delete the build directory ({build_dir}) and rerun the benchmark to rebuild it."
            ) from exc
        except OSError as exc:
            raise BTTUnavailableError(f"Unable to load BTT shared library at {library_path}: {exc}") from exc

    def _configure(self) -> None:
        self.lib.btt_new_default.restype = ctypes.c_void_p
        self.lib.btt_new_default.argtypes = []

        self.lib.btt_destroy.restype = ctypes.c_void_p
        self.lib.btt_destroy.argtypes = [ctypes.c_void_p]

        self.lib.btt_process.restype = None
        self.lib.btt_process.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]

        self.lib.btt_set_tracking_mode.restype = None
        self.lib.btt_set_tracking_mode.argtypes = [ctypes.c_void_p, ctypes.c_int]

        self.lib.btt_set_onset_tracking_callback.restype = None
        self.lib.btt_set_onset_tracking_callback.argtypes = [ctypes.c_void_p, self.onset_callback_type, ctypes.c_void_p]

        self.lib.btt_set_beat_tracking_callback.restype = None
        self.lib.btt_set_beat_tracking_callback.argtypes = [ctypes.c_void_p, self.beat_callback_type, ctypes.c_void_p]

        self.lib.btt_get_tempo_bpm.restype = ctypes.c_double
        self.lib.btt_get_tempo_bpm.argtypes = [ctypes.c_void_p]

        self.lib.btt_get_tempo_certainty.restype = ctypes.c_double
        self.lib.btt_get_tempo_certainty.argtypes = [ctypes.c_void_p]

        for name, arg_type in (
            ("btt_set_count_in_n", ctypes.c_int),
            ("btt_set_num_tempo_candidates", ctypes.c_int),
            ("btt_set_gaussian_tempo_histogram_decay", ctypes.c_double),
            ("btt_set_log_gaussian_tempo_weight_mean", ctypes.c_double),
            ("btt_set_log_gaussian_tempo_weight_width", ctypes.c_double),
            ("btt_set_onset_threshold", ctypes.c_double),
            ("btt_set_max_tempo", ctypes.c_double),
            ("btt_set_min_tempo", ctypes.c_double),
        ):
            func = getattr(self.lib, name, None)
            if func is not None:
                func.restype = None
                func.argtypes = [ctypes.c_void_p, arg_type]


def _apply_profile(library: _LoadedBTTLibrary, btt_handle: int, profile: str) -> None:
    settings = BTT_PROFILE_DEFAULTS[profile]
    for key, value in settings.items():
        setter = getattr(library.lib, f"btt_set_{key}", None)
        if setter is None:
            continue
        setter(btt_handle, value)


def _choose_base_tempo(tempo_updates: list[dict[str, float]], final_bpm: float) -> float:
    weighted_scores: defaultdict[float, float] = defaultdict(float)
    for update in tempo_updates:
        bpm = float(update["bpm"])
        certainty = max(0.05, float(update["certainty"]))
        if bpm <= 0:
            continue
        weighted_scores[_snap_game_tempo(bpm)] += certainty
    if weighted_scores:
        tempo = max(weighted_scores.items(), key=lambda item: (item[1], -abs(item[0] - final_bpm)))[0]
        # BTT often tracks at half-tempo for most of a fast song before converging.
        # If the end-of-song estimate is ~2x the timeline vote, the final value is
        # the reliable one — prefer it.
        if final_bpm > 130:
            snapped_final = _snap_game_tempo(final_bpm)
            if 1.85 <= snapped_final / max(tempo, 1.0) <= 2.15:
                return round(float(snapped_final), 3)
        return round(float(tempo), 3)
    return round(float(_snap_game_tempo(final_bpm if final_bpm > 0 else 120.0)), 3)


def _derive_tempo_sections(
    tempo_updates: list[dict[str, float]],
    beat_times: list[float],
    base_tempo: float,
    duration: float,
    min_length: float = 8.0,
    change_threshold: float = 0.14,
) -> list[TempoSection]:
    filtered = [
        update
        for update in tempo_updates
        if update["bpm"] > 0 and update["certainty"] >= 0.15 and 0.0 <= update["time"] <= duration
    ]
    if not filtered:
        return []

    snapped_tempi = np.array([_snap_game_tempo(item["bpm"]) for item in filtered], dtype=float)
    if len(snapped_tempi) >= 5:
        snapped_tempi = median_filter(snapped_tempi, size=min(9, len(snapped_tempi) // 2 * 2 + 1))

    segments: list[dict[str, float]] = []
    start_index = 0
    for index in range(1, len(filtered) + 1):
        at_end = index == len(filtered)
        if at_end or abs(snapped_tempi[index] - snapped_tempi[start_index]) >= 0.5:
            segment = filtered[start_index:index]
            start_time = float(segment[0]["time"])
            end_time = float(segment[-1]["time"])
            if index < len(filtered):
                end_time = max(end_time, float(filtered[index]["time"]))
            mean_certainty = float(np.mean([item["certainty"] for item in segment]))
            segments.append(
                {
                    "tempo": float(snapped_tempi[start_index]),
                    "start_time": start_time,
                    "end_time": end_time,
                    "confidence": mean_certainty,
                }
            )
            start_index = index

    collapsed: list[dict[str, float]] = []
    for segment in segments:
        if not collapsed:
            collapsed.append(segment)
            continue
        if segment["end_time"] - segment["start_time"] < min_length:
            if segment["confidence"] > collapsed[-1]["confidence"]:
                collapsed[-1]["tempo"] = segment["tempo"]
                collapsed[-1]["confidence"] = segment["confidence"]
            collapsed[-1]["end_time"] = max(collapsed[-1]["end_time"], segment["end_time"])
            continue
        collapsed.append(segment)

    sections: list[TempoSection] = []
    previous_tempo = base_tempo
    for segment in collapsed:
        tempo = float(segment["tempo"])
        if abs(tempo - previous_tempo) / max(previous_tempo, 1.0) < change_threshold:
            continue
        sections.append(
            TempoSection(
                tempo=round(tempo, 3),
                start_time=_snap_time_to_beats(float(segment["start_time"]), beat_times),
                confidence=round(min(0.98, max(0.25, float(segment["confidence"]))), 3),
            )
        )
        previous_tempo = tempo
    return sections


def _estimate_beat_offset(beat_times: list[float], tempo: float) -> float:
    if len(beat_times) < 2 or tempo <= 0:
        return 0.0
    beat_length = 60.0 / float(tempo)
    reference = beat_times[0]
    return round(reference % beat_length, 4)


def _snap_time_to_beats(value: float, beat_times: list[float]) -> float:
    if not beat_times:
        return round(value, 4)
    beat_array = np.array(beat_times, dtype=float)
    return round(float(beat_array[np.argmin(np.abs(beat_array - value))]), 4)


def _snap_game_tempo(value: float) -> float:
    if value <= 0:
        return 120.0
    return float(COMMON_TEMPI[np.argmin(np.abs(COMMON_TEMPI - value))])


def _vendor_root() -> Path:
    return Path(__file__).resolve().parents[1] / "vendor" / "Beat-and-Tempo-Tracking"


def _build_root() -> Path:
    import sys
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[1] / "build" / "btt"


def _shared_library_name() -> str:
    system = platform.system().lower()
    if system == "windows":
        return "btt.dll"
    if system == "darwin":
        return "libbtt.dylib"
    return "libbtt.so"


def _build_shared_library(source_root: Path, build_dir: Path, library_path: Path, metadata_path: Path) -> None:
    compiler = new_compiler()
    customize_compiler(compiler)

    source_files = [
        source_root / "src" / "BTT.c",
        source_root / "src" / "DFT.c",
        source_root / "src" / "Filter.c",
        source_root / "src" / "STFT.c",
        source_root / "src" / "Statistics.c",
        source_root / "src" / "fastsin.c",
    ]
    for file_path in source_files:
        if not file_path.exists():
            raise BTTUnavailableError(f"Missing BTT source file: {file_path}")

    compile_preargs: list[str] = []
    link_postargs: list[str] = []
    libraries: list[str] = []
    if compiler.compiler_type == "msvc":
        compile_preargs = ["/O2"]
        # distutils' export_symbols path is unreliable with setuptools-bundled distutils on
        # Windows — write the .def file ourselves and pass it directly to the MSVC linker.
        def_path = build_dir / "btt.def"
        def_path.write_text(
            "EXPORTS\n" + "\n".join(f"    {sym}" for sym in _exported_symbols()),
            encoding="utf-8",
        )
        link_postargs = [f"/DEF:{def_path}"]
    else:
        compile_preargs = ["-O2", "-std=c99", "-fPIC"]
        libraries = ["m"]

    try:
        objects = compiler.compile(
            [str(path) for path in source_files],
            output_dir=str(build_dir / "obj"),
            include_dirs=[str(source_root)],
            extra_preargs=compile_preargs,
        )
        compiler.link_shared_object(
            objects,
            str(library_path),
            libraries=libraries,
            extra_postargs=link_postargs,
        )
    except Exception as exc:
        raise BTTUnavailableError(
            "Unable to build Beat-and-Tempo-Tracking. Install a working C toolchain "
            "(MSVC Build Tools, clang, or gcc) and rerun the benchmark."
        ) from exc

    metadata = {
        "library_path": str(library_path),
        "source_root": str(source_root),
        "sources": [str(path) for path in source_files],
        "compiler_type": compiler.compiler_type,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _needs_rebuild(source_root: Path, library_path: Path) -> bool:
    if not library_path.exists():
        return True
    library_mtime = library_path.stat().st_mtime
    for source_path in source_root.rglob("*"):
        if source_path.is_file() and source_path.suffix in {".c", ".h"} and source_path.stat().st_mtime > library_mtime:
            return True
    return False


def _exported_symbols() -> list[str]:
    return [
        "btt_new_default",
        "btt_destroy",
        "btt_process",
        "btt_set_tracking_mode",
        "btt_set_onset_tracking_callback",
        "btt_set_beat_tracking_callback",
        "btt_get_tempo_bpm",
        "btt_get_tempo_certainty",
        "btt_set_count_in_n",
        "btt_set_num_tempo_candidates",
        "btt_set_gaussian_tempo_histogram_decay",
        "btt_set_log_gaussian_tempo_weight_mean",
        "btt_set_log_gaussian_tempo_weight_width",
        "btt_set_onset_threshold",
        "btt_set_max_tempo",
        "btt_set_min_tempo",
    ]
