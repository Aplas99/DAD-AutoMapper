from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

import librosa

from .analysis import analyze_audio
from .btt_adapter import BTTUnavailableError, analyze_with_btt, available_profiles
from .models import TempoSection


AUDIO_EXTENSIONS = (".ogg", ".mp3", ".wav", ".flac", ".m4a", ".mp4")
PULSE_RATIOS = (0.5, 1.0, 1.5, 2.0)


@dataclass(slots=True)
class DatasetEntry:
    song_id: str
    song_name: str
    audio_path: Path
    meta_json_path: Path
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GroundTruthMap:
    song_name: str
    base_tempo: float
    beat_offset: float
    tempo_sections: list[TempoSection]
    start_song_offset: float = 0.0
    end_song_offset: float = 0.0


@dataclass(slots=True)
class AnalyzerCapture:
    analyzer: str
    available: bool
    base_tempo: float
    beat_offset: float
    beat_times: list[float]
    tempo_sections: list[TempoSection]
    duration: float
    sample_rate: int
    debug: dict[str, Any] = field(default_factory=dict)
    onset_times: list[float] = field(default_factory=list)
    tempo_updates: list[dict[str, float]] = field(default_factory=list)
    error: str | None = None


@dataclass(slots=True)
class AnalyzerMetrics:
    analyzer: str
    base_bpm_error: float
    pulse_family_correct: bool
    pulse_family_error: float
    beat_offset_error_ms: float
    beat_grid_alignment_ms: float
    section_boundary_error_ms: float
    section_boundary_hits_500ms: int
    section_boundary_hits_250ms: int
    section_tempo_exact_matches: int
    section_tempo_pulse_matches: int
    predicted_section_count: int
    ground_truth_section_count: int
    section_edit_count: int
    base_needs_manual_correction: bool
    offset_needs_manual_correction: bool
    edit_burden_total: int
    composite_penalty: float
    available: bool = True
    error: str | None = None


def run_benchmark_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark Dead as Disco analyzers against mapped songs.")
    parser.add_argument(
        "--input",
        required=True,
        help="Dataset root directory or dataset manifest JSON file.",
    )
    parser.add_argument(
        "--output",
        default="benchmark_output",
        help="Directory where JSON, CSV, and Markdown reports will be written.",
    )
    parser.add_argument(
        "--btt-profiles",
        nargs="+",
        default=["default"],
        choices=list(available_profiles()),
        help="Beat-and-Tempo-Tracking profiles to benchmark.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of songs to benchmark.",
    )
    parser.add_argument(
        "--include-tags",
        nargs="+",
        default=None,
        help="Only include songs with at least one of these tags.",
    )
    parser.add_argument(
        "--allow-missing-btt",
        action="store_true",
        help="Continue baseline benchmarking even if BTT cannot be built or loaded.",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(Path(args.input))
    if args.include_tags:
        include_tags = set(args.include_tags)
        dataset = [entry for entry in dataset if include_tags.intersection(entry.tags)]
    if args.limit is not None:
        dataset = dataset[: args.limit]
    if not dataset:
        raise SystemExit("No dataset entries matched the requested input.")

    run_benchmark(
        dataset=dataset,
        output_dir=output_dir,
        btt_profiles=args.btt_profiles,
        allow_missing_btt=args.allow_missing_btt,
    )
    print(f"Benchmark output written to {output_dir}")
    return 0


def run_benchmark(
    dataset: list[DatasetEntry],
    output_dir: Path,
    btt_profiles: list[str],
    allow_missing_btt: bool,
) -> None:
    per_song_results: list[dict[str, Any]] = []
    aggregate_inputs: defaultdict[str, list[AnalyzerMetrics]] = defaultdict(list)
    aggregate_by_tag: defaultdict[tuple[str, str], list[AnalyzerMetrics]] = defaultdict(list)

    btt_available = True
    btt_error: str | None = None

    for entry in dataset:
        ground_truth = load_ground_truth(entry)
        baseline_capture = _capture_current_analyzer(entry.audio_path)

        analyzer_captures: list[AnalyzerCapture] = [baseline_capture]
        for profile in btt_profiles:
            try:
                btt_result = analyze_with_btt(entry.audio_path, profile=profile)
                analyzer_captures.append(
                    AnalyzerCapture(
                        analyzer=f"btt:{profile}",
                        available=True,
                        base_tempo=btt_result.base_tempo,
                        beat_offset=btt_result.beat_offset,
                        beat_times=btt_result.beat_times,
                        tempo_sections=btt_result.tempo_sections,
                        duration=btt_result.duration,
                        sample_rate=btt_result.sample_rate,
                        debug=btt_result.debug,
                        onset_times=btt_result.onset_times,
                        tempo_updates=btt_result.tempo_updates,
                    )
                )
            except BTTUnavailableError as exc:
                btt_available = False
                btt_error = str(exc)
                if not allow_missing_btt:
                    raise
                analyzer_captures.append(
                    AnalyzerCapture(
                        analyzer=f"btt:{profile}",
                        available=False,
                        base_tempo=0.0,
                        beat_offset=0.0,
                        beat_times=[],
                        tempo_sections=[],
                        duration=baseline_capture.duration,
                        sample_rate=baseline_capture.sample_rate,
                        error=str(exc),
                    )
                )

        song_metrics: list[AnalyzerMetrics] = []
        for capture in analyzer_captures:
            metrics = score_capture(capture, ground_truth, duration=baseline_capture.duration)
            song_metrics.append(metrics)
            aggregate_inputs[capture.analyzer].append(metrics)
            for tag in entry.tags:
                aggregate_by_tag[(capture.analyzer, tag)].append(metrics)

        per_song_results.append(
            {
                "song_id": entry.song_id,
                "song_name": entry.song_name,
                "audio_path": str(entry.audio_path),
                "meta_json_path": str(entry.meta_json_path),
                "tags": entry.tags,
                "ground_truth": _ground_truth_to_dict(ground_truth),
                "captures": [_capture_to_dict(capture) for capture in analyzer_captures],
                "metrics": [asdict(metric) for metric in song_metrics],
            }
        )

    _write_per_song_json(output_dir / "per_song_results.json", per_song_results)
    aggregate_rows = _build_aggregate_rows(aggregate_inputs)
    _write_aggregate_csv(output_dir / "aggregate_summary.csv", aggregate_rows)
    _write_summary_markdown(
        output_dir / "summary.md",
        dataset=dataset,
        aggregate_rows=aggregate_rows,
        aggregate_by_tag=aggregate_by_tag,
        per_song_results=per_song_results,
        btt_available=btt_available,
        btt_error=btt_error,
    )


def load_dataset(input_path: Path) -> list[DatasetEntry]:
    if input_path.is_file():
        return _load_manifest(input_path)
    return _scan_dataset_root(input_path)


def load_ground_truth(entry: DatasetEntry) -> GroundTruthMap:
    payload = _load_json_file(entry.meta_json_path)
    sections = [
        TempoSection(
            tempo=float(item["tempo"]),
            start_time=float(item["startAbsoluteTime"]),
            confidence=1.0,
        )
        for item in payload.get("customTempoSections", [])
    ]
    return GroundTruthMap(
        song_name=str(payload.get("songName", entry.song_name)),
        base_tempo=float(payload.get("tempo", 120.0)),
        beat_offset=float(payload.get("beatOffset", 0.0)),
        tempo_sections=sorted(sections, key=lambda item: item.start_time),
        start_song_offset=float(payload.get("startSongOffset", 0.0)),
        end_song_offset=float(payload.get("endSongOffset", 0.0)),
    )


def score_capture(capture: AnalyzerCapture, ground_truth: GroundTruthMap, duration: float) -> AnalyzerMetrics:
    if not capture.available:
        return AnalyzerMetrics(
            analyzer=capture.analyzer,
            base_bpm_error=float("inf"),
            pulse_family_correct=False,
            pulse_family_error=float("inf"),
            beat_offset_error_ms=float("inf"),
            beat_grid_alignment_ms=float("inf"),
            section_boundary_error_ms=float("inf"),
            section_boundary_hits_500ms=0,
            section_boundary_hits_250ms=0,
            section_tempo_exact_matches=0,
            section_tempo_pulse_matches=0,
            predicted_section_count=0,
            ground_truth_section_count=len(ground_truth.tempo_sections),
            section_edit_count=len(ground_truth.tempo_sections),
            base_needs_manual_correction=True,
            offset_needs_manual_correction=True,
            edit_burden_total=len(ground_truth.tempo_sections) + 2,
            composite_penalty=float("inf"),
            available=False,
            error=capture.error,
        )

    base_bpm_error = abs(capture.base_tempo - ground_truth.base_tempo)
    pulse_family_error = min(
        abs(capture.base_tempo - ground_truth.base_tempo * ratio) for ratio in PULSE_RATIOS
    )
    pulse_family_correct = pulse_family_error <= 0.75
    beat_offset_error_ms = abs(capture.beat_offset - ground_truth.beat_offset) * 1000.0

    predicted_beats = build_beat_grid(
        capture.base_tempo,
        capture.beat_offset,
        capture.tempo_sections,
        duration,
    )
    ground_truth_beats = build_beat_grid(
        ground_truth.base_tempo,
        ground_truth.beat_offset,
        ground_truth.tempo_sections,
        duration,
    )
    beat_grid_alignment_ms = nearest_grid_distance_ms(predicted_beats, ground_truth_beats)

    section_match = compare_sections(capture.tempo_sections, ground_truth.tempo_sections)
    base_needs_manual_correction = base_bpm_error > 0.75 and not pulse_family_correct
    offset_needs_manual_correction = beat_offset_error_ms > 35.0
    edit_burden_total = section_match["edit_count"] + int(base_needs_manual_correction) + int(offset_needs_manual_correction)
    composite_penalty = (
        min(base_bpm_error, 999.0)
        + min(pulse_family_error, 999.0) * 0.5
        + min(beat_offset_error_ms, 5000.0) / 40.0
        + min(beat_grid_alignment_ms, 5000.0) / 40.0
        + min(section_match["boundary_error_ms"], 5000.0) / 100.0
        + edit_burden_total * 1.5
    )

    return AnalyzerMetrics(
        analyzer=capture.analyzer,
        base_bpm_error=round(base_bpm_error, 3),
        pulse_family_correct=pulse_family_correct,
        pulse_family_error=round(pulse_family_error, 3),
        beat_offset_error_ms=round(beat_offset_error_ms, 3),
        beat_grid_alignment_ms=round(beat_grid_alignment_ms, 3),
        section_boundary_error_ms=round(section_match["boundary_error_ms"], 3),
        section_boundary_hits_500ms=section_match["hits_500ms"],
        section_boundary_hits_250ms=section_match["hits_250ms"],
        section_tempo_exact_matches=section_match["tempo_exact_matches"],
        section_tempo_pulse_matches=section_match["tempo_pulse_matches"],
        predicted_section_count=len(capture.tempo_sections),
        ground_truth_section_count=len(ground_truth.tempo_sections),
        section_edit_count=section_match["edit_count"],
        base_needs_manual_correction=base_needs_manual_correction,
        offset_needs_manual_correction=offset_needs_manual_correction,
        edit_burden_total=edit_burden_total,
        composite_penalty=round(composite_penalty, 3),
        available=True,
        error=None,
    )


def build_beat_grid(
    base_tempo: float,
    beat_offset: float,
    tempo_sections: list[TempoSection],
    duration: float,
    beat_limit: int = 128,
) -> list[float]:
    if base_tempo <= 0 or duration <= 0:
        return []
    sections = [TempoSection(base_tempo, 0.0, 1.0)] + sorted(tempo_sections, key=lambda item: item.start_time)
    beats: list[float] = []
    for index, section in enumerate(sections):
        tempo = max(section.tempo, 1.0)
        beat_length = 60.0 / tempo
        start_time = max(section.start_time, 0.0)
        if index == 0:
            beat_time = beat_offset
            while beat_time < start_time:
                beat_time += beat_length
        else:
            beat_time = start_time
        end_time = sections[index + 1].start_time if index + 1 < len(sections) else duration
        while beat_time <= end_time and len(beats) < beat_limit:
            beats.append(round(beat_time, 6))
            beat_time += beat_length
        if len(beats) >= beat_limit:
            break
    return beats


def nearest_grid_distance_ms(predicted: list[float], ground_truth: list[float]) -> float:
    if not predicted or not ground_truth:
        return float("inf")
    distances = []
    for beat_time in predicted:
        distances.append(min(abs(beat_time - target) for target in ground_truth) * 1000.0)
    return float(mean(distances))


def compare_sections(predicted: list[TempoSection], ground_truth: list[TempoSection]) -> dict[str, int | float]:
    if not ground_truth:
        # No boundaries to find; any predictions are false positives counted as edits.
        return {
            "boundary_error_ms": 0.0,
            "hits_500ms": 0,
            "hits_250ms": 0,
            "tempo_exact_matches": 0,
            "tempo_pulse_matches": 0,
            "edit_count": len(predicted),
        }

    remaining = predicted.copy()
    boundary_errors_ms: list[float] = []
    hits_500ms = 0
    hits_250ms = 0
    tempo_exact_matches = 0
    tempo_pulse_matches = 0
    matched_predictions = 0

    for target in ground_truth:
        if not remaining:
            break
        nearest = min(remaining, key=lambda item: abs(item.start_time - target.start_time))
        remaining.remove(nearest)
        matched_predictions += 1
        boundary_error_ms = abs(nearest.start_time - target.start_time) * 1000.0
        boundary_errors_ms.append(boundary_error_ms)
        if boundary_error_ms <= 500.0:
            hits_500ms += 1
        if boundary_error_ms <= 250.0:
            hits_250ms += 1
        if abs(nearest.tempo - target.tempo) <= 0.75:
            tempo_exact_matches += 1
        if min(abs(nearest.tempo - target.tempo * ratio) for ratio in PULSE_RATIOS) <= 0.75:
            tempo_pulse_matches += 1

    unmatched_ground_truth = max(0, len(ground_truth) - matched_predictions)
    unmatched_predictions = max(0, len(predicted) - matched_predictions)
    moved_markers = sum(1 for error in boundary_errors_ms if error > 500.0)
    tempo_mismatches = max(0, matched_predictions - tempo_exact_matches)
    edit_count = unmatched_ground_truth + unmatched_predictions + moved_markers + tempo_mismatches

    return {
        "boundary_error_ms": float(mean(boundary_errors_ms)) if boundary_errors_ms else float("inf"),
        "hits_500ms": hits_500ms,
        "hits_250ms": hits_250ms,
        "tempo_exact_matches": tempo_exact_matches,
        "tempo_pulse_matches": tempo_pulse_matches,
        "edit_count": edit_count,
    }


def _load_manifest(manifest_path: Path) -> list[DatasetEntry]:
    payload = _load_json_file(manifest_path)
    items = payload["songs"] if isinstance(payload, dict) else payload
    entries: list[DatasetEntry] = []
    for index, item in enumerate(items):
        audio_path = (manifest_path.parent / item["audio_path"]).resolve()
        meta_json_path = (manifest_path.parent / item["meta_json_path"]).resolve()
        song_name = item.get("song_name") or meta_json_path.parent.name
        entries.append(
            DatasetEntry(
                song_id=item.get("song_id", f"song_{index:03d}"),
                song_name=song_name,
                audio_path=audio_path,
                meta_json_path=meta_json_path,
                tags=sorted(set(item.get("tags", []))),
            )
        )
    return entries


def _scan_dataset_root(root: Path) -> list[DatasetEntry]:
    entries: list[DatasetEntry] = []
    for meta_json_path in sorted(root.rglob("meta.json")):
        song_dir = meta_json_path.parent
        audio_path = _find_audio_file(song_dir)
        if audio_path is None:
            continue
        payload = _load_json_file(meta_json_path)
        tags = _auto_tags_from_meta(payload)
        entries.append(
            DatasetEntry(
                song_id=song_dir.name,
                song_name=str(payload.get("songName", song_dir.name)),
                audio_path=audio_path,
                meta_json_path=meta_json_path,
                tags=tags,
            )
        )
    return entries


def _find_audio_file(song_dir: Path) -> Path | None:
    for candidate in ("audio.ogg", "Audio.ogg"):
        path = song_dir / candidate
        if path.exists():
            return path.resolve()
    for extension in AUDIO_EXTENSIONS:
        matches = sorted(song_dir.glob(f"*{extension}"))
        if matches:
            return matches[0].resolve()
    return None


def _load_json_file(path: Path) -> dict[str, Any] | list[Any]:
    encodings = ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be")
    last_error: UnicodeDecodeError | json.JSONDecodeError | None = None
    for encoding in encodings:
        try:
            return json.loads(path.read_text(encoding=encoding))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Unable to read JSON file {path}")


def _auto_tags_from_meta(payload: dict[str, Any]) -> list[str]:
    tags: set[str] = set()
    sections = payload.get("customTempoSections", [])
    base_tempo = float(payload.get("tempo", 120.0))
    tags.add("tempo_change" if sections else "constant_tempo")
    if base_tempo < 100:
        tags.add("halftime")
    for section in sections:
        tempo = float(section.get("tempo", base_tempo))
        ratio = tempo / max(base_tempo, 1.0)
        if abs(ratio - 1.5) < 0.08 or abs(ratio - (2.0 / 3.0)) < 0.08:
            tags.add("triplet_feel")
        if tempo < 100:
            tags.add("halftime")
    if len(sections) >= 2 or "triplet_feel" in tags:
        tags.add("hard_case")
    return sorted(tags)


def _capture_current_analyzer(audio_path: Path) -> AnalyzerCapture:
    result = analyze_audio(audio_path)
    return AnalyzerCapture(
        analyzer="current",
        available=True,
        base_tempo=result.base_tempo,
        beat_offset=result.beat_offset,
        beat_times=[round(value, 6) for value in result.beat_times],
        tempo_sections=result.tempo_sections,
        duration=result.duration,
        sample_rate=result.sample_rate,
        debug={"waveform_points": len(result.waveform_times)},
    )


def _capture_to_dict(capture: AnalyzerCapture) -> dict[str, Any]:
    return {
        "analyzer": capture.analyzer,
        "available": capture.available,
        "base_tempo": capture.base_tempo,
        "beat_offset": capture.beat_offset,
        "beat_times": capture.beat_times,
        "tempo_sections": [asdict(section) for section in capture.tempo_sections],
        "duration": capture.duration,
        "sample_rate": capture.sample_rate,
        "debug": capture.debug,
        "onset_times": capture.onset_times,
        "tempo_updates": capture.tempo_updates,
        "error": capture.error,
    }


def _ground_truth_to_dict(ground_truth: GroundTruthMap) -> dict[str, Any]:
    return {
        "song_name": ground_truth.song_name,
        "base_tempo": ground_truth.base_tempo,
        "beat_offset": ground_truth.beat_offset,
        "tempo_sections": [asdict(section) for section in ground_truth.tempo_sections],
        "start_song_offset": ground_truth.start_song_offset,
        "end_song_offset": ground_truth.end_song_offset,
    }


def _write_per_song_json(path: Path, payload: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_aggregate_rows(aggregate_inputs: dict[str, list[AnalyzerMetrics]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for analyzer, metrics_list in sorted(aggregate_inputs.items()):
        available_metrics = [metric for metric in metrics_list if metric.available]
        if not available_metrics:
            rows.append(
                {
                    "analyzer": analyzer,
                    "songs": len(metrics_list),
                    "available_songs": 0,
                    "mean_base_bpm_error": "",
                    "mean_beat_offset_error_ms": "",
                    "mean_beat_grid_alignment_ms": "",
                    "mean_section_boundary_error_ms": "",
                    "mean_edit_burden_total": "",
                    "pulse_family_correct_rate": "",
                    "mean_composite_penalty": "",
                    "error": next((metric.error for metric in metrics_list if metric.error), ""),
                }
            )
            continue
        rows.append(
            {
                "analyzer": analyzer,
                "songs": len(metrics_list),
                "available_songs": len(available_metrics),
                "mean_base_bpm_error": _mean_attr(available_metrics, "base_bpm_error"),
                "mean_beat_offset_error_ms": _mean_attr(available_metrics, "beat_offset_error_ms"),
                "mean_beat_grid_alignment_ms": _mean_attr(available_metrics, "beat_grid_alignment_ms"),
                "mean_section_boundary_error_ms": _mean_attr(available_metrics, "section_boundary_error_ms"),
                "mean_edit_burden_total": _mean_attr(available_metrics, "edit_burden_total"),
                "pulse_family_correct_rate": round(
                    sum(1 for item in available_metrics if item.pulse_family_correct) / len(available_metrics),
                    4,
                ),
                "mean_composite_penalty": _mean_attr(available_metrics, "composite_penalty"),
                "error": "",
            }
        )
    return rows


def _write_aggregate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_markdown(
    path: Path,
    dataset: list[DatasetEntry],
    aggregate_rows: list[dict[str, Any]],
    aggregate_by_tag: dict[tuple[str, str], list[AnalyzerMetrics]],
    per_song_results: list[dict[str, Any]],
    btt_available: bool,
    btt_error: str | None,
) -> None:
    lines = [
        "# Beat-and-Tempo-Tracking Benchmark Summary",
        "",
        f"- Songs benchmarked: {len(dataset)}",
        f"- BTT available: {'yes' if btt_available else 'no'}",
    ]
    if btt_error:
        lines.append(f"- BTT issue: {btt_error}")
    lines.extend(["", "## Overall", "", "| Analyzer | Songs | Mean BPM Error | Mean Offset Error (ms) | Mean Beat Grid Error (ms) | Mean Section Boundary Error (ms) | Mean Edit Burden | Mean Composite Penalty |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in aggregate_rows:
        lines.append(
            "| {analyzer} | {songs} | {mean_base_bpm_error} | {mean_beat_offset_error_ms} | "
            "{mean_beat_grid_alignment_ms} | {mean_section_boundary_error_ms} | {mean_edit_burden_total} | "
            "{mean_composite_penalty} |".format(**row)
        )

    if aggregate_by_tag:
        lines.extend(["", "## By Tag", ""])
        for (analyzer, tag), metrics_list in sorted(aggregate_by_tag.items()):
            available_metrics = [metric for metric in metrics_list if metric.available]
            if not available_metrics:
                continue
            lines.append(
                f"- `{analyzer}` / `{tag}`: "
                f"BPM { _mean_attr(available_metrics, 'base_bpm_error') }, "
                f"offset { _mean_attr(available_metrics, 'beat_offset_error_ms') } ms, "
                f"edit burden { _mean_attr(available_metrics, 'edit_burden_total') }"
            )

    comparison = _top_improvements_and_regressions(per_song_results)
    if comparison["improvements"] or comparison["regressions"]:
        lines.extend(["", "## Current vs BTT Default", ""])
        if comparison["improvements"]:
            lines.append("### Top Improvements")
            for item in comparison["improvements"]:
                lines.append(
                    f"- {item['song_name']}: composite delta {item['delta']}, "
                    f"current {item['current_penalty']} -> btt {item['btt_penalty']}"
                )
        if comparison["regressions"]:
            lines.append("")
            lines.append("### Top Regressions")
            for item in comparison["regressions"]:
                lines.append(
                    f"- {item['song_name']}: composite delta {item['delta']}, "
                    f"current {item['current_penalty']} -> btt {item['btt_penalty']}"
                )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _top_improvements_and_regressions(per_song_results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    deltas: list[dict[str, Any]] = []
    for song in per_song_results:
        metrics_by_analyzer = {item["analyzer"]: item for item in song["metrics"]}
        current = metrics_by_analyzer.get("current")
        btt = metrics_by_analyzer.get("btt:default")
        if not current or not btt or not current["available"] or not btt["available"]:
            continue
        delta = round(btt["composite_penalty"] - current["composite_penalty"], 3)
        deltas.append(
            {
                "song_name": song["song_name"],
                "delta": delta,
                "current_penalty": current["composite_penalty"],
                "btt_penalty": btt["composite_penalty"],
            }
        )
    improvements = sorted((item for item in deltas if item["delta"] < 0), key=lambda item: item["delta"])[:10]
    regressions = sorted((item for item in deltas if item["delta"] > 0), key=lambda item: item["delta"], reverse=True)[:10]
    return {"improvements": improvements, "regressions": regressions}


def _mean_attr(items: list[AnalyzerMetrics], attr: str) -> float:
    return round(float(mean(getattr(item, attr) for item in items)), 3)
