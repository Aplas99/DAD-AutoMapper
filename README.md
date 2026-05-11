# Dead as Disco Auto Mapper

Desktop beat-mapper for `Dead as Disco`.

Features:

- Import common audio formats.
- Auto-detect base tempo, beat offset, beat grid, and tempo-change sections.
- `Recommended Global` — promotes the whole map to a stronger 4/4 pulse.
- `Recommended Sections` — promotes only sections below `120 BPM`, leaving the base alone.
- `Sexy` — doubles the BPM of any part of the song below `120 BPM`, including the base tempo.
- `Undo` — steps back through any edit, bulk action, or offset drag.
- Middle-mouse drag on the timeline to scrub beat offset in real time.
- Review and edit markers against a waveform.
- Preview a metronome on top of the audio.
- Export `meta.json` and `audio.ogg` for the game's imported-song folder.

## Setup

```powershell
pip install -r requirements.txt
python -m dead_as_disco_mapper
```

To build the Windows executable:

```powershell
.\build_exe.bat
```

The built exe is fully self-contained — no Python or ffmpeg install needed on the target machine.

## Instructions

### 1. Set your folders

Set the two folder paths in the sidebar on first launch:

- **Import Folder** — where your source audio files live.
- **Game Export Folder** — the game's `ImportedSongs` directory.

Both are remembered between launches.

### 2. Import audio

Click **Import Audio** and pick a file. Supported: `.mp3 .wav .ogg .flac .m4a .mp4`. The song name pre-fills from the filename.

### 3. Auto Detect

Click **Auto Detect**. The analyzer estimates base BPM, beat offset, and any tempo-change sections. A progress bar shows while it runs. The waveform, yellow beat grid, and pink section markers appear when done.

### 4. Review the beat grid

Press **Play** to listen. Click anywhere on the waveform to move the playhead. Use **Zoom** to get closer.

If the yellow beat markers are early or late:

- Drag **middle mouse button** left or right on the timeline to scrub the offset live.
- Or edit **Offset** in the sidebar directly.

If the beat spacing is wrong, adjust **BPM**. Both update the grid instantly.

Use **Undo** at any point to step back.

### 5. Apply a recommendation (optional)

| Button | What it does |
|---|---|
| **Recommended Global** | Multiplies base tempo and all sections until ≥ 120 BPM. |
| **Recommended Sections** | Same, but only for section markers — base tempo unchanged. |
| **Sexy** | Doubles any tempo below 120 BPM exactly once, including the base. |

All three are no-ops if nothing qualifies.

### 6. Edit tempo sections

The **Tempo Sections** list shows each marker as `BPM @ time`. Clicking a row pans the timeline to it.

- **Add Marker** — places a new section at the playhead using the current BPM.
- **Replace** — overwrites the selected marker with the current playhead and BPM.
- **Delete** — removes the selected marker.

Section markers are independent anchors — they do not move when base BPM or offset changes.

### 7. Metronome

Check **Metronome** then press Play to hear a tick on every beat. Use **Metronome Volume** to balance it.

### 8. Start / End Trim (optional)

**Start Trim** and **End Trim** tell the game where song content begins and ends in seconds. Leave both at `0.0` if the audio needs no trimming.

### 9. Export

Click **Export to Game**. Two files are written into a subfolder named after the song:

- `meta.json` — tempo, offset, trim, and all custom tempo sections.
- `audio.ogg` — converted to Ogg Vorbis at 48 kHz stereo via bundled ffmpeg.

The app stays open after export for the next song.

## How It Works

Four layers, each with a narrow job:

- `analysis.py` — loads audio, runs beat tracking and per-section tempo refinement, returns an `AnalysisResult`.
- `models.py` — shared data structures (`SongProject`, `TempoSection`, `AnalysisResult`).
- `ui.py` — turns analysis output into an editable session; the final map is always what the user leaves behind.
- `exporter.py` — reads `SongProject` state and writes `meta.json` + `audio.ogg`.

### Section detection

For each structural segment the analyzer:

1. Runs `librosa.beat.beat_track` on the segment's audio.
2. Scores eight tempo candidates (base, harmonics at ×0.5 / ×0.67 / ×1.5 / ×2, plus the per-section result and its octave variants) by measuring how well synthetic beat grids align with real onsets.
3. Requires a candidate to beat the base-tempo score by a margin that scales with harmonic distance — stricter for octave-related tempos to prevent false positives.
4. Uses the winning alignment score directly as confidence.

### Audio conversion

ffmpeg converts source audio to Ogg Vorbis at 48 kHz stereo. When running as a bundled exe, ffmpeg is loaded from `_internal\ffmpeg.exe`. If ffmpeg is not found, export raises a clear error.

## Notes

- Auto-detection is a first pass. The intent is to get close enough that manual cleanup is fast.
- The exe bundles Python, all dependencies, and ffmpeg. Users need nothing extra.
