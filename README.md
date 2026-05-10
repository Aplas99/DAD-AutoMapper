# Dead as Disco Auto Mapper

Desktop beat-mapper for `Dead as Disco`.

Features:

- Import common audio formats.
- Auto-detect base tempo, beat offset, beat grid, and tempo-change sections.
- Apply `Recommended Global` to promote an entire slower map into a stronger 4/4-feeling pulse.
- Apply `Recommended Sections` to promote only tempo-change markers below `120 BPM` while leaving the base tempo alone.
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

## Instructions

### 1. Set your folders

On first launch, set the two folder paths in the sidebar:

- **Import Folder** — the folder where your source audio files live. The file browser opens here when you click Import Audio.
- **Game Export Folder** — the game's `ImportedSongs` directory. Both paths are remembered between launches.

### 2. Import audio

Click **Import Audio** (or use the `Open` menu item) and select your file. Supported formats: `.mp3`, `.wav`, `.ogg`, `.flac`, `.m4a`, `.mp4`. The song name is pre-filled from the filename and can be edited.

### 3. Run auto-detect

Click **Auto Detect**. The analyzer will estimate:

- base tempo (BPM)
- beat offset (seconds to the first beat)
- structural section boundaries
- tempo-change markers for sections that feel different from the base pulse

This takes a few seconds. A progress bar shows while it runs. When it finishes, the waveform, yellow beat grid, and pink tempo-section markers appear on the timeline.

### 4. Review the result

Listen back with **Play**. Playback starts from wherever the timeline cursor is — click anywhere on the waveform to move it. Use the **Zoom** slider to zoom in on a specific area.

Check that the yellow beat markers land on the beats of the song. If the grid is slightly early or late:

- Adjust **Offset** in the sidebar to shift all beats forward or backward in time.
- Hold **middle mouse button** and drag left or right on the timeline to scrub the offset in real time — drag speed scales with your current zoom level.
- Adjust **BPM** if the beat spacing itself is wrong.

Both controls update the beat grid instantly as you type or drag.

### 5. Apply a recommendation (optional)

If auto-detect found a base tempo that feels half-time (e.g. `65 BPM` for a song that clearly pulses at `130 BPM`):

- **Recommended Global** — doubles (or re-multiplies) the base tempo and all section tempos together so the whole map shifts to a stronger 4/4 pulse.
- **Recommended Sections** — leaves the base tempo alone and only promotes individual tempo-change markers that are below `120 BPM`.

Both buttons are no-ops if the tempo is already at a reasonable value.

### 6. Edit tempo sections

The **Tempo Sections** list in the sidebar shows each detected section as `BPM @ time`. Clicking a section pans the timeline to it.

- **Add Marker** — adds a new section at the current playhead position using the current BPM value.
- **Replace** — overwrites the selected marker with the current playhead position and BPM value.
- **Delete** — removes the selected marker.

Section markers are pink vertical lines on the timeline. They do not move when you change the base BPM or offset — they are independent anchors.

### 7. Preview with the metronome

Check **Metronome** before pressing Play to hear an audible tick on every beat. Use the **Metronome Volume** slider to balance it against the track. The tick uses the base BPM and offset — it does not yet follow tempo-section changes mid-song.

### 8. Set start and end trim (optional)

**Start Trim** and **End Trim** let you tell the game where the song content begins and ends within the audio file, in seconds. Leave both at `0.0` if the audio needs no trimming.

### 9. Export

Click **Export to Game**. The exporter writes two files into a subfolder named after the song inside your Game Export Folder:

- `meta.json` — tempo, offset, trim, and all custom tempo sections.
- `audio.ogg` — the source file converted to Ogg Vorbis at 48 kHz stereo via the bundled `ffmpeg`.

The app stays open after export so you can immediately start on the next song.

## How It Works

The tool is split into four main parts:

- `PySide6` provides the desktop application window, playback controls, saved settings, status indicators, and export workflow.
- `pyqtgraph` renders the waveform, beat grid, playhead, and tempo-change markers at interactive speed.
- `librosa`, `numpy`, and `scipy` handle beat tracking, onset analysis, structural segmentation, and tempo heuristics.
- `ffmpeg` converts user audio into a game-friendly `audio.ogg` file using Ogg Vorbis at `48 kHz` stereo.

At a high level, the workflow is:

1. The user imports an audio file.
2. The analyzer loads the audio and estimates:
   - base tempo
   - beat offset
   - section boundaries
   - tempo-change markers
3. The UI displays:
   - the waveform
   - the live beat grid
   - detected custom tempo markers
4. The user can optionally apply one of the recommendation modes:
   - `Recommended Global` to lift the whole map when the base pulse is too slow
   - `Recommended Sections` to lift only slower custom tempo sections
5. The user reviews and edits the map.
6. The exporter writes:
   - `meta.json`
   - `audio.ogg`

## Technology Interaction

The tool works because each layer has a narrow job and passes structured data to the next layer.

- `analysis.py` produces an `AnalysisResult` object containing waveform data, beat timing, section timing, and detected tempo sections.
- `models.py` defines the shared project data structures used by both the analyzer and the UI.
- `ui.py` turns that analysis data into an editable session. The waveform and markers come from the analyzer, but the final map is always whatever the user leaves in the editor.
- `exporter.py` reads the current `SongProject` state and converts it into the game format.

That separation matters:

- the analyzer can be imperfect without breaking export
- the UI can correct the analyzer without rerunning everything
- the exporter does not need to understand signal processing, only final project state

## Why This Works In Practice

The game format is simple enough that a full custom pipeline is realistic:

- one base `tempo`
- one `beatOffset`
- zero or more `customTempoSections`
- one `audio.ogg`

The harder part is not writing JSON. The harder part is getting close enough musically that manual cleanup is fast. That is why the analyzer is built as a first-pass mapper instead of a fully automatic authoritative mapper.

The section detection combines multiple signals:

- beat tracking for the main pulse
- onset strength for rhythmic intensity
- structural segmentation for large musical section changes
- tempo heuristics for switching between the base pulse and alternate feel sections such as `1.5x`

This is how the tool can get reasonably close on songs like `Golden` while still leaving the final decision to the editor.

## Audio Conversion

`ffmpeg` is the default converter because it is more reliable than writing Ogg Vorbis directly through the Python audio stack for arbitrary user files.

The exporter currently aims to match the working in-game files with:

- Ogg container
- Vorbis codec
- `48000 Hz`
- stereo

If `ffmpeg` is not found, the exporter raises a clear error rather than attempting a lower-quality fallback.

## UI Behavior

The UI is designed around iterative mapping rather than one-shot export.

- The app stays open after export so multiple songs can be processed in one session.
- Import and export folders are saved between launches.
- The metronome has a dedicated volume control and uses a generated soft ding instead of the Windows default beep.
- Auto-detect and export show a visible busy state so long-running tasks are obvious.
- Playback starts from the current timeline cursor, not from the beginning every time.
- Editing `BPM` or `Offset` immediately rebuilds the visible yellow beat markers from the live control values.
- Tempo section markers remain explicit anchors and do not silently move when only the base BPM or offset changes.
- The recommendation tools are deliberately split into global and section-only modes so users can choose whether the pulse correction should affect the whole song or only slower mapped sections.

## Notes

- Auto-detection is heuristic. The intent is to get close enough that manual cleanup is fast.
- The exporter writes `audio.ogg` through `ffmpeg` by default. If conversion is unavailable, it raises a clear error.
