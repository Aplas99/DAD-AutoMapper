from __future__ import annotations

import math
import struct
import tempfile
import wave
from pathlib import Path

import pyqtgraph as pg
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QEvent, QObject, QSettings, QTimer, Qt, QUrl
from PySide6.QtGui import QAction
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer, QSoundEffect
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)

from .analysis import analyze_audio, recommended_section_tempo_mapping, recommended_tempo_mapping
from .exporter import export_project
from .models import AnalysisResult, SongProject, TempoSection


class _MMBOffsetFilter(QObject):
    """Converts middle-mouse-button horizontal drag into beat-offset scrubbing."""

    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        self._window = window
        self._active = False
        self._start_x = 0.0
        self._start_offset = 0.0

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        t = event.type()
        if t == QEvent.Type.MouseButtonPress and event.button() == Qt.MiddleButton:
            self._active = True
            self._start_x = float(event.position().x())
            self._start_offset = self._window.offset_spin.value()
            return True
        if t == QEvent.Type.MouseMove and self._active:
            delta_px = float(event.position().x()) - self._start_x
            vb = self._window.plot.getPlotItem().vb
            x_range = vb.viewRange()[0]
            width_px = max(float(vb.width()), 1.0)
            delta_s = delta_px * (x_range[1] - x_range[0]) / width_px
            new_val = max(-2.0, min(2.0, round(self._start_offset + delta_s, 4)))
            self._window.offset_spin.setValue(new_val)
            return True
        if t == QEvent.Type.MouseButtonRelease and event.button() == Qt.MiddleButton:
            self._active = False
            return True
        return False


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Dead as Disco Auto Mapper")
        self.resize(1600, 900)

        self.settings = QSettings("Codex", "DeadAsDiscoAutoMapper")
        self.project = SongProject(song_name="Imported Song")
        self.analysis_result: AnalysisResult | None = None

        self.audio_output = QAudioOutput()
        self.audio_output.setVolume(0.8)
        self.player = QMediaPlayer()
        self.player.setAudioOutput(self.audio_output)
        self.player.positionChanged.connect(self._sync_playhead)

        self.metronome_enabled = False
        self.metronome_timer = QTimer(self)
        self.metronome_timer.timeout.connect(self._advance_metronome)
        self._metronome_next_ms = 0
        self._busy_depth = 0

        self.metronome_sound = QSoundEffect(self)
        self.metronome_sound.setVolume(0.18)
        self.metronome_sound.setSource(QUrl.fromLocalFile(str(self._ensure_metronome_sound())))

        self._build_ui()
        self._load_settings()

        self._mmb_filter = _MMBOffsetFilter(self)
        self.plot.viewport().installEventFilter(self._mmb_filter)

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)

        sidebar = QWidget()
        sidebar.setFixedWidth(360)
        sidebar_layout = QVBoxLayout(sidebar)
        outer.addWidget(sidebar)

        sidebar_layout.addWidget(QLabel("Import Folder"))
        import_folder_row = QHBoxLayout()
        self.import_folder_edit = QLineEdit()
        self.import_folder_edit.editingFinished.connect(self._persist_settings)
        import_folder_row.addWidget(self.import_folder_edit)
        import_folder_button = QPushButton("Browse")
        import_folder_button.clicked.connect(self._choose_import_folder)
        import_folder_row.addWidget(import_folder_button)
        sidebar_layout.addLayout(import_folder_row)

        sidebar_layout.addWidget(QLabel("Game Export Folder"))
        export_folder_row = QHBoxLayout()
        self.export_folder_edit = QLineEdit()
        self.export_folder_edit.editingFinished.connect(self._persist_settings)
        export_folder_row.addWidget(self.export_folder_edit)
        export_folder_button = QPushButton("Browse")
        export_folder_button.clicked.connect(self._choose_export_folder)
        export_folder_row.addWidget(export_folder_button)
        sidebar_layout.addLayout(export_folder_row)

        self.import_button = QPushButton("Import Audio")
        self.import_button.clicked.connect(self._import_audio)
        sidebar_layout.addWidget(self.import_button)

        self.auto_button = QPushButton("Auto Detect")
        self.auto_button.clicked.connect(self._auto_detect)
        sidebar_layout.addWidget(self.auto_button)

        self.recommended_global_button = QPushButton("Recommended Global")
        self.recommended_global_button.clicked.connect(self._apply_recommended_tempo)
        sidebar_layout.addWidget(self.recommended_global_button)

        self.recommended_sections_button = QPushButton("Recommended Sections")
        self.recommended_sections_button.clicked.connect(self._apply_recommended_sections)
        sidebar_layout.addWidget(self.recommended_sections_button)

        form = QFormLayout()
        self.song_name = QLineEdit("Imported Song")
        self.song_name.textChanged.connect(self._apply_song_name)
        form.addRow("Song", self.song_name)

        self.bpm_spin = QDoubleSpinBox()
        self.bpm_spin.setRange(40.0, 260.0)
        self.bpm_spin.setDecimals(3)
        self.bpm_spin.setValue(120.0)
        self.bpm_spin.valueChanged.connect(self._update_project_from_controls)
        form.addRow("BPM", self.bpm_spin)

        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(-2.0, 2.0)
        self.offset_spin.setDecimals(4)
        self.offset_spin.setSingleStep(0.001)
        self.offset_spin.valueChanged.connect(self._update_project_from_controls)
        form.addRow("Offset", self.offset_spin)

        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(0.0, 9999.0)
        self.start_spin.setDecimals(3)
        self.start_spin.valueChanged.connect(self._update_project_from_controls)
        form.addRow("Start Trim", self.start_spin)

        self.end_spin = QDoubleSpinBox()
        self.end_spin.setRange(0.0, 9999.0)
        self.end_spin.setDecimals(3)
        self.end_spin.valueChanged.connect(self._update_project_from_controls)
        form.addRow("End Trim", self.end_spin)

        sidebar_layout.addLayout(form)

        self.section_list = QListWidget()
        self.section_list.currentRowChanged.connect(self._focus_section)
        sidebar_layout.addWidget(QLabel("Tempo Sections"))
        sidebar_layout.addWidget(self.section_list, 1)

        section_controls = QHBoxLayout()
        add_section = QPushButton("Add Marker")
        add_section.clicked.connect(self._add_marker_at_playhead)
        section_controls.addWidget(add_section)

        replace_section = QPushButton("Replace")
        replace_section.clicked.connect(self._replace_selected_marker)
        section_controls.addWidget(replace_section)

        delete_section = QPushButton("Delete")
        delete_section.clicked.connect(self._delete_selected_marker)
        section_controls.addWidget(delete_section)
        sidebar_layout.addLayout(section_controls)

        playback_controls = QHBoxLayout()
        play_button = QPushButton("Play")
        play_button.clicked.connect(self._toggle_playback)
        playback_controls.addWidget(play_button)

        stop_button = QPushButton("Stop")
        stop_button.clicked.connect(self._stop_playback)
        playback_controls.addWidget(stop_button)

        self.metronome_box = QCheckBox("Metronome")
        self.metronome_box.toggled.connect(self._toggle_metronome)
        playback_controls.addWidget(self.metronome_box)
        sidebar_layout.addLayout(playback_controls)

        self.metronome_volume_slider = QSlider(Qt.Horizontal)
        self.metronome_volume_slider.setRange(0, 100)
        self.metronome_volume_slider.setValue(18)
        self.metronome_volume_slider.valueChanged.connect(self._update_metronome_volume)
        sidebar_layout.addWidget(QLabel("Metronome Volume"))
        sidebar_layout.addWidget(self.metronome_volume_slider)

        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(1, 100)
        self.zoom_slider.setValue(10)
        self.zoom_slider.valueChanged.connect(self._update_zoom)
        sidebar_layout.addWidget(QLabel("Zoom"))
        sidebar_layout.addWidget(self.zoom_slider)

        self.export_button = QPushButton("Export to Game")
        self.export_button.clicked.connect(self._export_song)
        sidebar_layout.addWidget(self.export_button)

        self.busy_bar = QProgressBar()
        self.busy_bar.setRange(0, 0)
        self.busy_bar.setVisible(False)
        sidebar_layout.addWidget(self.busy_bar)

        self.status_label = QLabel("Load audio to begin.")
        self.status_label.setWordWrap(True)
        sidebar_layout.addWidget(self.status_label)

        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        outer.addWidget(plot_container, 1)

        self.plot = pg.PlotWidget(background="#0d0f14")
        self.plot.showGrid(x=True, y=True, alpha=0.18)
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=True, y=False)
        self.plot.getPlotItem().setLabel("bottom", "Time", units="s")
        self.plot.scene().sigMouseClicked.connect(self._handle_plot_click)
        plot_layout.addWidget(self.plot)

        self.waveform_curve = self.plot.plot(pen=pg.mkPen("#23d6cf", width=1.1))
        self.waveform_fill_top = self.plot.plot(pen=pg.mkPen("#1ab4ff", width=1.0))
        self.waveform_fill_bottom = self.plot.plot(pen=pg.mkPen("#1ab4ff", width=1.0))
        self.playhead_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#ffffff", width=2))
        self.plot.addItem(self.playhead_line)

        self.beat_lines: list[pg.InfiniteLine] = []
        self.section_lines: list[pg.InfiniteLine] = []

        open_action = QAction("Open", self)
        open_action.triggered.connect(self._import_audio)
        self.menuBar().addAction(open_action)

    def _import_audio(self) -> None:
        initial_dir = self.import_folder_edit.text().strip() or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio",
            initial_dir,
            "Audio Files (*.mp3 *.wav *.ogg *.flac *.m4a *.mp4)",
        )
        if not path:
            return
        song_path = Path(path)
        self.import_folder_edit.setText(str(song_path.parent))
        self._persist_settings()
        self.project.audio_path = song_path
        self.project.song_name = song_path.stem
        self.song_name.setText(song_path.stem)
        self.player.setSource(QUrl.fromLocalFile(str(song_path)))
        self.status_label.setText(f"Loaded {song_path.name}. Run Auto Detect next.")

    def _auto_detect(self) -> None:
        if not self.project.audio_path:
            QMessageBox.warning(self, "No Audio", "Load an audio file first.")
            return
        self._set_busy(True, "Analyzing audio...")
        try:
            self.analysis_result = analyze_audio(self.project.audio_path)
            self.project.song_name = self.song_name.text().strip() or self.project.audio_path.stem
            self.project.base_tempo = self.analysis_result.base_tempo
            self.project.beat_offset = self.analysis_result.beat_offset
            self.project.beat_times = list(self.analysis_result.beat_times)
            self.project.tempo_sections = list(self.analysis_result.tempo_sections)
            self.project.waveform_times = list(self.analysis_result.waveform_times)
            self.project.waveform_values = list(self.analysis_result.waveform_values)
            self.project.raw_waveform_values = list(self.analysis_result.raw_waveform_values)
            self.project.duration = self.analysis_result.duration
            self.project.sample_rate = self.analysis_result.sample_rate
            self._sync_controls()
            self._draw_waveform()
            self._refresh_sections()
            self.status_label.setText(
                f"Detected {self.project.base_tempo:.2f} BPM, offset {self.project.beat_offset:.4f}, "
                f"{len(self.project.tempo_sections)} tempo markers."
            )
        except Exception as exc:  # pragma: no cover - surfaced in UI
            QMessageBox.critical(self, "Analysis Failed", str(exc))
        finally:
            self._set_busy(False)

    def _sync_controls(self) -> None:
        self.bpm_spin.setValue(self.project.base_tempo)
        self.offset_spin.setValue(self.project.beat_offset)
        self.start_spin.setValue(self.project.start_song_offset)
        self.end_spin.setValue(self.project.end_song_offset)

    def _apply_recommended_tempo(self) -> None:
        if self.project.base_tempo <= 0:
            QMessageBox.information(self, "No Tempo", "Run Auto Detect or enter a BPM first.")
            return
        original_tempo = self.project.base_tempo
        recommended_tempo, recommended_sections = recommended_tempo_mapping(
            self.project.base_tempo,
            self.project.tempo_sections,
        )
        if abs(recommended_tempo - original_tempo) < 0.01:
            self.status_label.setText(
                f"Recommended setting left the map at {self.project.base_tempo:.2f} BPM."
            )
            return
        self.project.base_tempo = recommended_tempo
        self.project.tempo_sections = recommended_sections
        self._sync_controls()
        self._draw_beat_grid()
        self._refresh_sections()
        self.status_label.setText(
            f"Recommended setting promoted {original_tempo:.2f} BPM to {recommended_tempo:.2f} BPM "
            "for a stronger 4/4 pulse."
        )

    def _apply_recommended_sections(self) -> None:
        if not self.project.tempo_sections:
            QMessageBox.information(self, "No Sections", "Load a map with tempo sections first.")
            return
        original_sections = list(self.project.tempo_sections)
        recommended_sections = recommended_section_tempo_mapping(self.project.tempo_sections)
        changes = sum(
            1
            for original, updated in zip(original_sections, recommended_sections)
            if abs(original.tempo - updated.tempo) >= 0.01
        )
        self.project.tempo_sections = recommended_sections
        self._refresh_sections()
        self._draw_beat_grid()
        if changes == 0:
            self.status_label.setText("Recommended Sections left all tempo markers unchanged.")
            return
        self.status_label.setText(
            f"Recommended Sections promoted {changes} marker"
            f"{'' if changes == 1 else 's'} below 120 BPM to a stronger 4/4 pulse."
        )

    def _apply_song_name(self, value: str) -> None:
        self.project.song_name = value.strip() or "Imported Song"

    def _update_project_from_controls(self) -> None:
        self.project.base_tempo = self.bpm_spin.value()
        self.project.beat_offset = self.offset_spin.value()
        self.project.start_song_offset = self.start_spin.value()
        self.project.end_song_offset = self.end_spin.value()
        self._draw_beat_grid()
        self._refresh_sections()

    def _draw_waveform(self) -> None:
        times = self.project.waveform_times
        peaks = self.project.waveform_values
        raw = self.project.raw_waveform_values or [0.0 for _ in peaks]
        self.waveform_curve.setData(times, raw)
        self.waveform_fill_top.setData(times, peaks)
        self.waveform_fill_bottom.setData(times, [-value for value in peaks])
        self.plot.setXRange(0.0, max(self.project.duration, 1.0), padding=0.01)
        max_peak = max(peaks or [1.0]) * 1.1
        self.plot.setYRange(-max_peak, max_peak)
        self._draw_beat_grid()
        self._draw_section_lines()

    def _draw_beat_grid(self) -> None:
        x_range = self.plot.viewRange()[0]

        beat_times = self._display_beat_times()
        target = beat_times[:2000]

        # Remove surplus lines from the end.
        while len(self.beat_lines) > len(target):
            self.plot.removeItem(self.beat_lines.pop())

        # Update existing lines in-place; add new ones only when needed.
        pen = pg.mkPen("#d8b400", width=1, style=Qt.DashLine)
        for i, bt in enumerate(target):
            if i < len(self.beat_lines):
                self.beat_lines[i].setPos(bt)
            else:
                line = pg.InfiniteLine(angle=90, movable=False, pen=pen)
                line.setPos(bt)
                self.plot.addItem(line)
                self.beat_lines.append(line)

        # Restore the view so item additions never reset the zoom.
        if x_range[1] > x_range[0]:
            self.plot.setXRange(x_range[0], x_range[1], padding=0)

    def _draw_section_lines(self) -> None:
        for line in self.section_lines:
            self.plot.removeItem(line)
        self.section_lines.clear()

        for section in self.project.tempo_sections:
            line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#ff1fe0", width=2))
            line.setPos(section.start_time)
            self.plot.addItem(line)
            self.section_lines.append(line)

    def _synthetic_beat_times(self) -> list[float]:
        if self.project.base_tempo <= 0 or self.project.duration <= 0:
            return []
        sections = [TempoSection(self.project.base_tempo, 0.0, 1.0)] + sorted(
            self.project.tempo_sections,
            key=lambda item: item.start_time,
        )
        beats: list[float] = []
        for index, section in enumerate(sections):
            tempo = max(section.tempo, 1.0)
            beat_length = 60.0 / tempo
            start_time = max(section.start_time, 0.0)
            if index == 0:
                beat_time = self.project.beat_offset
                while beat_time < start_time:
                    beat_time += beat_length
            else:
                beat_time = start_time
            end_time = sections[index + 1].start_time if index + 1 < len(sections) else self.project.duration
            while beat_time <= end_time:
                beats.append(round(beat_time, 4))
                beat_time += beat_length
        return beats

    def _display_beat_times(self) -> list[float]:
        # Always rebuild the displayed beat grid from the live controls so
        # BPM and offset edits immediately move the visible markers.
        return self._synthetic_beat_times()

    def _refresh_sections(self) -> None:
        self.section_list.clear()
        for section in self.project.tempo_sections:
            item = QListWidgetItem(
                f"{section.tempo:.2f} BPM @ {section.start_time:.3f}s  confidence {section.confidence:.2f}"
            )
            self.section_list.addItem(item)
        self._draw_section_lines()

    def _focus_section(self, row: int) -> None:
        if row < 0 or row >= len(self.project.tempo_sections):
            return
        section = self.project.tempo_sections[row]
        span = max(8.0, self.plot.viewRange()[0][1] - self.plot.viewRange()[0][0])
        start = max(section.start_time - span * 0.25, 0.0)
        end = min(section.start_time + span * 0.75, self.project.duration or section.start_time + span)
        self.plot.setXRange(start, end, padding=0.02)

    def _handle_plot_click(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        mouse_point = self.plot.getPlotItem().vb.mapSceneToView(event.scenePos())
        bounded = max(0.0, min(mouse_point.x(), self.project.duration or mouse_point.x()))
        self.playhead_line.setPos(bounded)

    def _cursor_seconds(self) -> float:
        if self.project.duration <= 0:
            return 0.0
        cursor = float(self.playhead_line.value())
        if cursor > 0:
            return max(0.0, min(cursor, self.project.duration))
        return max(0.0, min(float(self.player.position()) / 1000.0, self.project.duration))

    def _add_marker_at_playhead(self) -> None:
        time_value = self._cursor_seconds()
        tempo = self.bpm_spin.value()
        self.project.tempo_sections.append(TempoSection(tempo=tempo, start_time=round(time_value, 4), confidence=1.0))
        self.project.tempo_sections.sort(key=lambda item: item.start_time)
        self._refresh_sections()
        self._draw_beat_grid()

    def _replace_selected_marker(self) -> None:
        row = self.section_list.currentRow()
        if row < 0:
            return
        self.project.tempo_sections[row] = TempoSection(
            tempo=self.bpm_spin.value(),
            start_time=round(self._cursor_seconds(), 4),
            confidence=1.0,
        )
        self.project.tempo_sections.sort(key=lambda item: item.start_time)
        self._refresh_sections()
        self._draw_beat_grid()

    def _delete_selected_marker(self) -> None:
        row = self.section_list.currentRow()
        if row < 0:
            return
        del self.project.tempo_sections[row]
        self._refresh_sections()
        self._draw_beat_grid()

    def _toggle_playback(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
            self.metronome_timer.stop()
            return
        start_seconds = self._cursor_seconds()
        self.player.setPosition(int(start_seconds * 1000))
        self.playhead_line.setPos(start_seconds)
        self.player.play()
        if self.metronome_enabled:
            self._prime_metronome()

    def _stop_playback(self) -> None:
        self.player.stop()
        self.metronome_timer.stop()
        self.playhead_line.setPos(0.0)

    def _sync_playhead(self, position_ms: int) -> None:
        self.playhead_line.setPos(position_ms / 1000.0)

    def _toggle_metronome(self, checked: bool) -> None:
        self.metronome_enabled = checked
        if checked and self.player.playbackState() == QMediaPlayer.PlayingState:
            self._prime_metronome()
        else:
            self.metronome_timer.stop()

    def _update_metronome_volume(self, value: int) -> None:
        self.metronome_sound.setVolume(max(0.0, min(1.0, value / 100.0)))
        self.settings.setValue("metronome_volume", value)

    def _prime_metronome(self) -> None:
        current_ms = self.player.position()
        beat_length_ms = max(int((60.0 / max(self.project.base_tempo, 1.0)) * 1000), 1)
        offset_ms = int(self.project.beat_offset * 1000)
        if current_ms <= offset_ms:
            next_ms = offset_ms
        else:
            beats_since_offset = max((current_ms - offset_ms) // beat_length_ms, 0)
            next_ms = offset_ms + (beats_since_offset + 1) * beat_length_ms
        self._metronome_next_ms = next_ms
        self.metronome_timer.start(20)

    def _advance_metronome(self) -> None:
        if self.player.playbackState() != QMediaPlayer.PlayingState:
            self.metronome_timer.stop()
            return
        current_ms = self.player.position()
        if current_ms >= self._metronome_next_ms:
            self.metronome_sound.play()
            self.statusBar().showMessage(f"Tick {self._metronome_next_ms / 1000.0:.3f}s", 120)
            beat_length_ms = max(int((60.0 / max(self.project.base_tempo, 1.0)) * 1000), 1)
            self._metronome_next_ms += beat_length_ms

    def _update_zoom(self, value: int) -> None:
        if self.project.duration <= 0:
            return
        width = max(self.project.duration / max(value / 5.0, 1.0), 2.0)
        center = self.playhead_line.value()
        start = max(center - width / 2.0, 0.0)
        end = min(start + width, self.project.duration)
        self.plot.setXRange(start, end, padding=0.01)

    def _export_song(self) -> None:
        if not self.project.audio_path:
            QMessageBox.warning(self, "No Audio", "Load an audio file first.")
            return
        export_dir = self.export_folder_edit.text().strip()
        if not export_dir:
            QMessageBox.information(self, "Export Folder Required", "Pick the game's ImportedSongs folder first.")
            return
        self._persist_settings()
        try:
            self._set_busy(True, "Exporting song...")
            output = export_project(self.project, export_dir)
        except Exception as exc:  # pragma: no cover - surfaced in UI
            QMessageBox.critical(self, "Export Failed", str(exc))
            return
        finally:
            self._set_busy(False)
        self.status_label.setText(f"Exported to {output}. Ready for the next song.")
        self.statusBar().showMessage(f"Exported to {output}", 5000)

    def _choose_import_folder(self) -> None:
        current_dir = self.import_folder_edit.text().strip() or str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Select Audio Import Folder", current_dir)
        if not folder:
            return
        self.import_folder_edit.setText(folder)
        self._persist_settings()

    def _choose_export_folder(self) -> None:
        current_dir = self.export_folder_edit.text().strip() or str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Select Dead as Disco ImportedSongs Folder", current_dir)
        if not folder:
            return
        self.export_folder_edit.setText(folder)
        self._persist_settings()

    def _persist_settings(self) -> None:
        self.settings.setValue("import_folder", self.import_folder_edit.text().strip())
        self.settings.setValue("export_folder", self.export_folder_edit.text().strip())

    def _load_settings(self) -> None:
        self.import_folder_edit.setText(str(self.settings.value("import_folder", "")))
        self.export_folder_edit.setText(str(self.settings.value("export_folder", "")))
        volume_value = int(self.settings.value("metronome_volume", 18))
        self.metronome_volume_slider.setValue(max(0, min(100, volume_value)))

    def _set_busy(self, busy: bool, message: str | None = None) -> None:
        if busy:
            self._busy_depth += 1
            self.busy_bar.setVisible(True)
            QApplication.setOverrideCursor(Qt.WaitCursor)
            if message:
                self.status_label.setText(message)
        else:
            self._busy_depth = max(0, self._busy_depth - 1)
            if self._busy_depth == 0:
                self.busy_bar.setVisible(False)
                QApplication.restoreOverrideCursor()
        self.import_button.setEnabled(self._busy_depth == 0)
        self.auto_button.setEnabled(self._busy_depth == 0)
        self.export_button.setEnabled(self._busy_depth == 0)
        QApplication.processEvents()

    def _ensure_metronome_sound(self) -> Path:
        sound_path = Path(tempfile.gettempdir()) / "dead_as_disco_soft_ding.wav"
        if sound_path.exists():
            return sound_path

        sample_rate = 22050
        duration_seconds = 0.16
        frequency = 1174.66
        frame_count = int(sample_rate * duration_seconds)
        with wave.open(str(sound_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            frames = bytearray()
            for index in range(frame_count):
                time_pos = index / sample_rate
                envelope = math.exp(-18.0 * time_pos)
                overtone = math.sin(2.0 * math.pi * frequency * time_pos)
                shimmer = math.sin(2.0 * math.pi * frequency * 2.0 * time_pos) * 0.25
                sample = (overtone + shimmer) * envelope * 0.22
                frames.extend(struct.pack("<h", int(max(-1.0, min(1.0, sample)) * 32767)))
            wav_file.writeframes(bytes(frames))
        return sound_path
