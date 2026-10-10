"""
PySide6 Desktop Application Main Window (Phase 12).
Provides an interactive GUI for video selection, calibration options,
real-time analysis progress, results inspection, and report export.
"""

from pathlib import Path
import subprocess
import sys
from typing import Optional

try:
    from PySide6.QtCore import Qt, QTimer, Slot
    from PySide6.QtGui import QColor, QFont, QIcon, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QDoubleSpinBox,
        QFileDialog,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QStackedWidget,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )
    HAVE_PYSIDE6 = True
except ImportError:
    HAVE_PYSIDE6 = False

from app.windows.analysis_worker import AnalysisWorker
from core.pipeline.orchestrator import AnalysisSessionResult
from core.video.metadata import get_video_metadata


DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1a1a1a;
    color: #e0e0e0;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #3d3d3d;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 14px;
    font-weight: bold;
    color: #00d4ff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}
QPushButton {
    background-color: #007acc;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #0098ff;
}
QPushButton:pressed {
    background-color: #005a9e;
}
QPushButton#startBtn {
    background-color: #00b060;
    font-size: 15px;
    padding: 12px 24px;
}
QPushButton#startBtn:hover {
    background-color: #00d674;
}
QProgressBar {
    border: 1px solid #444;
    border-radius: 5px;
    text-align: center;
    background-color: #2b2b2b;
    color: white;
    font-weight: bold;
    height: 24px;
}
QProgressBar::chunk {
    background-color: #00d4ff;
    border-radius: 4px;
}
QTableWidget {
    background-color: #242424;
    gridline-color: #383838;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
}
QHeaderView::section {
    background-color: #2d2d2d;
    color: #00d4ff;
    padding: 6px;
    border: 1px solid #383838;
    font-weight: bold;
}
QTabWidget::pane {
    border: 1px solid #3d3d3d;
}
QTabBar::tab {
    background-color: #2a2a2a;
    color: #bbb;
    padding: 8px 18px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background-color: #007acc;
    color: white;
    font-weight: bold;
}
"""


if HAVE_PYSIDE6:
    class MainWindow(QMainWindow):
        """Main PySide6 Application Window."""

        def __init__(self):
            super().__init__()
            self.setWindowTitle("Dashcam Speed Analyzer — Desktop Application")
            self.resize(1050, 750)
            self.setStyleSheet(DARK_STYLESHEET)

            self.selected_video: Optional[Path] = None
            self.current_result: Optional[AnalysisSessionResult] = None
            self.worker: Optional[AnalysisWorker] = None

            self.stacked_widget = QStackedWidget()
            self.setCentralWidget(self.stacked_widget)

            self._init_selection_page()
            self._init_progress_page()
            self._init_results_page()

            self.stacked_widget.setCurrentIndex(0)

        # -------------------------------------------------------------
        # 1. Video Selection & Configuration View
        # -------------------------------------------------------------
        def _init_selection_page(self) -> None:
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(40, 40, 40, 40)
            layout.setSpacing(20)

            # Title
            title_lbl = QLabel("DASHCAM SPEED ANALYZER")
            title_lbl.setStyleSheet("font-size: 26px; font-weight: bold; color: #00d4ff;")
            title_lbl.setAlignment(Qt.AlignCenter)
            layout.addWidget(title_lbl)

            subtitle_lbl = QLabel("Automatic Monocular Computer-Vision & Geometric Speed Estimation")
            subtitle_lbl.setStyleSheet("font-size: 14px; color: #aaa;")
            subtitle_lbl.setAlignment(Qt.AlignCenter)
            layout.addWidget(subtitle_lbl)

            layout.addSpacing(15)

            # Video selector box
            sel_group = QGroupBox("Video Input")
            sel_layout = QVBoxLayout(sel_group)

            btn_row = QHBoxLayout()
            self.choose_btn = QPushButton("Select Dashcam Video...")
            self.choose_btn.clicked.connect(self._on_select_video_clicked)
            btn_row.addWidget(self.choose_btn)

            self.file_path_lbl = QLabel("No video selected.")
            self.file_path_lbl.setStyleSheet("color: #bbb; font-style: italic;")
            btn_row.addWidget(self.file_path_lbl, stretch=1)
            sel_layout.addLayout(btn_row)

            self.meta_preview_lbl = QLabel("")
            self.meta_preview_lbl.setStyleSheet("color: #00ff88; font-family: monospace;")
            sel_layout.addWidget(self.meta_preview_lbl)
            layout.addWidget(sel_group)

            # Optional Calibration Group
            calib_group = QGroupBox("Optional Calibration Settings (Advanced)")
            calib_layout = QVBoxLayout(calib_group)

            self.manual_calib_cb = QCheckBox("Enable Manual FOV Override (Default: Automatic Geometry)")
            self.manual_calib_cb.toggled.connect(self._on_manual_calib_toggled)
            calib_layout.addWidget(self.manual_calib_cb)

            fov_row = QHBoxLayout()
            fov_row.addWidget(QLabel("Camera Horizontal FOV (°):"))
            self.fov_spin = QDoubleSpinBox()
            self.fov_spin.setRange(40.0, 180.0)
            self.fov_spin.setValue(110.0)
            self.fov_spin.setEnabled(False)
            fov_row.addWidget(self.fov_spin)

            fov_row.addWidget(QLabel("Camera Height (m):"))
            self.height_spin = QDoubleSpinBox()
            self.height_spin.setRange(0.5, 4.0)
            self.height_spin.setValue(1.35)
            fov_row.addWidget(self.height_spin)

            fov_row.addWidget(QLabel("Road Grid Square (m):"))
            self.grid_size_spin = QDoubleSpinBox()
            self.grid_size_spin.setRange(0.2, 5.0)
            self.grid_size_spin.setValue(1.0)
            self.grid_size_spin.setSingleStep(0.2)
            fov_row.addWidget(self.grid_size_spin)
            fov_row.addStretch()
            calib_layout.addLayout(fov_row)
            layout.addWidget(calib_group)

            layout.addStretch()

            # Start Analysis Button
            self.start_btn = QPushButton("START ANALYSIS", objectName="startBtn")
            self.start_btn.setEnabled(False)
            self.start_btn.clicked.connect(self._on_start_analysis_clicked)
            layout.addWidget(self.start_btn)

            self.stacked_widget.addWidget(page)

        # -------------------------------------------------------------
        # 2. Progress View
        # -------------------------------------------------------------
        def _init_progress_page(self) -> None:
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(60, 60, 60, 60)
            layout.setSpacing(25)

            prog_title = QLabel("Analyzing Dashcam Footage...")
            prog_title.setStyleSheet("font-size: 22px; font-weight: bold; color: #00d4ff;")
            prog_title.setAlignment(Qt.AlignCenter)
            layout.addWidget(prog_title)

            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            layout.addWidget(self.progress_bar)

            self.status_lbl = QLabel("Initializing pipeline orchestrator...")
            self.status_lbl.setStyleSheet("font-size: 14px; color: #ccc;")
            self.status_lbl.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.status_lbl)

            layout.addStretch()

            self.cancel_btn = QPushButton("Cancel Analysis")
            self.cancel_btn.setStyleSheet("background-color: #d32f2f;")
            self.cancel_btn.clicked.connect(self._on_cancel_clicked)
            layout.addWidget(self.cancel_btn, alignment=Qt.AlignCenter)

            self.stacked_widget.addWidget(page)

        # -------------------------------------------------------------
        # 3. Results View
        # -------------------------------------------------------------
        def _init_results_page(self) -> None:
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(20, 20, 20, 20)
            layout.setSpacing(15)

            # Summary Cards Row
            cards_row = QHBoxLayout()

            self.card_ego = QLabel("Ego Vehicle:\nAvg: -- km/h | Max: -- km/h")
            self.card_ego.setStyleSheet("background: #252525; padding: 10px; border-radius: 6px; border: 1px solid #333; font-weight: bold;")
            cards_row.addWidget(self.card_ego)

            self.card_vehs = QLabel("Surrounding Traffic:\nTotal Tracked: --")
            self.card_vehs.setStyleSheet("background: #252525; padding: 10px; border-radius: 6px; border: 1px solid #333; font-weight: bold;")
            cards_row.addWidget(self.card_vehs)

            self.card_calib = QLabel("Calibration:\nScale: --")
            self.card_calib.setStyleSheet("background: #252525; padding: 10px; border-radius: 6px; border: 1px solid #333; font-weight: bold;")
            cards_row.addWidget(self.card_calib)

            layout.addLayout(cards_row)

            # Tabs: Table, Plots, Report
            tabs = QTabWidget()

            # Tab 1: Vehicle Speed Table
            self.table = QTableWidget()
            self.table.setColumnCount(7)
            self.table.setHorizontalHeaderLabels([
                "Vehicle ID", "Class", "Speed (km/h)", "Relative Speed", "Direction", "Depth (m)", "Confidence"
            ])
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.table.setEditTriggers(QTableWidget.NoEditTriggers)  # Read-only results table
            self.table.setSelectionBehavior(QTableWidget.SelectRows)
            self.table.setAlternatingRowColors(True)
            tabs.addTab(self.table, "Vehicle Speed Table")

            # Tab 2: Speed Plots
            plot_tab = QWidget()
            plot_layout = QVBoxLayout(plot_tab)
            self.plot_img_lbl = QLabel("Speed profile plot will appear here.")
            self.plot_img_lbl.setAlignment(Qt.AlignCenter)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(self.plot_img_lbl)
            plot_layout.addWidget(scroll)
            tabs.addTab(plot_tab, "Speed Graphs")

            layout.addWidget(tabs, stretch=1)

            # Bottom Export Buttons
            action_row = QHBoxLayout()
            self.export_csv_btn = QPushButton("Export CSV...")
            self.export_csv_btn.clicked.connect(self._on_export_csv)
            action_row.addWidget(self.export_csv_btn)

            self.export_json_btn = QPushButton("Export JSON...")
            self.export_json_btn.clicked.connect(self._on_export_json)
            action_row.addWidget(self.export_json_btn)

            self.open_dir_btn = QPushButton("Open Outputs Folder")
            self.open_dir_btn.clicked.connect(self._on_open_outputs)
            action_row.addWidget(self.open_dir_btn)

            action_row.addStretch()

            self.new_analysis_btn = QPushButton("Analyze Another Video")
            self.new_analysis_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
            action_row.addWidget(self.new_analysis_btn)

            layout.addLayout(action_row)
            self.stacked_widget.addWidget(page)

        # -------------------------------------------------------------
        # UI Handlers & Slots
        # -------------------------------------------------------------
        @Slot()
        def _on_select_video_clicked(self) -> None:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Dashcam Video",
                str(Path.cwd() / "sample_data"),
                "Video Files (*.mp4 *.avi *.mov *.mkv)",
            )
            if file_path:
                self.selected_video = Path(file_path)
                self.file_path_lbl.setText(self.selected_video.name)
                try:
                    meta = get_video_metadata(self.selected_video)
                    self.meta_preview_lbl.setText(
                        f"✓ Loaded: {meta.width}x{meta.height} | {meta.fps:.1f} FPS | "
                        f"{meta.frame_count} frames | {meta.duration:.1f}s | Codec: {meta.codec}"
                    )
                    self.start_btn.setEnabled(True)
                except Exception as e:
                    self.meta_preview_lbl.setText(f"Error reading video: {e}")
                    self.start_btn.setEnabled(False)

        @Slot(bool)
        def _on_manual_calib_toggled(self, checked: bool) -> None:
            self.fov_spin.setEnabled(checked)

        @Slot()
        def _on_start_analysis_clicked(self) -> None:
            if not self.selected_video:
                return

            self.stacked_widget.setCurrentIndex(1)
            self.progress_bar.setValue(0)
            self.status_lbl.setText("Starting analysis pipeline...")

            user_fov = self.fov_spin.value() if self.manual_calib_cb.isChecked() else None
            camera_h = self.height_spin.value()
            grid_sq_m = self.grid_size_spin.value()

            self.worker = AnalysisWorker(
                video_path=self.selected_video,
                output_dir=Path("outputs"),
                user_fov=user_fov,
                camera_height=camera_h,
                square_size_meters=grid_sq_m,
                generate_video=True,
                sample_step=1,
                ego_only=True,
            )
            self.worker.sig_progress.connect(self._on_worker_progress)
            self.worker.sig_finished.connect(self._on_worker_finished)
            self.worker.sig_error.connect(self._on_worker_error)
            self.worker.start()

        @Slot(float, str)
        def _on_worker_progress(self, frac: float, msg: str) -> None:
            self.progress_bar.setValue(int(frac * 100))
            self.status_lbl.setText(msg)

        @Slot(object)
        def _on_worker_finished(self, result: AnalysisSessionResult) -> None:
            self.current_result = result
            self._populate_results(result)
            self.stacked_widget.setCurrentIndex(2)

        @Slot(str)
        def _on_worker_error(self, err_msg: str) -> None:
            QMessageBox.critical(self, "Analysis Error", f"An error occurred during video analysis:\n\n{err_msg}")
            self.stacked_widget.setCurrentIndex(0)

        @Slot()
        def _on_cancel_clicked(self) -> None:
            if self.worker:
                self.worker.cancel()
            self.stacked_widget.setCurrentIndex(0)

        def _populate_results(self, res: AnalysisSessionResult) -> None:
            # Cards
            avg_ego = res.summary["ego_vehicle"]["average_speed_kmh"]
            max_ego = res.summary["ego_vehicle"]["max_speed_kmh"]
            self.card_ego.setText(f"Ego Vehicle (Own Car):\nAvg: {avg_ego:.1f} km/h | Max: {max_ego:.1f} km/h")

            num_veh = res.summary.get("total_unique_vehicles", 0)
            avg_veh = res.summary["surrounding_vehicles"]["average_speed_kmh"]
            if res.vehicle_records:
                self.card_vehs.setText(f"Surrounding Traffic:\nTracked: {num_veh} vehicles | Avg: {avg_veh:.1f} km/h")
            else:
                self.card_vehs.setText("Surrounding Traffic:\nDormant (Ego-Only Analysis Mode)")

            scale_src = res.calibration.scale_source
            fov_val = res.calibration.fov_estimate.fov_h_deg
            fov_str = f"{fov_val:.1f}°" if fov_val else "Auto (uncalibrated)"
            self.card_calib.setText(f"Calibration:\nFOV: {fov_str} | Scale: {scale_src}")

            # Populate Table
            self.table.setRowCount(0)
            seen_tracks = {}
            for r in res.vehicle_records:
                tid = r["vehicle_id"]
                if tid not in seen_tracks:
                    seen_tracks[tid] = r

            for tid, r in seen_tracks.items():
                row = self.table.rowCount()
                self.table.insertRow(row)

                spd_str = f"{r['vehicle_speed']:.1f}" if r["vehicle_speed"] is not None else "Calculating"
                rel_str = f"{r['relative_speed']:.1f}" if r["relative_speed"] is not None else "--"
                depth_str = f"{r['depth_meters']:.1f}" if r.get("depth_meters") is not None else "--"
                conf_str = f"{r['confidence'] * 100:.0f}%"

                def make_cell(text: str) -> QTableWidgetItem:
                    item = QTableWidgetItem(text)
                    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    item.setTextAlignment(Qt.AlignCenter)
                    return item

                self.table.setItem(row, 0, make_cell(f"#{tid}"))
                self.table.setItem(row, 1, make_cell(str(r["vehicle_class"]).upper()))
                self.table.setItem(row, 2, make_cell(spd_str))
                self.table.setItem(row, 3, make_cell(rel_str))
                self.table.setItem(row, 4, make_cell(str(r["direction"]).replace("_", " ").title()))
                self.table.setItem(row, 5, make_cell(depth_str))
                self.table.setItem(row, 6, make_cell(conf_str))

            # Populate Plot
            if res.output_plot_path and res.output_plot_path.exists():
                pix = QPixmap(str(res.output_plot_path))
                self.plot_img_lbl.setPixmap(pix.scaledToWidth(850, Qt.SmoothTransformation))

        @Slot()
        def _on_export_csv(self) -> None:
            if not self.current_result or not self.current_result.output_csv_path:
                return
            dest, _ = QFileDialog.getSaveFileName(self, "Save CSV", "speed_results.csv", "CSV Files (*.csv)")
            if dest:
                import shutil
                shutil.copyfile(self.current_result.output_csv_path, dest)
                QMessageBox.information(self, "Export Successful", f"Saved CSV to:\n{dest}")

        @Slot()
        def _on_export_json(self) -> None:
            if not self.current_result or not self.current_result.output_json_path:
                return
            dest, _ = QFileDialog.getSaveFileName(self, "Save JSON", "speed_results.json", "JSON Files (*.json)")
            if dest:
                import shutil
                shutil.copyfile(self.current_result.output_json_path, dest)
                QMessageBox.information(self, "Export Successful", f"Saved JSON to:\n{dest}")

        @Slot()
        def _on_open_outputs(self) -> None:
            out_dir = Path("outputs").resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            if sys.platform == "win32":
                subprocess.run(["explorer", str(out_dir)])
            elif sys.platform == "darwin":
                subprocess.run(["open", str(out_dir)])
            else:
                subprocess.run(["xdg-open", str(out_dir)])

else:
    class MainWindow:
        pass
