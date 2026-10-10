# Dashcam Speed Analyzer

A modular, scientifically grounded desktop application for estimating ego-vehicle and surrounding-vehicle speeds from heterogeneous dashcam footage using monocular computer vision and geometric estimation.

---

## 1. Project Objective

The goal of this project is to build an automated desktop application that accepts dashcam video from diverse cameras (without requiring users to manually enter camera parameters such as FOV) and estimates:
1. **Ego Vehicle Speed**: Forward velocity of the camera-carrying vehicle.
2. **Surrounding Vehicle Speeds**: Physical speed of other detected vehicles in the scene.
3. **Vehicle Direction & Trajectory**: Movement vectors (`same_direction`, `opposite_direction`, `crossing`, `stationary`).
4. **Reliability / Confidence Metrics**: Explicit uncertainty and error boundaries for every speed calculation.

### Fundamental Scientific Limitation: Monocular Scale Ambiguity
A single monocular camera video lacks absolute metric scale by default. Estimating real-world speed without prior calibration is an ill-posed inverse problem. Therefore:
* The system **never fabricates camera parameters or assumes arbitrary pixel displacement equals physical speed**.
* Results explicitly report **speed, confidence, estimation method, and scale source** (e.g., road geometry, known dimensions, vanishing points).
* When geometric constraints are insufficient, the system reports low confidence or marks scale as unavailable rather than reporting false precision.

---

## 2. Architecture: Decoupled Core Engine

The application strictly separates computer vision and geometric processing from the GUI:

```text
                 DESKTOP APPLICATION
                         │
                ┌────────┴────────┐
                │                 │
           PySide6 GUI        CORE ENGINE
                                  │
                ┌─────────────────┼─────────────────┐
                ↓                 ↓                 ↓
           Calibration       Ego Motion         Vehicles
                                  │                 │
                                  └────────┬────────┘
                                           ↓
                                    Speed Estimation
                                           ↓
                                    Results & Video
```

The core engine can be invoked seamlessly via:
* **PySide6 Desktop Application** (`python app/main.py` or `dashcam-gui`)
* **Command Line Interface (CLI)** (`python -m cli.analyze` or `dashcam-analyze`)
* Automated Python API / Batch testing pipelines

---

## 3. Directory Layout

```text
dashcam-speed-analyzer/
├── app/                        # PySide6 Desktop GUI
│   ├── main.py                 # Application launcher
│   ├── windows/
│   │   ├── main_window.py      # Main window (Selection, Progress, Results views)
│   │   └── analysis_worker.py  # QThread background worker
│   ├── widgets/
│   └── resources/
│
├── core/                       # Core CV & Estimation Engine (Independent of GUI)
│   ├── pipeline/               # Pipeline stages & orchestrator
│   │   ├── stages.py           # PipelineStage enumeration
│   │   └── orchestrator.py     # End-to-end 10-stage analysis orchestrator
│   ├── video/                  # Video I/O, validation, metadata, seeking
│   │   ├── exceptions.py       # Domain-specific exceptions
│   │   ├── validator.py        # File & container integrity validator
│   │   ├── metadata.py         # VideoMetadata dataclass & extraction
│   │   ├── reader.py           # VideoReader & read_frames generator
│   │   └── writer.py           # Context-managed VideoWriter
│   ├── calibration/            # Camera & scene calibration
│   │   ├── camera.py           # Intrinsics, Extrinsics, CameraModel
│   │   ├── vanishing_point.py  # RANSAC vanishing point estimation
│   │   ├── fov.py              # FOV & focal length recovery
│   │   ├── distortion.py       # Lens distortion model
│   │   └── road_geometry.py    # Horizon, pitch, roll, road ROI mask
│   ├── motion/                 # Feature tracking, VO, and ego speed
│   │   ├── feature_detection.py# Stationary feature detector (dynamic box exclusion)
│   │   ├── optical_flow.py     # Bidirectional Lucas-Kanade optical flow
│   │   ├── visual_odometry.py  # Essential matrix decomposition
│   │   └── ego_speed.py        # Scale resolution & kinematic Kalman filter
│   ├── detection/              # Vehicle detection & tracking
│   │   ├── vehicle_detector.py # YOLOv8 & synthetic vehicle detectors
│   │   └── tracker.py          # ByteTrack-inspired two-stage multi-object tracker
│   ├── depth/                  # Depth & road geometry
│   │   ├── road_plane.py       # Ground contact depth projection
│   │   └── monocular_depth.py  # Fused ground + dimensional depth estimator
│   ├── speed/                  # Speed estimation & confidence
│   │   ├── vehicle_speed.py    # 3D trajectory differentiation & ego fusion
│   │   └── confidence.py       # Multi-factor reliability engine
│   └── results/                # Export, annotation, and statistics
│       ├── video_annotator.py  # Telemetry HUD & visual overlays
│       ├── exporter.py         # CSV & JSON exporters
│       ├── statistics.py       # Matplotlib speed plots & run metrics
│       └── benchmark.py        # MAE, RMSE, MedAE validation metrics
│
├── cli/                        # Command Line Interface
│   └── analyze.py              # Video inspection & full pipeline CLI
│
├── sample_data/                # Synthetic test video generator & sample footage
│   ├── generate_synthetic.py
│   ├── synthetic_dashcam_30fps.mp4
│   └── synthetic_dashcam_60fps.mp4
│
├── outputs/                    # Default output directory
│   ├── synthetic_dashcam_30fps_annotated.mp4
│   ├── synthetic_dashcam_30fps_results.csv
│   ├── synthetic_dashcam_30fps_results.json
│   └── synthetic_dashcam_30fps_speed_graph.png
│
├── tests/                      # Automated test suite (45 pytest test cases)
│   ├── conftest.py
│   ├── test_validator.py
│   ├── test_metadata.py
│   ├── test_reader.py
│   ├── test_writer.py
│   ├── test_synthetic.py
│   ├── test_calibration.py
│   ├── test_motion.py
│   ├── test_detection_and_tracking.py
│   ├── test_depth_and_speed.py
│   ├── test_results_export.py
│   ├── test_benchmark.py
│   ├── test_orchestrator.py
│   └── test_end_to_end.py
│
├── docs/                       # Technical Documentation
│   └── architecture.md         # Comprehensive algorithmic & mathematical derivation
│
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Packaging configuration
├── .gitignore
├── .env.example
└── README.md
```

---

## 4. Completed Development Phases (1 to 15)

- [x] **Phase 1: Video Input & Metadata Extraction** — Container validation, metadata extraction, seeking, streaming, custom error hierarchy.
- [x] **Phase 2: Camera & Scene Calibration** — RANSAC vanishing point estimation, road horizon, pitch/roll, honest FOV status reporting.
- [x] **Phase 3: Stationary Feature Detection** — Shi-Tomasi corner detection with dynamic vehicle bounding box exclusion.
- [x] **Phase 4: Visual Odometry** — Bidirectional Lucas-Kanade optical flow, forward-backward error checking, Essential matrix egomotion.
- [x] **Phase 5: Ego Speed Estimation** — Road plane metric scale resolution, kinematic 1D Kalman filtering ($[x, v, a]^T$).
- [x] **Phase 6: Vehicle Detection** — YOLOv8 vehicle detection (car, truck, bus, motorcycle) with offline synthetic fallback.
- [x] **Phase 7: Multi-Object Vehicle Tracking** — ByteTrack two-stage association, persistent vehicle IDs, occlusion handling, trajectory history.
- [x] **Phase 8: Depth & Road Geometry** — Ground-contact tire projection fused with vehicle dimensional priors.
- [x] **Phase 9: Surrounding Vehicle Speed** — 3D ground trajectory differentiation fused with ego speed; direction classification (`same_direction`, `opposite_direction`, `crossing`, `stationary`).
- [x] **Phase 10: Video Annotation & Reports** — Telemetry HUD overlay, persistent label badges, CSV log export, JSON session dump, Matplotlib speed plots.
- [x] **Phase 11: CLI Integration** — Full pipeline CLI with `--inspect-only`, `--output-dir`, `--no-video`, `--step`, `--fov`, `--camera-height`, and `--json`.
- [x] **Phase 12: PySide6 Desktop GUI** — Modern dark-theme desktop application with background `QThread` processing, live progress bar, interactive results table, speed plot viewer, and CSV/JSON export.
- [x] **Phase 13: End-to-End Integration** — Unified `PipelineOrchestrator` linking all 10 stages.
- [x] **Phase 14: Validation & Benchmarking** — Ground-truth benchmarking module computing MAE, RMSE, MedAE, and MPE.
- [x] **Phase 15: Packaging & Deployment** — `pyproject.toml` packaging with console scripts `dashcam-analyze` and `dashcam-gui`.

---

## 5. Quickstart & Installation

### 1. Setup Virtual Environment
```bash
python -m venv .venv

# On Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# On Linux / macOS
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
pytest tests/ -v
```
*(All 45 unit and end-to-end integration tests execute in ~11s)*

---

## 6. How to Run the Application

### Option A: PySide6 Desktop Application (GUI)
Launch the desktop graphical interface:
```bash
python app/main.py
```
*Features:*
* **Select Video**: File dialog with instant metadata preview.
* **Calibration Options**: Optional manual FOV or camera height override.
* **Asynchronous Execution**: Responsive GUI thread using background `AnalysisWorker`.
* **Results Dashboard**: Top summary cards, vehicle speed table, embedded speed plots, and CSV/JSON export buttons.

### Option B: Command-Line Interface (CLI)
Run the full 10-stage pipeline on any video:
```bash
python -m cli.analyze sample_data/synthetic_dashcam_30fps.mp4 --output-dir outputs/
```

Options:
* Fast inspection only:
  ```bash
  python -m cli.analyze sample_data/synthetic_dashcam_30fps.mp4 --inspect-only
  ```
* Fast analysis skipping video rendering:
  ```bash
  python -m cli.analyze sample_data/synthetic_dashcam_30fps.mp4 --no-video
  ```
* Specify manual camera calibration:
  ```bash
  python -m cli.analyze my_dashcam.mp4 --fov 115.0 --camera-height 1.40
  ```
* Machine-readable JSON output:
  ```bash
  python -m cli.analyze sample_data/synthetic_dashcam_30fps.mp4 --json
  ```

---

## 7. Output Formats

### CSV Output (`outputs/*_results.csv`)
```csv
timestamp,ego_speed,vehicle_id,vehicle_class,vehicle_speed,relative_speed,direction,confidence
0.200,0.0,1,car,1.4,1.4,stationary,0.80
0.233,1.4,1,car,0.6,1.3,stationary,0.83
...
```

### JSON Output (`outputs/*_results.json`)
Contains input video properties, full camera & scene calibration findings, summary statistics (average/max ego speed, total vehicles, mean confidence), and per-frame tracking records.

### Speed Profile Graph (`outputs/*_speed_graph.png`)
Generates high-resolution dark-themed plots of:
1. Ego Vehicle Speed vs Time.
2. Individual Surrounding Vehicle Speeds vs Time.
