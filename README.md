# 🚗 Driver Assistant - Vehicle Safety Alert System

A real-time driver assistance system that detects road hazards, lane departures, and traffic signals using computer vision and deep learning. Supports two deployment modes:

- **Distributed** (recommended for Pi + laptop): Raspberry Pi captures video and handles GPIO/display; a Windows laptop runs heavy AI inference over ZMQ (port **5555**).
- **Monolithic**: Full pipeline on one machine via `driver_assistant.py` (Windows webcam/video or onboard Pi with CSI camera).

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-orange.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Raspberry%20Pi-lightgrey.svg)

---

## 📋 Table of Contents

- [Features](#-features)
- [Deployment Modes](#-deployment-modes)
- [System Architecture](#-system-architecture)
- [Technologies Used](#-technologies-used)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
  - [Windows Setup (Monolithic)](#windows-setup-monolithic)
  - [Distributed Setup (Laptop + Pi)](#distributed-setup-laptop--pi)
  - [Raspberry Pi Setup (Monolithic)](#raspberry-pi-setup-monolithic)
- [Usage](#-usage)
  - [Distributed Mode (Laptop Brain + Pi Eyes)](#distributed-mode-laptop-brain--pi-eyes)
  - [Monolithic Mode](#monolithic-mode)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Documentation](#-documentation)

---

## ✨ Features

- **Object Detection**: Custom six-class ONNX head (vehicle, pedestrian, traffic light + green/red/yellow); classes listed in `config.yaml` under `yolo.classes`
- **Lane Detection**: Classical computer vision pipeline with polynomial curve fitting
- **Dynamic Danger Zone**: Trapezoidal collision detection zone that aligns with detected lanes
 - **Programmatic Beeps**: Tones generated in software (no external audio files required by default)
 - **Deployment helpers**: systemd service and `scripts/setup-pi.sh` for Raspberry Pi auto-start
- **Priority-Based Alerts**: Collision warnings take precedence over other alerts
- **Audio & Haptic Feedback**: Beep patterns via speakers and GPIO buzzer
- **Cross-Platform**: Runs on Windows (webcam/video) and Raspberry Pi (CSI camera)
- **Configurable**: All parameters tunable via `config.yaml`
- **Telemetry Logging**: JSON Lines format for performance analysis
- **Distributed Inference**: Offload YOLO and lane detection to a Windows laptop while the Pi handles camera, HDMI preview, and GPIO

---

## 🔄 Deployment Modes

| Mode | Entry points | Best for |
|------|--------------|----------|
| **Distributed** | `Laptop_Brain/laptop_server.py` + `Pi_Eyes/pi_client.py` | Pi camera + HDMI display with laptop doing AI on port 5555 |
| **Monolithic** | `driver_assistant.py` | Single-machine dev (Windows video/webcam) or full onboard Pi with systemd |

> Do **not** run the monolithic systemd service and distributed `pi_client.py` on the same Pi at the same time.

**Distributed quick start** — see [docs/DISTRIBUTED_START_GUIDE.md](docs/DISTRIBUTED_START_GUIDE.md) and [docs/DISTRIBUTED_OPERATIONS_GUIDE.md](docs/DISTRIBUTED_OPERATIONS_GUIDE.md).

---

## 🏗 System Architecture

### Monolithic (single machine)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DRIVER ASSISTANT SYSTEM                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────────────────────────────────────────┐  │
│  │   CAPTURE    │    │              PROCESSING PIPELINE                 │  │
│  │              │    │                                                  │  │
│  │ • CSI Camera │──▶│  Frame ──▶ Lane Detection ──▶ YOLO Detection    │  │
│  │ • Webcam     │    │              │                      │            │  │
│  │ • Video File │    │              ▼                      ▼            │  │
│  └──────────────┘    │      Lane Polynomials      Bounding Boxes        │  │
│                      │              │                      │            │  │
│                      │              └──────────┬───────────┘            │  │
│                      │                         ▼                        │  │
│                      │              ┌─────────────────────┐             │  │
│                      │              │  DANGER ZONE CHECK  │             │  │
│                      │              │  (Dynamic/Fixed)    │             │  │
│                      │              └──────────┬──────────┘             │  │
│                      │                         ▼                        │  │
│                      │              ┌─────────────────────┐             │  │
│                      │              │  ALERT DECISION     │             │  │
│                      │              │  ENGINE             │             │  │
│                      │              │  • Priority Queue   │             │  │
│                      │              │  • Cooldown Logic   │             │  │
│                      │              └──────────┬──────────┘             │  │
│                      └───────────────────────────────────────────────────┘ │
│                                                │                            │
│                      ┌─────────────────────────┼─────────────────────────┐  │
│                      │                         ▼                         │  │
│                      │  ┌─────────┐  ┌──────────────┐  ┌─────────────┐   │  │
│                      │  │ DISPLAY │  │ AUDIO ALERTS │  │ GPIO BUZZER │   │  │
│                      │  │ Overlay │  │   (Beeps)    │  │  (Pi Only)  │   │  │
│                      │  └─────────┘  └──────────────┘  └─────────────┘   │  │
│                      │                    OUTPUT LAYER                   │  │
│                      └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Distributed (laptop brain + Pi eyes)

```
┌──────────────────────┐         ZMQ (port 5555)         ┌──────────────────────────────┐
│   Raspberry Pi       │  JPEG frames ──────────────────▶ │   Windows Laptop             │
│   Pi_Eyes/           │                                  │   Laptop_Brain/              │
│   pi_client.py       │  ◀── detections, lanes, alerts ─ │   laptop_server.py           │
│                      │                                  │                              │
│ • CSI camera         │                                  │ • YOLO + lane detection      │
│ • HDMI preview       │                                  │ • Alert decision engine      │
│ • GPIO LED / buzzer  │                                  │ • Optional debug window      │
└──────────────────────┘                                  └──────────────────────────────┘
```

---

## 🛠 Technologies Used

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Object Detection** | YOLOv11s + ONNX Runtime | Detect traffic objects (CPU inference) |
| **Lane Detection** | OpenCV (Classical CV) | HSV filtering, Canny edges, Hough transform |
| **Configuration** | PyYAML | Load settings from `config.yaml` |
| **Audio** | Pygame / winsound | Play alert beep patterns |
| **Camera (Pi)** | picamera2 / libcamera | CSI camera interface |
| **GPIO (Pi)** | RPi.GPIO | Buzzer and status LED control |
| **Networking** | ZeroMQ (pyzmq) | Distributed frame + detection protocol |
| **Visualization** | OpenCV | Real-time overlay rendering |

---

## 📦 Prerequisites

### Common Requirements
- Python 3.9 or higher
- Git

### Windows
- Webcam (optional, can use video files)
- 4GB+ RAM recommended

### Raspberry Pi
- Raspberry Pi 4 (4GB+ RAM recommended)
- Raspberry Pi Camera Module v2 or v3
- MicroSD card (32GB+ recommended)
- Raspberry Pi OS (64-bit recommended)
- Passive buzzer (optional, connects to GPIO 18)

---

## 🚀 Installation

Clone the repository once on each machine that needs it:

```bash
git clone https://github.com/Jithmal-hettiarachchi/AUTONOMOUS-SYSTEM-FOR-VEHICULAR-SAFETY.git
cd Driver-Assistant-Distributed
```

### Windows Setup (Monolithic)

For local development with webcam or video files (root project venv):

1. **Create virtual environment**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```powershell
   python -c "import cv2; import onnxruntime; print('Ready!')"
   ```

### Distributed Setup (Laptop + Pi)

Use **separate virtual environments** in `Laptop_Brain/` and `Pi_Eyes/`.

#### Windows laptop (Brain)

```powershell
cd Laptop_Brain
python -m venv venv
.\venv\Scripts\activate
pip install -r ..\requirements_laptop.txt
pip install -r ..\requirements.txt
```

Place the ONNX model at `models/object.onnx` in the project root (shared by the laptop server).

#### Raspberry Pi (Eyes)

```bash
cd Pi_Eyes
python3 -m venv venv
source venv/bin/activate
pip install -r ../requirements-pi.txt
```

Ensure the camera is enabled (`sudo raspi-config` → Interface Options → Camera) and the `pi` user is in the `video` and `gpio` groups.

### Raspberry Pi Setup (Monolithic)

#### Option A: Automatic Setup (Recommended)

1. **Clone the repository** (if not already done — see clone command above).

2. **Run the setup script**
   ```bash
   chmod +x scripts/setup-pi.sh
   ./scripts/setup-pi.sh
   ```

3. **Reboot** (required for group permissions)
   ```bash
   sudo reboot
   ```

#### Option B: Manual Setup

1. **Update system**
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

2. **Install system dependencies**
   ```bash
   sudo apt install -y python3-pip python3-venv python3-opencv
   sudo apt install -y libatlas-base-dev libhdf5-dev pulseaudio
   ```

3. **Enable camera**
   ```bash
   sudo raspi-config
   # Navigate to: Interface Options → Camera → Enable
   # Reboot when prompted
   ```

4. **Create venv at project root**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

5. **Install dependencies**
   ```bash
   pip install -r requirements-pi.txt
   ```

6. **Add user to required groups**
   ```bash
   sudo usermod -aG video,gpio,audio $USER
   # Logout and login again for changes to take effect
   ```

7. **Wire the buzzer (optional)**
   ```
   Buzzer (+) ──▶ GPIO 18 (Pin 12)
   Buzzer (-) ──▶ GND (Pin 14)
   ```

---

## 🔄 Auto-Start on Boot (Raspberry Pi)

The system can be configured to start automatically when the Raspberry Pi boots.

### Enable Auto-Start

1. **Install the service file** (if not already done by `scripts/setup-pi.sh`)
   ```bash
   # Canonical unit lives in the repo (setup-pi.sh copies this file):
   sudo cp systemd/driver-assistant.service /etc/systemd/system/
   sudo systemctl daemon-reload
   ```

2. **Enable the service**
   ```bash
   sudo systemctl enable driver-assistant
   ```

3. **Start the service**
   ```bash
   sudo systemctl start driver-assistant
   ```

### Service Management Commands

| Command | Description |
|---------|-------------|
| `sudo systemctl start driver-assistant` | Start the service |
| `sudo systemctl stop driver-assistant` | Stop the service |
| `sudo systemctl restart driver-assistant` | Restart the service |
| `sudo systemctl status driver-assistant` | Check service status |
| `sudo systemctl enable driver-assistant` | Enable auto-start on boot |
| `sudo systemctl disable driver-assistant` | Disable auto-start |

### View Logs

```bash
# View application stdout/stderr (service file appends here)
tail -f ~/Driver-Assistant/logs/service.log

# Optional: systemd journal (may be sparse if logging goes to the file above)
journalctl -u driver-assistant -f

# View telemetry data
tail -f ~/Driver-Assistant/telemetry.jsonl
```

### Troubleshooting Auto-Start

If the service fails to start:

1. **Check status for errors**
   ```bash
   sudo systemctl status driver-assistant
   ```

2. **Check detailed logs**
   ```bash
   journalctl -u driver-assistant -n 50 --no-pager
   ```

3. **Common issues:**
   - Camera not enabled: Run `sudo raspi-config` and enable camera
   - Permission denied: Ensure user is in `video`, `gpio`, `audio` groups
   - Python not found: Check the path in the service file matches your setup

4. **Edit service file if needed**
   ```bash
   sudo nano /etc/systemd/system/driver-assistant.service
   # After editing:
   sudo systemctl daemon-reload
   sudo systemctl restart driver-assistant
   ```

---

## 🎮 Usage

### Distributed Mode (Laptop Brain + Pi Eyes)

**Start the laptop first**, then the Pi. Replace `<LAPTOP_IP>` with your Windows machine's LAN address (e.g. `192.168.1.100`).

#### Phase 1 — Windows laptop (Brain)

```cmd
cd "D:\AAA\Driver-Assistant-Distributed-main\Laptop_Brain"
venv\Scripts\activate
python laptop_server.py
```

Wait until the terminal shows **ONLINE** and listening on port **5555**. If Windows Firewall prompts you, click **Allow Access**.

#### Phase 2 — Raspberry Pi (Eyes)

```bash
cd /home/pi/Desktop/Driver-Assistant-Distributed-main/Pi_Eyes
source venv/bin/activate
sudo libcamerify venv/bin/python pi_client.py --server-ip <LAPTOP_IP>
```

#### Expected behavior

| Machine | What you should see |
|---------|---------------------|
| **Laptop** | Frame processing logs; optional AI debug window |
| **Pi** | Camera on; live HDMI preview with detection boxes and GPIO alerts |

#### Shutting down safely

Press **Ctrl+C** on the laptop terminal first. If the port stays locked (`ZMQError: Address already in use`), wait 10 seconds or run `taskkill /f /im python.exe` on Windows / `pkill -9 python` on the Pi.

Full run, shutdown, and troubleshooting steps:

- [docs/DISTRIBUTED_START_GUIDE.md](docs/DISTRIBUTED_START_GUIDE.md)
- [docs/DISTRIBUTED_OPERATIONS_GUIDE.md](docs/DISTRIBUTED_OPERATIONS_GUIDE.md)

---

### Monolithic Mode

#### Run with Video File (Windows/Pi)
```bash
python driver_assistant.py --source video --video-path videos/test.mp4 --display
```

#### Run with Webcam (Windows)
```bash
python driver_assistant.py --source webcam --camera-index 0 --display
```

#### Run with CSI Camera (Raspberry Pi)
```bash
python driver_assistant.py --source csi --display
```

#### Run Headless (No Display - Pi Production)
```bash
python driver_assistant.py --source csi
```

#### Command Line Arguments

Defined in `src/main.py` (`parse_args`). Entry point: `driver_assistant.py`.

| Argument | Description | Default |
|----------|-------------|---------|
| `--source` | `csi`, `webcam`, `video`, or `ip` | `csi` on Raspberry Pi, `webcam` on Windows |
| `--video-path` | Video file path (required if `source=video`) | — |
| `--ip-url` | Stream URL (required if `source=ip`) | — |
| `--camera-index` | Webcam index for `webcam` | `0` |
| `--display` / `--headless` | Show OpenCV UI vs no window | On Windows, display defaults on unless `--headless`; on Pi, headless unless `--display` |
| `--lane-debug` | Lane pipeline debug panel (sets `lane_detection.debug_view`) | off |
| `--config` | YAML config path | searches default `config.yaml` via loader |
| `--model` | Override ONNX path | from config |
| `--yolo-skip` | Override `yolo.frame_skip` | from config |
| `--confidence` | Override detection confidence | from config |
| `--log-file`, `--log-level` | Telemetry file and log level | `telemetry.jsonl`, `INFO` |
| `--resolution` | e.g. `640x480` | from config |
| `--disable-ir` / `--enable-ir` | IR sensor off (default) vs on | IR disabled unless `--enable-ir` |

Run `python driver_assistant.py --help` for the full list.

---

## ⚙ Configuration

All parameters are in `config.yaml`:

### Key Settings

```yaml
# Frame processing (see repository config.yaml for full file)
capture:
  resolution: [640, 480]
  target_fps: 15

yolo:
  confidence_threshold: 0.25
  frame_skip: 5                 # Run YOLO every N frames (matches default config.yaml)

danger_zone:
  top_left_y: 0.78              # Example; must match top_right_y
  top_right_y: 0.78

alerts:
  cooldown_ms: 300
  traffic_light_cooldown_ms: 10000
  alert_hold_frames: 5
```

### Tuning Tips

| Problem | Solution |
|---------|----------|
| Too many false collision alerts | Increase `danger_zone.top_left_y` (e.g., 0.75-0.85) |
| Missing distant hazards | Decrease `danger_zone.top_left_y` (e.g., 0.60-0.65) |
| Alerts too frequent | Increase `alerts.cooldown_ms` |
| Low FPS on Pi | Increase `yolo.frame_skip` to 5-6 |
| Lane detection unstable | Increase `lane_detection.ema_alpha` (0.4-0.5) |
| Traffic light alerts repeat too often | Increase `alerts.traffic_light_cooldown_ms` |
| Alerts disappear on skipped frames | Increase `alerts.alert_hold_frames` to persist display |

---

## 📁 Project Structure

```
Driver-Assistant-Distributed/
├── driver_assistant.py          # Monolithic entry point
├── config.yaml                  # Shared configuration
├── requirements.txt             # Base / Windows dependencies
├── requirements-pi.txt          # Pi dependencies (includes base)
├── requirements_laptop.txt      # Laptop brain (ZMQ + ONNX) extras
├── models/
│   └── object.onnx              # YOLOv11s ONNX model (project root)
├── videos/                      # Test videos
├── src/                         # Core pipeline (used by monolithic + laptop server)
│   ├── main.py                  # Monolithic application
│   ├── config.py
│   ├── alerts/                  # Decision engine, audio, GPIO buzzer
│   ├── capture/                 # CSI, webcam, IP, video adapters
│   ├── detection/               # YOLO ONNX inference
│   ├── environment/             # Night / environment helpers
│   ├── lane/                    # Classical CV lane pipeline
│   ├── display/                 # Overlay renderer
│   ├── gpio/                    # Status LEDs, passive buzzer PWM
│   ├── overtake/                # Advisory overtake assistant
│   ├── sensors/                 # LiDAR, IR (optional)
│   └── telemetry/               # JSONL logging and metrics
├── Laptop_Brain/
│   ├── laptop_server.py         # ZMQ server: inference on Windows (port 5555)
│   └── venv/                    # Laptop-only virtual environment
├── Pi_Eyes/
│   ├── pi_client.py             # ZMQ client: camera, HDMI, GPIO on Pi
│   └── venv/                    # Pi client virtual environment
├── docs/
│   ├── DISTRIBUTED_START_GUIDE.md
│   ├── DISTRIBUTED_OPERATIONS_GUIDE.md
│   ├── DEPLOYMENT.md
│   ├── TROUBLESHOOTING.md
│   └── WINDOWS_USAGE.md
├── scripts/
│   └── setup-pi.sh              # Pi venv + systemd install
└── systemd/
    ├── driver-assistant.service # Monolithic auto-start unit
    └── install-service.sh
```

Runtime log file: `telemetry.jsonl` (gitignored).

---

## 🔬 How It Works

### 1. Frame Capture
- Captures frames from CSI (Pi), webcam, video file, or IP stream (`--source ip` + `--ip-url`)
- Maintains target FPS with adaptive timing

### 2. Lane Detection (Every Frame)
```
Frame → ROI Crop → HSV Filter (white/yellow) → Gaussian Blur 
→ Canny Edges → Hough Lines → Polynomial Fit → EMA Smoothing
```

### 3. Object Detection (With Frame Skipping)
```
Frame → Resize to 640x640 → ONNX Inference → NMS 
→ Filter by confidence → Classify (pedestrian/vehicle/etc.)
```
- Runs every N frames (configurable) to maintain performance
- Caches results for skipped frames

### 4. Dynamic Danger Zone
- When **both lanes detected**: Trapezoid aligns with lane boundaries
- When **lanes not detected**: Falls back to fixed trapezoid from config
- Objects inside the zone trigger collision alerts

### 5. Alert Priority System

Aligned with `AlertType` in `src/alerts/types.py` (collision and lane departure are highest; traffic/stop signs depend on detections and decision engine).

| Priority | Alert Type | Trigger |
|----------|------------|---------|
| 1 (Highest) | `COLLISION_IMMINENT` | Obstacle in danger zone (+ LiDAR check when `lidar.required_for_collision`) |
| 2 | `LANE_DEPARTURE_LEFT` / `LANE_DEPARTURE_RIGHT` | Lane departure |
| 2 | `TRAFFIC_LIGHT_RED` | Red light (warning) |
| 3 | `TRAFFIC_LIGHT_YELLOW`, `STOP_SIGN` | Yellow light / stop sign |
| 4 | `TRAFFIC_LIGHT_GREEN` | Informational green |
| 5 | `SYSTEM_WARNING` | Lowest |

### 6. Output
- **Display**: Overlays showing lanes, danger zone, detections, alerts
- **Audio**: Distinct beep patterns for each alert type
- **GPIO Buzzer**: Physical buzzer feedback (Raspberry Pi only)
- **Telemetry**: JSON logs for analysis

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [docs/DISTRIBUTED_START_GUIDE.md](docs/DISTRIBUTED_START_GUIDE.md) | Step-by-step: start laptop server, then Pi client |
| [docs/DISTRIBUTED_OPERATIONS_GUIDE.md](docs/DISTRIBUTED_OPERATIONS_GUIDE.md) | Expected results, safe shutdown, ZMQ port fixes |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Full deployment reference |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | General troubleshooting |
| [docs/WINDOWS_USAGE.md](docs/WINDOWS_USAGE.md) | Windows monolithic usage examples |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Detailed system architecture |
| [ARCHITECTURE_SUMMARY.md](ARCHITECTURE_SUMMARY.md) | Condensed architecture for quick reference |

---

## 📄 License

This project is for educational purposes. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO
- [ONNX Runtime](https://onnxruntime.ai/) for efficient inference
- [OpenCV](https://opencv.org/) for computer vision

---

**⚠️ Disclaimer**: This is a prototype system for educational purposes. Do not rely on it as your sole safety system while driving. Always pay attention to the road and follow traffic laws.
