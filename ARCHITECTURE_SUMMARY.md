# Vehicle Safety Alert System - Architecture Summary

> **For LLM Context**: This is a condensed architecture summary optimized for AI assistants to understand the system quickly.

## System Purpose

Real-time driver assistant running on Raspberry Pi 4 that:
- Detects objects (vehicles, pedestrians, traffic lights) using YOLO
- Detects lane boundaries using classical CV
- Measures distance with TF-Luna LiDAR
- Evaluates collision risk in danger zone
- Provides audio/visual alerts
- Advisory overtake safety evaluation

## Core Workflow

```
Frame Capture → [YOLO Detection] + [Lane Detection] + [LiDAR] → Danger Zone Evaluation → Alert Decision → Output
```

### Processing Loop (main.py)

1. **Capture** frame from camera (CSI/IP/Webcam/Video)
2. **YOLO** inference every Nth frame (`yolo.frame_skip` in `config.yaml`, default `5`), with cache on skipped frames
3. **Lane** detection (every frame; typical single-frame cost is environment-dependent)
4. **LiDAR** distance reading (background thread)
5. **Danger Zone** - check if objects intersect trapezoid
6. **Alert Decision** - priority-based (collision > lane departure > traffic light)
7. **Outputs**:
   - Audio alert (pygame/winsound)
   - GPIO buzzer (GPIO18)
   - GPIO outputs (collision: GPIO22, braking: GPIO5)
   - Status LEDs (system: GPIO17, alert: GPIO27)
   - Display overlay (optional)
   - Telemetry logging (JSONL)

## Deployment: monolithic vs distributed

The system supports two deployment shapes. **Monolithic** runs the full pipeline on one machine via `driver_assistant.py` (CLI entry for `src.main`) with `config.yaml`—typical for an onboard Raspberry Pi with CSI camera and optional TF-Luna LiDAR; production auto-start commonly uses `systemd/driver-assistant.service`. **Distributed** splits roles: `Pi_Eyes/pi_client.py` on the Pi captures and sends JPEG frames over ZMQ REQ to `Laptop_Brain/laptop_server.py` on a PC (ZMQ REP), which performs YOLO, lane detection, and alert logic and returns detections for local overlay and GPIO. Use distributed mode to offload inference when the edge device is network-connected to a more capable host; do not run the monolithic systemd service for that same Pi preview path at the same time.

## Module Map

| Module | Path | Purpose |
|--------|------|---------|
| **Capture** | `src/capture/` | Frame acquisition from multiple sources |
| **Detection** | `src/detection/` | YOLOv11s ONNX inference |
| **Lane** | `src/lane/` | Classical CV lane detection pipeline |
| **Alerts** | `src/alerts/` | Audio playback, decision engine |
| **Sensors** | `src/sensors/` | TF-Luna LiDAR, IR sensor |
| **GPIO** | `src/gpio/` | Status LEDs, braking output |
| **Overtake** | `src/overtake/` | Advisory overtake assistant |
| **Display** | `src/display/` | Visualization overlay |
| **Telemetry** | `src/telemetry/` | JSON Lines logging |

## Camera Sources

| Source | Adapter | Platform |
|--------|---------|----------|
| CSI | `CSICameraAdapter` (picamera2) | Raspberry Pi |
| IP | `IPCameraAdapter` (MJPEG/RTSP) | Any |
| Webcam | `OpenCVCameraAdapter` | Windows/Linux |
| Video | `VideoFileAdapter` | Testing |

Factory pattern: `CameraFactory.create(source_type, config)`

## Detection Classes

```python
0: vehicle
1: pedestrian
2: trafficLight (generic)
3: trafficLight-Green
4: trafficLight-Red
5: trafficLight-Yellow
```

Model: `models/object.onnx` (YOLOv11s, 640×640 input)

## Lane Detection Pipeline

```
ROI Mask → HSV Filter → Gaussian Blur → Canny Edge → Hough Lines → Geometric Filter → Polynomial Fit → EMA Smoothing
```

- Slope-based classification: negative slope = left lane, positive = right
- `LaneResult`: contains `left_lane`, `right_lane`, `valid`, `partial`

## LiDAR Integration

- **Hardware**: TF-Luna UART at 115200 baud
- **Pins**: TX→GPIO15, RX→GPIO14
- **Purpose**: Collision confirmation (reduces false positives)
- **Config**: `lidar.collision_threshold_cm: 600`

```python
# Collision triggers when:
# 1. Object in danger zone (vision) AND
# 2. LiDAR distance < threshold
```

## Alert Priority

See `AlertType.priority` in `src/alerts/types.py`. Collision is highest; exact ordering includes red/yellow/green lights and stop sign.

| Tier | Examples | Condition |
|------|----------|-----------|
| 1 | `COLLISION_IMMINENT` | Danger zone (+ LiDAR when required) |
| 2 | Lane departure, red light | Per decision engine |
| 3–5 | Yellow/green lights, stop sign, system warning | Lower urgency |

## GPIO Pin Assignments

| Pin | Function |
|-----|----------|
| GPIO5 | Braking output (COLLISION_IMMINENT) |
| GPIO14/15 | LiDAR UART TX/RX |
| GPIO17 | System running LED |
| GPIO18 | Buzzer |
| GPIO22 | Collision detection output |
| GPIO27 | Alert active LED |

## Overtake Assistant (Advisory)

- **Advisory only** - NOT a safety system
- Evaluates: clearance zone, lane markings (broken/solid), vehicle presence
- Status: `DISABLED`, `UNSAFE`, `SAFE`
- Config: `traffic_side: "left"` (UK/Japan/India/Sri Lanka) or `"right"` (US/Europe)

## Configuration (config.yaml)

Key sections:
```yaml
capture:          # Resolution, FPS, timeout
yolo:             # Model path, confidence, frame_skip
lane_detection:   # ROI, HSV thresholds, EMA alpha
danger_zone:      # Trapezoid coordinates
lidar:            # Port, threshold, enabled
gpio_leds:        # Pin assignments
overtake_assistant: # Traffic side, zone width
alerts:           # Cooldowns, hold frames
```

## Entry Points

- **CLI**: `python driver_assistant.py --source csi`
- **Main loop**: `src/main.py` → `DriverAssistant.run()`
- **Setup**: 12 modules initialized in `setup()` method

## Test Suite

```bash
pytest --tb=short
```

## Key Data Structures

```python
@dataclass
class Detection:
    label: DetectionLabel
    confidence: float
    bbox: Tuple[x1, y1, x2, y2]

@dataclass
class LaneResult:
    left_lane: Optional[LanePolynomial]
    right_lane: Optional[LanePolynomial]
    valid: bool
    partial: bool

@dataclass
class AlertEvent:
    alert_type: AlertType
    timestamp: float
    confidence: float = 1.0
```

## Telemetry Analysis

```bash
python tools/analyze_telemetry.py telemetry.jsonl --output-dir reports/
```

Generates: FPS graph, latency distribution, detection reliability, alert breakdown.

## Platform Detection

```python
from src.utils.platform import is_raspberry_pi, is_windows
# Auto-selects: picamera2 vs OpenCV, GPIO.BCM vs mock
```

---

*This summary covers the essential architecture. For full details, see [ARCHITECTURE.md](ARCHITECTURE.md).*
