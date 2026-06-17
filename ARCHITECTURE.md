# Vehicle Safety Alert System — System Architecture Document

**Version:** 2.0  
**Date:** 2026-02-17  
**Classification:** Safety-Critical Alerting Software (Proof-of-Concept)  
**Target Platform:** Raspberry Pi 4 (8GB RAM) / Windows 10/11

> **See also:** [ARCHITECTURE_SUMMARY.md](ARCHITECTURE_SUMMARY.md) for a condensed overview optimized for LLM context.

---

## Implementation Notes (recent updates)

### Version 2.0 Changes (February 2026)
- **TF-Luna LiDAR Integration**: Added UART-based LiDAR sensor for collision confirmation, reducing false positives from vision-only detection.
- **Overtake Assistant**: Advisory-only module for evaluating overtake safety (clearance zone, lane marking detection, vehicle presence).
- **IP Camera Support**: Network stream capture (MJPEG/RTSP) with auto-reconnection and latency minimization.
- **GPIO Braking Output**: GPIO5 output pin triggered on COLLISION_IMMINENT alert (demonstration only).
- **GPIO Status LEDs**: System running (GPIO17), Alert active (GPIO27), Collision output (GPIO22).
- **CSI Camera Improvements**: Better error handling, increased buffer count, warmup time, hardware alignment.
- **Telemetry Analysis Tool**: Standalone tool (`tools/analyze_telemetry.py`) for analyzing telemetry files and generating performance graphs.
- **Traffic Side Configuration**: Support for left-hand (UK, Japan, India, Sri Lanka) and right-hand traffic (US, Europe).
- **Camera Diagnostic Tool**: `tools/diagnose_camera.py` for troubleshooting camera issues.

### Version 1.0 Features
- Dynamic lane-aligned `DangerZone` implemented: when both lanes are detected the danger polygon is updated from lane polynomials; otherwise the configured trapezoid is used as a fallback.
- Traffic-light alerts simplified to a single detection alert (implementation uses a single traffic-light detected type with a separate `alerts.traffic_light_cooldown_ms` configuration to avoid repeated notifications).
- Alert display persistence: `alerts.alert_hold_frames` controls how many frames a display alert remains visible across YOLO-skip frames.
- Programmatic audio fallback: tones can be generated in software (pygame on Linux, `winsound` on Windows) so external WAV files are optional; GPIO buzzer remains available as reinforcement on Raspberry Pi.
- Deployment helpers: `scripts/setup-pi.sh` installs `systemd/driver-assistant.service`; optional `systemd/install-service.sh` for manual installs.
- Telemetry rounding: telemetry numeric fields (latencies, temperatures) are rounded for readability in JSONL output.


---

## 1. System Overview

### 1.1 System Description

The Vehicle Safety Alert System is a real-time, on-device visual perception and alerting system designed for forward-facing vehicle camera deployment. The system processes video frames to detect traffic lights, pedestrians, vehicles, and lane boundaries, issuing prioritized audio alerts when hazardous conditions are identified.

The system operates under strict latency, determinism, and fault-tolerance constraints, treating all alert pathways as safety-critical. All inference and processing occurs locally on CPU-only hardware with no cloud dependency.

### 1.2 System Boundaries

**In Scope:**
- Real-time frame acquisition from CSI camera (Raspberry Pi), IP camera (network), webcam, or video file
- YOLOv11s-based object detection for 6 classes (traffic lights ×3, pedestrians, vehicles)
- Classical computer vision lane detection with temporal stabilization
- **TF-Luna LiDAR distance measurement for collision confirmation**
- Trapezoidal danger zone collision risk evaluation
- **Overtake safety advisory (clearance zone evaluation)**
- Priority-based audio alert dispatch (3.5mm audio + GPIO buzzer)
- **GPIO outputs for collision (GPIO22) and braking (GPIO5)**
- Optional graphical overlay rendering
- Performance telemetry and JSON Lines logging
- **Telemetry analysis tool with graph generation**
- Optional IR distance sensor integration

**Out of Scope (Non-Goals):**
- Vehicle control or actuation of any kind
- Night-time, low-light, HDR, or IR-based visual operation
- Automotive safety certification (ISO 26262, ASIL)
- Cloud connectivity or remote processing
- Deep learning-based lane detection
- Multi-camera or surround-view configurations
- GPS, IMU, or vehicle CAN bus integration
- Real-time video streaming or recording

### 1.3 Safety Intent

This system is classified as **advisory alerting software**. It:
- Does NOT control the vehicle
- Does NOT replace driver attention
- Provides supplementary warnings only
- Degrades safely and audibly on failure

---

## 2. High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              MAIN PROCESSING LOOP                               │
│                          (Single-threaded, Deterministic)                       │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                             FRAME ACQUISITION                                   │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌──────────────────┐    │
│  │  CSI Camera Adapter │    │  Webcam Adapter     │    │ Video File       │    │
│  │  (libcamera/Pi)     │    │  (OpenCV/Windows)   │    │ Adapter          │    │
│  └─────────────────────┘    └─────────────────────┘    └──────────────────┘    │
│                    └──────────────────┼──────────────────┘                      │
│                                       ▼                                         │
│                          ┌────────────────────────┐                             │
│                          │  Frame (BGR, 640×480)  │                             │
│                          └────────────────────────┘                             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
┌───────────────────────────────────┐   ┌───────────────────────────────────────┐
│      YOLO OBJECT DETECTION        │   │         LANE DETECTION                │
│  (Scheduled, Frame-Skipped)       │   │     (Every Frame, CV-Based)           │
│                                   │   │                                       │
│  ┌─────────────────────────────┐  │   │  ┌─────────────────────────────────┐  │
│  │  YOLOv11s ONNX Model        │  │   │  │  ROI Masking (Trapezoid)        │  │
│  │  (640×640, CPU EP)          │  │   │  └─────────────────────────────────┘  │
│  └─────────────────────────────┘  │   │                  │                    │
│               │                   │   │                  ▼                    │
│               ▼                   │   │  ┌─────────────────────────────────┐  │
│  ┌─────────────────────────────┐  │   │  │  HSV Color Segmentation         │  │
│  │  Detection Cache (TTL)      │  │   │  │  (White + Yellow Isolation)     │  │
│  │  Max Age: 400ms             │  │   │  └─────────────────────────────────┘  │
│  └─────────────────────────────┘  │   │                  │                    │
│               │                   │   │                  ▼                    │
│               ▼                   │   │  ┌─────────────────────────────────┐  │
│  ┌─────────────────────────────┐  │   │  │  Gaussian Blur + Canny Edge     │  │
│  │  Detections:                │  │   │  └─────────────────────────────────┘  │
│  │  - traffic_light_red        │  │   │                  │                    │
│  │  - traffic_light_yellow     │  │   │                  ▼                    │
│  │  - traffic_light_green      │  │   │  ┌─────────────────────────────────┐  │
│  │  - pedestrian               │  │   │  │  Probabilistic Hough Transform  │  │
│  │  - vehicle                  │  │   │  └─────────────────────────────────┘  │
│  └─────────────────────────────┘  │   │                  │                    │
│                                   │   │                  ▼                    │
│                                   │   │  ┌─────────────────────────────────┐  │
│                                   │   │  │  Geometric Filtering            │  │
│                                   │   │  │  (Slope, Length, Position)      │  │
│                                   │   │  └─────────────────────────────────┘  │
│                                   │   │                  │                    │
│                                   │   │                  ▼                    │
│                                   │   │  ┌─────────────────────────────────┐  │
│                                   │   │  │  Polynomial Fitting (2nd Order) │  │
│                                   │   │  └─────────────────────────────────┘  │
│                                   │   │                  │                    │
│                                   │   │                  ▼                    │
│                                   │   │  ┌─────────────────────────────────┐  │
│                                   │   │  │  Temporal Stabilization (EMA)   │  │
│                                   │   │  └─────────────────────────────────┘  │
│                                   │   │                  │                    │
│                                   │   │                  ▼                    │
│                                   │   │  ┌─────────────────────────────────┐  │
│                                   │   │  │  Left Lane / Right Lane / None  │  │
│                                   │   │  └─────────────────────────────────┘  │
└───────────────────────────────────┘   └───────────────────────────────────────┘
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        TRAPEZOIDAL DANGER ZONE EVALUATION                       │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         Danger Zone (Fixed Trapezoid)                    │   │
│  │                              ┌─────────┐                                 │   │
│  │                             /           \                                │   │
│  │                            /             \                               │   │
│  │                           /               \                              │   │
│  │                          /                 \                             │   │
│  │                         ┌───────────────────┐                            │   │
│  │                                                                          │   │
│  │  • Collision Risk = BBox(pedestrian|vehicle) ∩ Trapezoid ≠ ∅            │   │
│  │  • Lane Departure = Vehicle Center outside Lane Boundaries               │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           ALERT DECISION ENGINE                                 │
│                                                                                 │
│     Priority 1 (Highest): COLLISION_RISK                                        │
│     Priority 2:           LANE_DEPARTURE                                        │
│     Priority 3 (Lowest):  TRAFFIC_LIGHT_STATE                                   │
│                                                                                 │
│     Rule: Only ONE alert active at a time. Higher priority preempts lower.      │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
┌─────────────────────────┐ ┌─────────────────────┐ ┌─────────────────────────────┐
│   AUDIO ALERT OUTPUT    │ │  GPIO BUZZER OUTPUT │ │  OPTIONAL DISPLAY OUTPUT    │
│   (3.5mm Analog)        │ │  (Fallback/Reinf.)  │ │  (Non-blocking Render)      │
│                         │ │                     │ │                             │
│  - Non-blocking playback│ │  - BCM Pin Config   │ │  - BBox Overlays            │
│  - Priority preemption  │ │  - Pattern-based    │ │  - Lane Overlays            │
│  - Failure isolation    │ │  - Failure tolerant │ │  - Danger Zone Viz          │
└─────────────────────────┘ └─────────────────────┘ └─────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          TELEMETRY & LOGGING                                    │
│                                                                                 │
│  Output: JSON Lines (.jsonl)                                                    │
│  Fields: capture_fps, lane_latency_ms, yolo_latency_ms, decision_latency_ms,    │
│          alert_type, alert_latency_ms, cpu_temperature_c, dropped_frames        │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     OPTIONAL: IR DISTANCE SENSOR                                │
│                     (Final Development Stage)                                   │
│                                                                                 │
│  - Supplementary proximity input                                                │
│  - Vision-based alerts take priority                                            │
│  - Fully disableable via CLI                                                    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Module Decomposition

### 3.1 Frame Capture Module

**Purpose:**  
Acquire raw video frames from the configured source at a target rate of ≥15 FPS.

**Inputs:**
- Configuration: source type (CSI, webcam, video file), resolution (default 640×480)
- Platform-specific camera handle

**Outputs:**
- `Frame` object: BGR uint8 numpy array, shape (480, 640, 3)
- Metadata: timestamp (monotonic), frame sequence number, capture latency

**Failure Behavior:**
| Failure Mode | Response |
|--------------|----------|
| Camera disconnection | Attempt reconnect (max 3 retries, 500ms interval) |
| Reconnect failure | Emit audible warning, continue with stale frame flag |
| Frame timeout | Return `None`, increment `dropped_frames` counter |
| Corrupted frame | Discard, increment `dropped_frames` counter |

**Determinism Guarantees:**
- Fixed polling interval (target: 66.67ms for 15 FPS)
- Non-blocking read with configurable timeout (default: 100ms)
- Frame timestamp assigned immediately upon capture

**Interface:**
```python
class FrameCaptureModule(ABC):
    @abstractmethod
    def initialize(self, config: CaptureConfig) -> bool: ...
    
    @abstractmethod
    def capture(self) -> Optional[Frame]: ...
    
    @abstractmethod
    def release(self) -> None: ...
    
    @abstractmethod
    def is_healthy(self) -> bool: ...
```

---

### 3.2 YOLO Object Detection Module

**Purpose:**  
Detect traffic lights (3 states), pedestrians, and vehicles using YOLOv11s ONNX model with CPU inference.

**Inputs:**
- Frame: BGR uint8 (640×480)
- Frame skip flag (from scheduler)

**Outputs:**
- List of `Detection` objects, or cached detections if skipped
- Inference latency (ms)

**Detection Schema:**
```python
@dataclass
class Detection:
    label: str           # One of: traffic_light_red, traffic_light_yellow,
                         # traffic_light_green, pedestrian, vehicle
    confidence: float    # Range: [0.0, 1.0]
    bbox: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    timestamp: float     # Monotonic timestamp of inference
```

**Failure Behavior:**
| Failure Mode | Response |
|--------------|----------|
| Model file not found | **Abort startup** (hard failure) |
| Model load failure | **Abort startup** (hard failure) |
| ONNX Runtime error | Log error, return empty detections, continue |
| Inference timeout (>500ms) | Log warning, return cached detections |

**Determinism Guarantees:**
- Model loaded exactly once at startup
- Fixed input resolution: 640×640 (letterbox padding applied)
- Deterministic NMS parameters (IoU threshold: 0.45, confidence threshold: 0.25)
- Frame skip interval configurable (default: skip 2 frames, infer on 3rd)
- Cache TTL enforced: detections older than 400ms marked stale

**Preprocessing Pipeline:**
1. Resize frame from 640×480 to 640×640 with letterbox padding
2. Convert BGR → RGB
3. Normalize to [0, 1] float32
4. Transpose to NCHW format: (1, 3, 640, 640)

**Postprocessing Pipeline:**
1. Extract raw outputs from ONNX model
2. Apply confidence thresholding
3. Apply class-specific NMS
4. Scale bounding boxes back to original frame coordinates
5. Filter to permitted class set only

---

### 3.3 Lane Detection Module (Hybrid CV)

**Purpose:**  
Detect left and right lane boundaries using classical computer vision techniques with temporal stabilization. Explicitly reject non-lane road markings.

**Inputs:**
- Frame: BGR uint8 (640×480)
- Previous lane state (for temporal filtering)

**Outputs:**
- `LaneResult` containing left lane, right lane (each optional)
- Processing latency (ms)
- Confidence/validity flags

**Lane Output Schema:**
```python
@dataclass
class LanePolynomial:
    coefficients: Tuple[float, float, float]  # a, b, c for ax² + bx + c
    y_range: Tuple[int, int]                  # (y_start, y_end)
    confidence: float                          # [0.0, 1.0]

@dataclass
class LaneResult:
    left_lane: Optional[LanePolynomial]
    right_lane: Optional[LanePolynomial]
    valid: bool
    timestamp: float
```

**Processing Pipeline (Mandatory Order):**

| Stage | Operation | Parameters |
|-------|-----------|------------|
| 1 | ROI Masking | Trapezoid: bottom 50% of frame |
| 2 | HSV Conversion | BGR → HSV |
| 3 | White Mask | H: [0,180], S: [0,30], V: [200,255] |
| 4 | Yellow Mask | H: [15,35], S: [80,255], V: [150,255] |
| 5 | Combined Mask | white_mask OR yellow_mask |
| 6 | Gaussian Blur | kernel: (5,5), sigma: 0 |
| 7 | Canny Edge | low: 50, high: 150 |
| 8 | Hough Transform | rho: 2, theta: π/180, threshold: 50, minLength: 40, maxGap: 100 |
| 9 | Geometric Filter | slope ∈ [0.5, 2.0], length > 40px |
| 10 | Left/Right Split | x < center → left candidates, x > center → right candidates |
| 11 | Polynomial Fit | np.polyfit degree=2, RANSAC optional |
| 12 | Temporal EMA | α = 0.3, reuse last valid for up to 5 frames |

**Non-Lane Marking Rejection Strategy:**

| Marking Type | Rejection Method |
|--------------|------------------|
| Zebra crossings | Horizontal line filter (slope < 0.3), cluster density check |
| Stop lines | Horizontal orientation rejection |
| Directional arrows | Geometric consistency (arrows have multiple orientations) |
| Road text | Connected component size filter, aspect ratio check |

**Failure Behavior:**
| Failure Mode | Response |
|--------------|----------|
| No lines detected | Return `LaneResult(valid=False)`, reuse last valid up to 5 frames |
| Single lane only | Return partial result, flag for decision engine |
| Temporal timeout (>5 frames) | Clear lane state, disable lane departure alerts |
| Processing exception | Log error, return invalid result, continue pipeline |

**Determinism Guarantees:**
- Fixed ROI coordinates (computed once from resolution)
- Fixed HSV thresholds (configurable via config file, not runtime)
- Deterministic Hough parameters
- EMA alpha fixed at startup
- No random sampling in RANSAC (fixed seed or deterministic alternative)

---

### 3.4 Trapezoidal Danger Zone Module

**Purpose:**  
Define the forward collision corridor and evaluate whether detected objects pose collision risk.

**Inputs:**
- List of `Detection` objects (pedestrians, vehicles)
- Frame dimensions

**Outputs:**
- List of objects flagged as collision risk
- Danger zone polygon (for visualization)

**Danger Zone Definition:**
```python
@dataclass
class DangerZone:
    # Coordinates for 640×480 frame
    top_left: Tuple[int, int]     = (240, 240)   # 37.5% from left, 50% from top
    top_right: Tuple[int, int]    = (400, 240)   # 62.5% from left, 50% from top  
    bottom_left: Tuple[int, int]  = (80, 480)    # 12.5% from left, bottom
    bottom_right: Tuple[int, int] = (560, 480)   # 87.5% from left, bottom
```

**Collision Evaluation:**
```python
def is_collision_risk(detection: Detection, zone: DangerZone) -> bool:
    """Returns True if bounding box intersects danger zone polygon."""
    bbox_polygon = box_to_polygon(detection.bbox)
    zone_polygon = Polygon([zone.top_left, zone.top_right, 
                           zone.bottom_right, zone.bottom_left])
    return bbox_polygon.intersects(zone_polygon)
```

**Failure Behavior:**
- No detections: Return empty collision list (not a failure)
- Invalid bbox coordinates: Skip detection, log warning
- Zone computation error: Use default zone, log error

**Determinism Guarantees:**
- Zone coordinates fixed at startup based on resolution
- Intersection calculation uses exact polygon math (Shapely or equivalent)
- Consistent ordering of collision risk evaluation

---

### 3.5 Alert Decision Engine

**Purpose:**  
Evaluate all hazard inputs and determine the single highest-priority alert to dispatch.

**Inputs:**
- Collision risk list (from danger zone module)
- Lane departure status (from lane detection + vehicle position)
- Traffic light detections (red, yellow, green)
- IR distance sensor reading (optional, when enabled)
- Current active alert state

**Outputs:**
- `AlertEvent` or `None`
- Alert state transition log entry

**Priority Table (Immutable):**
| Priority | Alert Type | Trigger Condition |
|----------|------------|-------------------|
| 1 (Highest) | `COLLISION_IMMINENT` | Pedestrian or vehicle in danger zone |
| 2 | `LANE_DEPARTURE_LEFT` | Vehicle center crosses left lane boundary |
| 2 | `LANE_DEPARTURE_RIGHT` | Vehicle center crosses right lane boundary |
| 3 | `TRAFFIC_LIGHT_RED` | Red traffic light detected with confidence > 0.5 |
| 3 | `TRAFFIC_LIGHT_YELLOW` | Yellow traffic light detected with confidence > 0.5 |

**Alert Event Schema:**
```python
@dataclass
class AlertEvent:
    alert_type: AlertType          # Enum value
    priority: int                  # 1-3
    timestamp: float               # Monotonic time
    trigger_source: str            # Module that triggered
    confidence: float              # Aggregated confidence
    suppressed_alerts: List[str]   # Lower-priority alerts that were suppressed
```

**State Machine:**
```
                    ┌─────────────┐
                    │    IDLE     │
                    └──────┬──────┘
                           │ hazard detected
                           ▼
                    ┌─────────────┐
         ┌─────────│   ACTIVE    │─────────┐
         │         └──────┬──────┘         │
         │ higher priority│                │ hazard cleared
         │ alert detected │                │
         ▼                │                ▼
    ┌──────────┐          │         ┌─────────────┐
    │ PREEMPT  │──────────┘         │  COOLDOWN   │
    └──────────┘                    └──────┬──────┘
                                           │ cooldown expired (300ms)
                                           ▼
                                    ┌─────────────┐
                                    │    IDLE     │
                                    └─────────────┘
```

**Failure Behavior:**
- All inputs missing/invalid: Remain in IDLE, no alert
- Conflicting priorities: Log warning, select first in priority order
- State machine error: Reset to IDLE, log error

**Determinism Guarantees:**
- Priority order is fixed and immutable
- State transitions occur at defined frame boundaries
- Cooldown timer uses monotonic clock
- No probabilistic alert suppression

---

### 3.6 Audio Alert System

**Purpose:**  
Dispatch audio alerts via 3.5mm analog output (primary) and GPIO buzzer (secondary/fallback).

**Inputs:**
- `AlertEvent` from decision engine
- Audio configuration (volume, sound files)

**Outputs:**
- Audio playback initiated (non-blocking)
- GPIO buzzer pattern activated
- Playback status for telemetry

**Audio File Mapping:**
| Alert Type | Audio File | Duration (ms) | GPIO Pattern |
|------------|------------|---------------|--------------|
| `COLLISION_IMMINENT` | `collision.wav` | 500 | Continuous |
| `LANE_DEPARTURE_LEFT` | `lane_left.wav` | 400 | 3 short beeps |
| `LANE_DEPARTURE_RIGHT` | `lane_right.wav` | 400 | 3 short beeps |
| `TRAFFIC_LIGHT_RED` | `red_light.wav` | 600 | 2 long beeps |
| `TRAFFIC_LIGHT_YELLOW` | `yellow_light.wav` | 400 | 1 long beep |

**Playback Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                    Audio Alert Manager                       │
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │ Alert Queue │───▶│ Priority    │───▶│ Non-blocking    │  │
│  │ (size: 1)   │    │ Comparator  │    │ Playback Thread │  │
│  └─────────────┘    └─────────────┘    └─────────────────┘  │
│                            │                    │           │
│                            │ preempt signal     │           │
│                            ▼                    ▼           │
│                     ┌─────────────┐    ┌─────────────────┐  │
│                     │ GPIO Buzzer │    │ Audio Backend   │  │
│                     │ Controller  │    │ (pygame/alsa)   │  │
│                     └─────────────┘    └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Failure Behavior:**
| Failure Mode | Response |
|--------------|----------|
| Audio file missing | Log error, use GPIO buzzer only |
| Audio device unavailable | Log warning, use GPIO buzzer only |
| GPIO unavailable (Windows) | Graceful skip, audio-only mode |
| Playback thread crash | Restart thread, log error, continue |

**Determinism Guarantees:**
- Alert queue size fixed at 1 (immediate preemption)
- Playback initiated within 10ms of alert event
- GPIO patterns are fixed duration, not audio-synced
- Audio backend initialized once at startup

---

### 3.7 Telemetry & Logging Module

**Purpose:**  
Record performance metrics and system state in JSON Lines format for offline analysis.

**Inputs:**
- Timing measurements from all pipeline stages
- Alert events
- System health indicators (CPU temp, dropped frames)

**Outputs:**
- Append-only `.jsonl` log file
- Optional stdout/stderr logging

**Telemetry Record Schema:**
```json
{
  "timestamp": "2026-01-28T14:30:00.123456Z",
  "frame_seq": 12345,
  "capture_fps": 15.2,
  "capture_latency_ms": 12.5,
  "lane_latency_ms": 8.3,
  "yolo_latency_ms": 95.2,
  "yolo_skipped": false,
  "decision_latency_ms": 0.5,
  "alert_type": "COLLISION_IMMINENT",
  "alert_latency_ms": 2.1,
  "cpu_temperature_c": 62.5,
  "dropped_frames": 0,
  "lane_valid": true,
  "detections_count": 3,
  "collision_risks": 1
}
```

**Failure Behavior:**
| Failure Mode | Response |
|--------------|----------|
| Log file write error | Buffer in memory (max 1000 entries), retry |
| Disk full | Rotate log, delete oldest, continue |
| CPU temp read error | Log null, continue |

**Determinism Guarantees:**
- One log entry per processed frame
- Timestamp precision: microseconds
- Buffered writes with configurable flush interval (default: 1 second)
- Log rotation at fixed size (default: 100MB)

---

### 3.8 IR Distance Sensor Module (Optional)

**Purpose:**  
Provide supplementary proximity sensing as a secondary collision warning input.

**Inputs:**
- GPIO pin configuration
- Polling interval

**Outputs:**
- Distance reading (cm) or `None` if unavailable
- Sensor health status

**Integration Rules:**
1. IR sensor is **advisory only** — never overrides vision-based alerts
2. Vision collision alerts always take priority
3. IR may supplement with earlier warning if vision detects no risk
4. Fully disableable via `--disable-ir` CLI flag

**Failure Behavior:**
| Failure Mode | Response |
|--------------|----------|
| Sensor not connected | Disable IR module, log warning |
| Invalid readings | Ignore, use vision only |
| GPIO error | Disable IR module, continue |

**Determinism Guarantees:**
- Fixed polling interval (default: 100ms)
- Readings smoothed with 3-sample median filter
- No impact on vision pipeline timing

---

### 3.9 TF-Luna LiDAR Module (NEW)

**Purpose:**  
Provide distance measurement for collision confirmation, reducing false positives from vision-only detection.

**Hardware:**
- Sensor: Benewake TF-Luna (single-point LiDAR)
- Interface: UART at 115200 baud
- Range: 10cm - 800cm

**Connection:**
| TF-Luna | Raspberry Pi |
|---------|--------------|
| 5V | Pin 2 (5V) |
| GND | Pin 6 (GND) |
| TX | Pin 10 (GPIO15/RXD) |
| RX | Pin 8 (GPIO14/TXD) |

**Inputs:**
- UART serial port (`/dev/ttyAMA0`)
- Configuration: threshold, EMA alpha

**Outputs:**
- Distance reading (cm)
- Signal strength
- Valid flag

**Features:**
- Background thread for continuous reading
- EMA filtering (α=0.3) for noise reduction
- Minimum signal strength validation
- Automatic reconnection on disconnect

**Integration with Collision Detection:**
```python
# Collision alert only triggers when BOTH conditions met:
# 1. Object detected in danger zone (vision)
# 2. LiDAR distance < threshold (600cm default)

# If LiDAR unavailable, system degrades to vision-only
```

**Configuration:**
```yaml
lidar:
  enabled: true
  port: "/dev/ttyAMA0"
  collision_threshold_cm: 600
  required_for_collision: true
```

**Failure Behavior:**
| Failure Mode | Response |
|--------------|----------|
| Sensor not connected | Disable LiDAR, fallback to vision-only |
| Invalid readings | Ignore, use cached value |
| UART error | Attempt reconnection |

---

### 3.10 GPIO Output Module (NEW)

**Purpose:**  
Control status LEDs and external system outputs via GPIO pins.

**Pin Assignments:**
| Pin | Function | Trigger |
|-----|----------|---------|
| GPIO17 | System LED | ON when system running |
| GPIO27 | Alert LED | ON during active alerts |
| GPIO22 | Collision Output | HIGH when object in danger zone |
| GPIO5 | Braking Output | HIGH when COLLISION_IMMINENT |
| GPIO18 | Buzzer | Pattern-based alerts |

**Braking Output (GPIO5) - DEMONSTRATION ONLY:**
```python
# Triggered when:
# - COLLISION_IMMINENT alert is generated
# - Both vision AND LiDAR confirm collision risk

# WARNING: Real autonomous braking requires:
# - Safety-critical redundant hardware
# - Fail-safe mechanisms
# - Regulatory compliance
```

**Configuration:**
```yaml
gpio_leds:
  enabled: true
  system_led_pin: 17
  alert_led_pin: 27
  collision_output_pin: 22
  braking_output_pin: 5
```

**Failure Behavior:**
| Failure Mode | Response |
|--------------|----------|
| GPIO not available | Disable GPIO module, log warning |
| Pin conflict | Skip conflicting pin, continue |

---

### 3.11 Overtake Assistant Module (NEW)

**Purpose:**  
Advisory-only module for evaluating overtake safety. **NOT a safety system.**

**⚠️ DISCLAIMER:**
This module provides advisory information only and must NEVER be relied upon for actual driving decisions. Limitations include:
- Camera blind spots
- Processing delays
- Detection errors
- Cannot see around obstacles
- Does not account for vehicle speeds

**Evaluation Criteria:**
1. **Lane Stability** - Both lanes detected for N consecutive frames
2. **Clearance Zone** - Area to overtaking side clear of vehicles
3. **Lane Marking** - Broken line (overtaking allowed) vs solid line

**Status Output:**
| Status | Meaning |
|--------|---------|
| `DISABLED` | Cannot evaluate (conditions not met) |
| `UNSAFE` | Do not overtake (vehicle in zone or solid line) |
| `SAFE` | Advisory: conditions appear favorable |

**Traffic Side Configuration:**
```yaml
overtake_assistant:
  enabled: true
  traffic_side: "left"   # Drive on left → Overtake on RIGHT (UK, Japan, India, Sri Lanka)
  # traffic_side: "right" # Drive on right → Overtake on LEFT (US, Europe)
  safe_frames_required: 8
  zone_width_ratio: 1.5
```

**Clearance Zone Geometry:**
- Zone is calculated relative to detected lanes
- Width: 1.5× lane width (configurable)
- Extends from `zone_y_top_ratio` to bottom of frame

**Integration:**
```python
# Called after lane detection and YOLO inference
overtake_status = overtake_assistant.evaluate(
    lane_result=lane_result,
    detections=detections,
    frame_shape=frame.shape
)

# Status displayed on overlay (optional)
# Does NOT trigger any audio alerts
```

---

## 4. Execution & Scheduling Model

### 4.1 Main Loop Structure

The system uses a **single-threaded main loop** with explicit scheduling checkpoints for deterministic behavior.

```python
def main_loop():
    frame_counter = 0
    yolo_skip_interval = 3  # Run YOLO every 3rd frame
    
    while running:
        loop_start = time.monotonic()
        
        # 1. Frame Acquisition (blocking with timeout)
        frame = capture_module.capture()
        if frame is None:
            telemetry.log_dropped_frame()
            continue
        
        # 2. Lane Detection (every frame)
        lane_result = lane_module.process(frame)
        
        # 3. YOLO Detection (scheduled)
        if frame_counter % yolo_skip_interval == 0:
            detections = yolo_module.infer(frame)
            detection_cache.update(detections)
        else:
            detections = detection_cache.get_valid()
        
        # 4. Danger Zone Evaluation
        collision_risks = danger_zone.evaluate(detections)
        
        # 5. Alert Decision
        alert = decision_engine.evaluate(
            collision_risks=collision_risks,
            lane_result=lane_result,
            traffic_lights=filter_traffic_lights(detections)
        )
        
        # 6. Alert Dispatch (non-blocking)
        if alert:
            audio_system.dispatch(alert)
        
        # 7. Optional Display (non-blocking)
        if display_enabled:
            renderer.render(frame, detections, lane_result, alert)
        
        # 8. Telemetry
        telemetry.log_frame(...)
        
        # 9. Frame timing enforcement
        frame_counter += 1
        enforce_frame_timing(loop_start, target_fps=15)
```

### 4.2 Threading Model

**Primary Thread (Main Loop):**
- Frame capture
- Lane detection
- YOLO inference
- Danger zone evaluation
- Alert decision
- Telemetry logging

**Secondary Thread (Audio Playback):**
- Non-blocking audio dispatch
- Managed by audio backend (pygame/ALSA)
- Communicates via thread-safe queue (size 1)

**Optional Thread (Display Rendering):**
- OpenCV `imshow` in separate thread if enabled
- Double-buffered frame handoff
- Never blocks main loop

### 4.3 Frame Skipping Strategy

| Component | Execution Frequency | Rationale |
|-----------|---------------------|-----------|
| Frame Capture | Every frame | Maintain real-time video continuity |
| Lane Detection | Every frame | Low latency requirement, fast CV pipeline |
| YOLO Inference | Every Nth frame (N=3 default) | CPU bottleneck, ~100ms per inference |
| Alert Decision | Every frame | Use cached YOLO + fresh lane data |

**YOLO Frame Skip Configuration:**
```python
@dataclass
class InferenceSchedule:
    skip_interval: int = 3        # Infer on frames 0, 3, 6, ...
    cache_ttl_ms: float = 400.0   # Max age for cached detections
    min_confidence: float = 0.25  # Discard detections below this
```

### 4.4 Detection Cache Management

```python
class DetectionCache:
    def __init__(self, ttl_ms: float = 400.0):
        self.ttl_ms = ttl_ms
        self.detections: List[Detection] = []
        self.timestamp: float = 0.0
    
    def update(self, detections: List[Detection]) -> None:
        self.detections = detections
        self.timestamp = time.monotonic()
    
    def get_valid(self) -> List[Detection]:
        age_ms = (time.monotonic() - self.timestamp) * 1000
        if age_ms > self.ttl_ms:
            return []  # Stale, return empty
        return self.detections
    
    def is_stale(self) -> bool:
        age_ms = (time.monotonic() - self.timestamp) * 1000
        return age_ms > self.ttl_ms
```

### 4.5 Non-Blocking Guarantees

| Operation | Blocking Allowed? | Mitigation |
|-----------|-------------------|------------|
| Frame capture | Yes (with timeout) | 100ms timeout, skip on failure |
| YOLO inference | Yes (bounded) | Single-threaded, measured latency |
| Lane detection | No | Fast CV pipeline, <20ms typical |
| Alert decision | No | O(n) evaluation, n < 20 |
| Audio dispatch | No | Queue handoff to audio thread |
| Display render | No | Double buffer, separate thread |
| Telemetry write | No | Buffered async writes |

---

## 5. Data Contracts & Interfaces

### 5.1 Frame Format Contract

```python
@dataclass
class Frame:
    data: np.ndarray          # Shape: (480, 640, 3), dtype: uint8, BGR
    timestamp: float          # Monotonic time of capture
    sequence: int             # Frame counter (0-indexed)
    source: FrameSource       # Enum: CSI, WEBCAM, VIDEO_FILE
    
    def validate(self) -> bool:
        return (
            self.data.shape == (480, 640, 3) and
            self.data.dtype == np.uint8 and
            self.timestamp > 0
        )
```

### 5.2 Detection Output Schema

```python
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, List

class DetectionClass(Enum):
    TRAFFIC_LIGHT_RED = "traffic_light_red"
    TRAFFIC_LIGHT_YELLOW = "traffic_light_yellow"
    TRAFFIC_LIGHT_GREEN = "traffic_light_green"
    PEDESTRIAN = "pedestrian"
    VEHICLE = "vehicle"

@dataclass
class Detection:
    label: DetectionClass
    confidence: float                        # [0.0, 1.0]
    bbox: Tuple[int, int, int, int]         # (x_min, y_min, x_max, y_max)
    timestamp: float                         # Inference timestamp
    
    def to_dict(self) -> dict:
        return {
            "label": self.label.value,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox)
        }
```

### 5.3 Lane Output Format

```python
@dataclass
class LanePolynomial:
    # Polynomial: y = ax² + bx + c (solved for x given y)
    coefficients: Tuple[float, float, float]  # (a, b, c)
    y_range: Tuple[int, int]                  # (y_top, y_bottom)
    confidence: float                          # [0.0, 1.0]
    point_count: int                           # Number of points used in fit
    
    def evaluate(self, y: int) -> float:
        """Return x coordinate for given y."""
        a, b, c = self.coefficients
        return a * y**2 + b * y + c

@dataclass
class LaneResult:
    left_lane: Optional[LanePolynomial]
    right_lane: Optional[LanePolynomial]
    valid: bool                               # Both lanes detected and stable
    partial: bool                             # Only one lane detected
    timestamp: float
    latency_ms: float
    
    def get_lane_center(self, y: int) -> Optional[float]:
        """Return lane center x-coordinate at given y."""
        if not self.valid:
            return None
        left_x = self.left_lane.evaluate(y)
        right_x = self.right_lane.evaluate(y)
        return (left_x + right_x) / 2
```

### 5.4 Alert Event Schema

```python
class AlertType(Enum):
    COLLISION_IMMINENT = "collision_imminent"
    LANE_DEPARTURE_LEFT = "lane_departure_left"
    LANE_DEPARTURE_RIGHT = "lane_departure_right"
    TRAFFIC_LIGHT_RED = "traffic_light_red"
    TRAFFIC_LIGHT_YELLOW = "traffic_light_yellow"
    SYSTEM_WARNING = "system_warning"        # Camera disconnect, etc.

@dataclass
class AlertEvent:
    alert_type: AlertType
    priority: int                            # 1 (highest) to 3 (lowest)
    timestamp: float
    trigger_source: str                      # Module name
    confidence: float
    metadata: dict                           # Type-specific data
    
    def to_dict(self) -> dict:
        return {
            "alert_type": self.alert_type.value,
            "priority": self.priority,
            "timestamp": self.timestamp,
            "trigger_source": self.trigger_source,
            "confidence": round(self.confidence, 4),
            "metadata": self.metadata
        }
```

### 5.5 Telemetry JSON Lines Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["timestamp", "frame_seq"],
  "properties": {
    "timestamp": {
      "type": "string",
      "format": "date-time"
    },
    "frame_seq": {
      "type": "integer",
      "minimum": 0
    },
    "capture_fps": {
      "type": "number",
      "minimum": 0
    },
    "capture_latency_ms": {
      "type": "number",
      "minimum": 0
    },
    "lane_latency_ms": {
      "type": "number",
      "minimum": 0
    },
    "yolo_latency_ms": {
      "type": ["number", "null"],
      "minimum": 0
    },
    "yolo_skipped": {
      "type": "boolean"
    },
    "decision_latency_ms": {
      "type": "number",
      "minimum": 0
    },
    "alert_type": {
      "type": ["string", "null"],
      "enum": [
        "collision_imminent",
        "lane_departure_left", 
        "lane_departure_right",
        "traffic_light_red",
        "traffic_light_yellow",
        "system_warning",
        null
      ]
    },
    "alert_latency_ms": {
      "type": ["number", "null"],
      "minimum": 0
    },
    "cpu_temperature_c": {
      "type": ["number", "null"]
    },
    "dropped_frames": {
      "type": "integer",
      "minimum": 0
    },
    "lane_valid": {
      "type": "boolean"
    },
    "detections_count": {
      "type": "integer",
      "minimum": 0
    },
    "collision_risks": {
      "type": "integer",
      "minimum": 0
    }
  }
}
```

**Example Log Entry:**
```json
{"timestamp":"2026-01-28T14:30:00.123456Z","frame_seq":12345,"capture_fps":15.2,"capture_latency_ms":12.5,"lane_latency_ms":8.3,"yolo_latency_ms":95.2,"yolo_skipped":false,"decision_latency_ms":0.5,"alert_type":"collision_imminent","alert_latency_ms":2.1,"cpu_temperature_c":62.5,"dropped_frames":0,"lane_valid":true,"detections_count":3,"collision_risks":1}
```

---

## 6. Safety & Failure Handling

### 6.1 Failure Mode Analysis

| Component | Failure Mode | Detection Method | Response | Severity |
|-----------|--------------|------------------|----------|----------|
| Camera | Disconnection | Read timeout | Reconnect × 3, audible warning | High |
| Camera | Corrupted frame | Shape/dtype check | Discard frame, continue | Low |
| YOLO Model | File not found | Startup check | **Abort startup** | Critical |
| YOLO Model | Load failure | ONNX exception | **Abort startup** | Critical |
| YOLO Model | Inference error | Runtime exception | Return empty, use cache | Medium |
| Lane Detection | No lines found | Empty result | Disable lane alerts | Medium |
| Lane Detection | Single lane only | Partial result | Warn, partial operation | Low |
| Audio | Device unavailable | Init failure | GPIO buzzer only | Medium |
| Audio | File missing | Load check | GPIO buzzer only | Medium |
| GPIO | Unavailable (Windows) | Platform check | Audio only | Low |
| Telemetry | Write failure | IO exception | Buffer in memory | Low |

### 6.2 Camera Failure Handling

```python
class CameraFailureHandler:
    MAX_RETRIES = 3
    RETRY_INTERVAL_MS = 500
    
    def handle_failure(self, capture_module: FrameCaptureModule) -> bool:
        """Attempt to recover from camera failure."""
        for attempt in range(self.MAX_RETRIES):
            logging.warning(f"Camera failure, retry {attempt + 1}/{self.MAX_RETRIES}")
            time.sleep(self.RETRY_INTERVAL_MS / 1000)
            
            if capture_module.reconnect():
                logging.info("Camera reconnected successfully")
                return True
        
        # All retries failed
        logging.error("Camera recovery failed, emitting warning")
        audio_system.dispatch(AlertEvent(
            alert_type=AlertType.SYSTEM_WARNING,
            priority=1,
            timestamp=time.monotonic(),
            trigger_source="camera",
            confidence=1.0,
            metadata={"message": "camera_failure"}
        ))
        return False
```

### 6.3 Model Load Failure Handling

```python
def initialize_yolo_module(model_path: str) -> YOLOModule:
    """Initialize YOLO module. Aborts on failure."""
    if not os.path.exists(model_path):
        logging.critical(f"YOLO model not found: {model_path}")
        sys.exit(1)
    
    try:
        module = YOLOModule(model_path)
        module.warmup()  # Run dummy inference to verify
        return module
    except Exception as e:
        logging.critical(f"YOLO model load failed: {e}")
        sys.exit(1)
```

### 6.4 Lane Detection Failure Handling

```python
class LaneFailureState:
    def __init__(self, max_invalid_frames: int = 5):
        self.max_invalid_frames = max_invalid_frames
        self.invalid_frame_count = 0
        self.last_valid_result: Optional[LaneResult] = None
        self.alerts_disabled = False
    
    def update(self, result: LaneResult) -> LaneResult:
        if result.valid:
            self.invalid_frame_count = 0
            self.last_valid_result = result
            self.alerts_disabled = False
            return result
        
        self.invalid_frame_count += 1
        
        if self.invalid_frame_count <= self.max_invalid_frames:
            # Reuse last valid result
            if self.last_valid_result:
                return LaneResult(
                    left_lane=self.last_valid_result.left_lane,
                    right_lane=self.last_valid_result.right_lane,
                    valid=False,  # Mark as stale
                    partial=self.last_valid_result.partial,
                    timestamp=result.timestamp,
                    latency_ms=result.latency_ms
                )
        
        # Exceeded threshold, disable lane alerts
        self.alerts_disabled = True
        logging.warning("Lane detection failed, disabling lane departure alerts")
        return LaneResult(
            left_lane=None,
            right_lane=None,
            valid=False,
            partial=False,
            timestamp=result.timestamp,
            latency_ms=result.latency_ms
        )
```

### 6.5 Audio Failure Handling

```python
class AudioFailureHandler:
    def __init__(self):
        self.audio_available = False
        self.gpio_available = False
    
    def initialize(self) -> None:
        # Try audio backend
        try:
            pygame.mixer.init()
            self.audio_available = True
        except Exception as e:
            logging.warning(f"Audio backend unavailable: {e}")
        
        # Try GPIO (Raspberry Pi only)
        if platform.system() == "Linux":
            try:
                import RPi.GPIO as GPIO
                GPIO.setmode(GPIO.BCM)
                self.gpio_available = True
            except Exception as e:
                logging.warning(f"GPIO unavailable: {e}")
    
    def dispatch(self, alert: AlertEvent) -> bool:
        """Dispatch alert via available outputs. Never raises."""
        success = False
        
        if self.audio_available:
            try:
                play_audio(alert)
                success = True
            except Exception as e:
                logging.error(f"Audio playback failed: {e}")
        
        if self.gpio_available:
            try:
                trigger_buzzer(alert)
                success = True
            except Exception as e:
                logging.error(f"GPIO buzzer failed: {e}")
        
        return success
```

### 6.6 Safe Degradation Matrix

| Failure | Lane Alerts | Collision Alerts | Traffic Alerts | Audio | System Status |
|---------|-------------|------------------|----------------|-------|---------------|
| Camera failure | ❌ Disabled | ❌ Disabled | ❌ Disabled | ✅ Warning | Degraded |
| YOLO failure | ✅ Active | ❌ Disabled | ❌ Disabled | ✅ Active | Degraded |
| Lane failure | ❌ Disabled | ✅ Active | ✅ Active | ✅ Active | Degraded |
| Audio failure | ✅ Active | ✅ Active | ✅ Active | GPIO only | Degraded |
| GPIO failure | ✅ Active | ✅ Active | ✅ Active | Audio only | Degraded |
| All healthy | ✅ Active | ✅ Active | ✅ Active | ✅ Active | Nominal |

---

## 7. Performance & Latency Budget

### 7.1 Per-Stage Latency Budget

| Stage | Target (ms) | Max (ms) | Notes |
|-------|-------------|----------|-------|
| Frame Capture | 15 | 30 | CSI camera polling |
| YOLO Preprocessing | 5 | 10 | Resize, normalize |
| YOLO Inference | 80 | 120 | YOLOv11s CPU, 640×640 |
| YOLO Postprocessing | 5 | 10 | NMS, coordinate scaling |
| Lane Detection | 15 | 25 | Full CV pipeline |
| Danger Zone Eval | 1 | 2 | Polygon intersection |
| Alert Decision | 1 | 2 | Priority evaluation |
| Audio Dispatch | 2 | 5 | Queue handoff |
| Display Render | 10 | 20 | Optional, non-blocking |
| **Total (with YOLO)** | **124** | **214** | — |
| **Total (YOLO skipped)** | **44** | **84** | — |

### 7.2 End-to-End Latency Analysis

**Detection-to-Alert Latency (Median Target: ≤300ms):**

| Scenario | Latency Calculation | Expected |
|----------|---------------------|----------|
| YOLO frame (best) | Capture + YOLO + Decision + Audio | ~110ms |
| YOLO frame (worst) | With queuing delays | ~170ms |
| Cached detection (best) | Capture + Decision + Audio | ~25ms |
| Cached detection (worst) | Near cache TTL expiry | ~425ms |

**Worst-Case Analysis:**
- Frame N: YOLO inference completes at t=0
- Frame N+1: Cache used, t+67ms
- Frame N+2: Cache used, t+134ms (approaching TTL)
- Frame N+3: New YOLO inference, t+201ms

Maximum detection-to-alert latency with 400ms TTL and 3-frame skip:
- Object appears at frame N (between YOLO runs)
- Detected at frame N+2 (next YOLO)
- Alert at N+2 + inference time
- **Worst case: ~300ms** (meets PRD requirement)

### 7.3 FPS Analysis

| Configuration | Expected FPS | Bottleneck |
|---------------|--------------|------------|
| YOLO every frame | 8-10 FPS | YOLO inference |
| YOLO every 3rd frame | 14-16 FPS | Frame capture |
| YOLO every 5th frame | 15+ FPS | Frame capture |
| Lane only (no YOLO) | 30+ FPS | Frame capture |

**Recommended Configuration:** YOLO every 3rd frame (skip_interval=3)

### 7.4 Memory Budget (Raspberry Pi 4, 8GB)

| Component | Estimated Memory | Notes |
|-----------|------------------|-------|
| OS + System | 500 MB | Raspberry Pi OS |
| Python Runtime | 100 MB | Interpreter + stdlib |
| ONNX Runtime | 200 MB | CPU execution provider |
| YOLOv11s Model | 150 MB | Loaded weights |
| OpenCV | 100 MB | Core + image buffers |
| Frame Buffers | 50 MB | 3× BGR frames |
| Detection Cache | 10 MB | Cached results |
| **Total** | **~1.1 GB** | Well within 8GB |

### 7.5 Thermal Considerations

| CPU Temperature | System Response |
|-----------------|-----------------|
| < 60°C | Normal operation |
| 60-70°C | Log warning, continue |
| 70-80°C | Log warning, recommend cooling |
| > 80°C | Log critical, system continues |

**Note:** Thermal management is monitoring only. Active cooling is optional and does not affect system correctness.

---

## 8. Platform Abstraction Strategy

### 8.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     CORE LOGIC (Platform-Independent)           │
│                                                                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐  │
│  │ Lane        │ │ YOLO        │ │ Alert       │ │ Telemetry │  │
│  │ Detection   │ │ Detection   │ │ Decision    │ │ Logging   │  │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘  │
│                                                                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐    │
│  │ Danger Zone │ │ Frame       │ │ Configuration           │    │
│  │ Evaluation  │ │ Processing  │ │ Management              │    │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐
│ Camera Adapter  │ │ Audio Adapter   │ │ GPIO Adapter            │
│                 │ │                 │ │                         │
│ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────────────┐ │
│ │ CSI Camera  │ │ │ │ ALSA/PulseA │ │ │ │ RPi.GPIO            │ │
│ │ (Pi)        │ │ │ │ (Pi)        │ │ │ │ (Pi)                │ │
│ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────────────┘ │
│ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────────────┐ │
│ │ OpenCV      │ │ │ │ DirectSound │ │ │ │ Stub (No-op)        │ │
│ │ (Windows)   │ │ │ │ (Windows)   │ │ │ │ (Windows)           │ │
│ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────────────┘ │
│ ┌─────────────┐ │ └─────────────────┘ └─────────────────────────┘
│ │ Video File  │ │
│ │ (Both)      │ │
│ └─────────────┘ │
└─────────────────┘
```

### 8.2 Platform Adapter Interfaces

```python
# camera_adapter.py
from abc import ABC, abstractmethod

class CameraAdapter(ABC):
    @abstractmethod
    def initialize(self, config: CameraConfig) -> bool: ...
    
    @abstractmethod
    def capture(self) -> Optional[np.ndarray]: ...
    
    @abstractmethod
    def release(self) -> None: ...
    
    @abstractmethod
    def is_available(self) -> bool: ...

class CSICameraAdapter(CameraAdapter):
    """Raspberry Pi CSI camera via libcamera/picamera2."""
    pass

class OpenCVCameraAdapter(CameraAdapter):
    """Windows webcam via OpenCV VideoCapture."""
    pass

class VideoFileAdapter(CameraAdapter):
    """Video file playback (both platforms)."""
    pass
```

```python
# audio_adapter.py
from abc import ABC, abstractmethod

class AudioAdapter(ABC):
    @abstractmethod
    def initialize(self, config: AudioConfig) -> bool: ...
    
    @abstractmethod
    def play(self, sound_file: str, blocking: bool = False) -> bool: ...
    
    @abstractmethod
    def stop(self) -> None: ...
    
    @abstractmethod
    def is_available(self) -> bool: ...

class ALSAAudioAdapter(AudioAdapter):
    """Raspberry Pi audio via ALSA/pygame."""
    pass

class WindowsAudioAdapter(AudioAdapter):
    """Windows audio via pygame/DirectSound."""
    pass
```

```python
# gpio_adapter.py
from abc import ABC, abstractmethod

class GPIOAdapter(ABC):
    @abstractmethod
    def initialize(self, config: GPIOConfig) -> bool: ...
    
    @abstractmethod
    def set_output(self, pin: int, state: bool) -> None: ...
    
    @abstractmethod
    def pulse(self, pin: int, duration_ms: int) -> None: ...
    
    @abstractmethod
    def cleanup(self) -> None: ...

class RPiGPIOAdapter(GPIOAdapter):
    """Raspberry Pi GPIO via RPi.GPIO library."""
    pass

class StubGPIOAdapter(GPIOAdapter):
    """No-op stub for Windows (logs actions only)."""
    def set_output(self, pin: int, state: bool) -> None:
        logging.debug(f"GPIO stub: pin {pin} = {state}")
    
    def pulse(self, pin: int, duration_ms: int) -> None:
        logging.debug(f"GPIO stub: pin {pin} pulse {duration_ms}ms")
```

### 8.3 Platform Detection and Factory

```python
# platform_factory.py
import platform

def create_camera_adapter(config: CameraConfig) -> CameraAdapter:
    if config.source == FrameSource.VIDEO_FILE:
        return VideoFileAdapter(config.video_path)
    
    if platform.system() == "Linux" and is_raspberry_pi():
        return CSICameraAdapter(config)
    else:
        return OpenCVCameraAdapter(config)

def create_audio_adapter(config: AudioConfig) -> AudioAdapter:
    if platform.system() == "Linux":
        return ALSAAudioAdapter(config)
    else:
        return WindowsAudioAdapter(config)

def create_gpio_adapter(config: GPIOConfig) -> GPIOAdapter:
    if platform.system() == "Linux" and is_raspberry_pi():
        return RPiGPIOAdapter(config)
    else:
        return StubGPIOAdapter()

def is_raspberry_pi() -> bool:
    try:
        with open("/proc/device-tree/model", "r") as f:
            return "Raspberry Pi" in f.read()
    except:
        return False
```

### 8.4 CLI Configuration

Argument parsing is implemented in `src/main.py` (`parse_args`). The user-facing entry point is `driver_assistant.py` in the repository root (it calls `src.main.main()`).

- **Sources:** `--source {csi,webcam,video,ip}` — default is `csi` on Raspberry Pi and `webcam` on other platforms.
- **Networking:** `--ip-url` required when `--source ip`.
- **Display:** mutually exclusive `--display` and `--headless`; default display behavior differs by OS (see code after parse).
- **Overrides:** `--model`, `--yolo-skip`, `--confidence`, `--resolution WxH`, `--lane-debug`, `--log-file`, `--log-level`.
- **IR:** `--enable-ir` enables the optional IR sensor; otherwise IR stays disabled (`--disable-ir` is parsed but paired logic favors `--enable-ir`).

Authoritative, up-to-date text:

```bash
python driver_assistant.py --help
```

**Example Invocations:**

```bash
# Raspberry Pi (headless, production; model path from config.yaml unless overridden)
python driver_assistant.py --source csi --headless

# Raspberry Pi (with display)
python driver_assistant.py --source csi --display

# Windows (webcam)
python driver_assistant.py --source webcam --display

# Video file
python driver_assistant.py --source video --video-path videos/test.mp4 --display

# Optional IP camera
python driver_assistant.py --source ip --ip-url http://192.168.1.100:8080/video --display
```

---

## 9. Development & Testing Strategy

### 9.1 Mandatory Development Order

Per PRD Section 13, development must proceed in this order:

| Phase | Module | Deliverables | Dependencies |
|-------|--------|--------------|--------------|
| 1 | Lane Detection | Full CV pipeline, temporal stabilization, unit tests | OpenCV |
| 2 | YOLO Object Detection | ONNX inference, preprocessing, postprocessing, unit tests | ONNX Runtime, Phase 1 |
| 3 | Alert Logic + Audio | Decision engine, audio dispatch, GPIO, integration tests | Phase 1, Phase 2 |
| 4 | IR Distance Sensor | GPIO integration, fusion logic, validation | Phase 3 |

### 9.2 Module-Level Testing

**Lane Detection Tests:**

| Test Case | Input | Expected Output | Category |
|-----------|-------|-----------------|----------|
| Clear highway lanes | Highway image with white lanes | Both lanes detected, valid=True | Positive |
| Urban lanes (yellow) | Urban road with yellow markings | Both lanes detected | Positive |
| Single lane visible | Curved road, one lane | Partial result, one lane | Edge |
| Zebra crossing | Crosswalk in ROI | Lanes detected, crosswalk ignored | Robustness |
| Stop line | Stop line in frame | Lanes detected, stop line ignored | Robustness |
| Directional arrow | Arrow marking | Lanes detected, arrow ignored | Robustness |
| Road text | "SLOW" painted | Lanes detected, text ignored | Robustness |
| No lanes | Off-road surface | valid=False, no crash | Negative |
| Night image | Low light | valid=False (out of scope) | Boundary |
| Rain/wet road | Reflections | Best effort, may be invalid | Environmental |

**YOLO Detection Tests:**

| Test Case | Input | Expected Output | Category |
|-----------|-------|-----------------|----------|
| Single pedestrian | Pedestrian in frame | 1 detection, correct class | Positive |
| Multiple vehicles | 3 cars | 3 detections | Positive |
| Red traffic light | Red light visible | traffic_light_red detected | Positive |
| Yellow traffic light | Yellow light visible | traffic_light_yellow detected | Positive |
| Green traffic light | Green light visible | traffic_light_green detected | Positive |
| Empty scene | Road only | Empty detection list | Negative |
| Occluded pedestrian | Partial visibility | Detection if conf > threshold | Edge |
| Small/distant objects | Far objects | May not detect (acceptable) | Boundary |
| Model load (invalid) | Corrupt .onnx | Abort startup | Error |

**Alert Decision Tests:**

| Test Case | Inputs | Expected Alert | Priority Check |
|-----------|--------|----------------|----------------|
| Collision only | Pedestrian in danger zone | COLLISION_IMMINENT | P1 |
| Lane departure only | Vehicle crosses left lane | LANE_DEPARTURE_LEFT | P2 |
| Red light only | Red light detected | TRAFFIC_LIGHT_RED | P3 |
| Collision + lane departure | Both conditions | COLLISION_IMMINENT | P1 > P2 |
| Collision + red light | Both conditions | COLLISION_IMMINENT | P1 > P3 |
| Lane + red light | Both conditions | LANE_DEPARTURE | P2 > P3 |
| All three | All conditions | COLLISION_IMMINENT | P1 highest |
| None | No hazards | No alert | — |

### 9.3 Integration Testing

**End-to-End Test Scenarios:**

| Scenario | Setup | Validation |
|----------|-------|------------|
| Continuous operation (1hr) | Video loop, all modules | No crash, memory stable |
| Camera disconnect/reconnect | Unplug/replug USB | Recovery, audible warning |
| High CPU load | Background stress | Graceful degradation |
| Rapid state changes | Video with frequent hazards | Correct priority handling |
| All alerts triggered | Synthetic video | Each alert fires correctly |

### 9.4 Performance Validation

**Latency Profiling:**
```python
def profile_frame_processing():
    """Profile each stage of frame processing."""
    metrics = {}
    
    t0 = time.monotonic()
    frame = capture_module.capture()
    metrics["capture_ms"] = (time.monotonic() - t0) * 1000
    
    t0 = time.monotonic()
    lane_result = lane_module.process(frame)
    metrics["lane_ms"] = (time.monotonic() - t0) * 1000
    
    t0 = time.monotonic()
    detections = yolo_module.infer(frame)
    metrics["yolo_ms"] = (time.monotonic() - t0) * 1000
    
    # ... additional stages
    
    return metrics
```

**Stability Criteria:**
- 1-hour continuous run without crash
- Memory usage stable (no growth >5%)
- FPS within 10% of target
- Alert latency within budget 95th percentile

### 9.5 Test Fixtures and Mock Data

```
tests/
├── fixtures/
│   ├── images/
│   │   ├── highway_clear.jpg
│   │   ├── urban_lanes.jpg
│   │   ├── zebra_crossing.jpg
│   │   ├── stop_line.jpg
│   │   ├── directional_arrow.jpg
│   │   ├── road_text_slow.jpg
│   │   ├── pedestrian_single.jpg
│   │   ├── vehicles_multiple.jpg
│   │   └── traffic_light_*.jpg
│   └── videos/
│       ├── highway_5min.mp4
│       ├── urban_mixed.mp4
│       └── stress_test.mp4
├── unit/
│   ├── test_lane_detection.py
│   ├── test_yolo_detection.py
│   ├── test_danger_zone.py
│   ├── test_alert_decision.py
│   └── test_audio_system.py
├── integration/
│   ├── test_full_pipeline.py
│   ├── test_failure_recovery.py
│   └── test_platform_adapters.py
└── performance/
    ├── test_latency.py
    ├── test_memory.py
    └── test_stability.py
```

---

## 10. Explicit Non-Goals

The following are explicitly **out of scope** for this system:

### 10.1 Vehicle Control
- The system does NOT actuate brakes, steering, or throttle
- The system does NOT send commands to vehicle CAN bus
- The system does NOT interface with vehicle ECUs

### 10.2 Certification and Compliance
- The system does NOT claim ISO 26262 compliance
- The system does NOT claim any ASIL rating
- The system does NOT meet automotive functional safety standards
- The system is a **proof-of-concept** only

### 10.3 Environmental Conditions
- Night-time operation is NOT supported
- Low-light or HDR operation is NOT supported
- Infrared camera operation is NOT supported
- Rain, snow, fog handling is NOT guaranteed

### 10.4 Advanced Perception
- Multi-camera / surround view is NOT supported
- 3D depth estimation is NOT supported
- Object tracking across frames is NOT implemented (detections are per-frame)
- Scene understanding beyond specified classes is NOT supported

### 10.5 Connectivity
- Cloud connectivity is NOT used
- OTA updates are NOT implemented
- Remote telemetry streaming is NOT implemented
- Vehicle telematics integration is NOT supported

### 10.6 Additional Sensors
- GPS/GNSS integration is NOT supported
- IMU/accelerometer integration is NOT supported
- LIDAR integration is NOT supported
- Radar integration is NOT supported
- Only optional IR distance sensor is in scope (final phase)

### 10.7 User Interface
- Mobile app is NOT provided
- Web dashboard is NOT provided
- Voice interface is NOT supported
- Display is optional, audio-only is primary mode

### 10.8 Recording and Playback
- Video recording (dashcam functionality) is NOT implemented
- Incident capture is NOT implemented
- Evidence storage is NOT supported

---

## Appendix A: Configuration Schema

```yaml
# config.yaml - System Configuration

system:
  log_level: INFO
  log_file: telemetry.jsonl
  telemetry_flush_interval_s: 1.0

capture:
  resolution: [640, 480]
  target_fps: 15
  timeout_ms: 100
  reconnect_attempts: 3
  reconnect_interval_ms: 500

yolo:
  model_path: models/object.onnx
  input_size: [640, 640]
  confidence_threshold: 0.25
  iou_threshold: 0.45
  frame_skip: 5
  cache_ttl_ms: 500

lane_detection:
  roi_top_ratio: 0.5          # ROI starts at 50% from top
  hsv_white:
    h: [0, 180]
    s: [0, 30]
    v: [200, 255]
  hsv_yellow:
    h: [15, 35]
    s: [80, 255]
    v: [150, 255]
  gaussian_kernel: [5, 5]
  canny_low: 50
  canny_high: 150
  hough_rho: 2
  hough_theta_deg: 1
  hough_threshold: 50
  hough_min_length: 40
  hough_max_gap: 100
  slope_range: [0.5, 2.0]
  min_line_length: 40
  ema_alpha: 0.3
  max_invalid_frames: 5

danger_zone:
  # Coordinates as ratios of frame dimensions
  top_left: [0.375, 0.5]
  top_right: [0.625, 0.5]
  bottom_left: [0.125, 1.0]
  bottom_right: [0.875, 1.0]

alerts:
  cooldown_ms: 300
  collision_sound: sounds/collision.wav
  lane_left_sound: sounds/lane_left.wav
  lane_right_sound: sounds/lane_right.wav
  red_light_sound: sounds/red_light.wav
  yellow_light_sound: sounds/yellow_light.wav
  system_warning_sound: sounds/warning.wav

gpio:  # Raspberry Pi only
  buzzer_pin: 18
  enabled: true

ir_sensor:  # Optional
  enabled: false
  gpio_trigger: 23
  gpio_echo: 24
  poll_interval_ms: 100
  threshold_cm: 50
```

---

## Appendix B: Directory Structure

```
Driver-Assistant/
├── README.md
├── ARCHITECTURE.md              # This document
├── ARCHITECTURE_SUMMARY.md      # Condensed LLM-friendly summary
├── requirements.txt
├── requirements-pi.txt          # Raspberry Pi specific
├── requirements-dev.txt         # Development/testing
├── requirements-analysis.txt    # Telemetry analysis tools
├── config.yaml                  # Default configuration
├── driver_assistant.py          # CLI entry point
│
├── src/
│   ├── __init__.py
│   ├── main.py                  # Main processing loop
│   ├── config.py                # Configuration management
│   │
│   ├── capture/
│   │   ├── __init__.py
│   │   ├── adapter.py           # Abstract camera adapter
│   │   ├── factory.py           # Camera factory
│   │   ├── frame.py             # Frame dataclass
│   │   ├── csi_camera.py        # Raspberry Pi CSI (picamera2)
│   │   ├── ip_camera.py         # IP camera (MJPEG/RTSP)
│   │   ├── opencv_camera.py     # USB webcam
│   │   └── video_file.py        # Video file playback
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── detector.py          # YOLO ONNX inference
│   │   ├── preprocessing.py     # Input preprocessing
│   │   ├── postprocessing.py    # NMS, coordinate scaling
│   │   └── result.py            # Detection dataclass
│   │
│   ├── lane/
│   │   ├── __init__.py
│   │   ├── pipeline.py          # Main lane detection
│   │   ├── color_filter.py      # HSV segmentation
│   │   ├── edge_detection.py    # Canny edges
│   │   ├── hough_lines.py       # Line extraction
│   │   ├── geometric_filter.py  # Slope-based classification
│   │   ├── polynomial_fit.py    # Lane fitting
│   │   ├── temporal.py          # EMA stabilization
│   │   └── result.py            # LaneResult dataclass
│   │
│   ├── alerts/
│   │   ├── __init__.py
│   │   ├── decision.py          # Priority-based decisions
│   │   ├── audio.py             # Audio playback
│   │   ├── gpio_buzzer.py       # GPIO buzzer control
│   │   └── types.py             # Alert types
│   │
│   ├── sensors/
│   │   ├── __init__.py
│   │   ├── lidar.py             # TF-Luna LiDAR
│   │   └── ir_distance.py       # Optional IR sensor
│   │
│   ├── gpio/
│   │   ├── __init__.py
│   │   └── status_leds.py       # Status LEDs + braking output
│   │
│   ├── overtake/
│   │   ├── __init__.py
│   │   ├── assistant.py         # Main overtake module
│   │   ├── clearance.py         # Clearance zone calculation
│   │   ├── state.py             # State machine
│   │   ├── line_analysis.py     # Broken/solid detection
│   │   └── types.py             # OvertakeStatus enum
│   │
│   ├── display/
│   │   ├── __init__.py
│   │   └── renderer.py          # Optional overlay rendering
│   │
│   ├── telemetry/
│   │   ├── __init__.py
│   │   └── logger.py            # JSON Lines logging
│   │
│   └── utils/
│       ├── __init__.py
│       ├── platform.py          # Platform detection
│       ├── timing.py            # Timing utilities
│       └── geometry.py          # Polygon operations
│
├── tools/
│   ├── analyze_telemetry.py     # Telemetry analysis with graphs
│   └── diagnose_camera.py       # Camera diagnostic tool
│
├── models/
│   └── object.onnx              # YOLOv11s model
│
├── sounds/
│   ├── collision.wav
│   ├── lane_left.wav
│   ├── lane_right.wav
│   ├── red_light.wav
│   ├── yellow_light.wav
│   └── warning.wav
│
├── tests/
│   ├── conftest.py              # pytest fixtures
│   ├── test_*.py                # Test modules
│   └── fixtures/                # Test images/videos
│
├── Pi_Eyes/
│   └── pi_client.py             # ZMQ client: Pi camera → remote inference (distributed mode)
├── Laptop_Brain/
│   └── laptop_server.py         # ZMQ server: inference on PC for Pi_Eyes
├── scripts/
│   └── setup-pi.sh              # Pi venv + copy systemd unit + daemon-reload
├── systemd/
│   ├── driver-assistant.service # systemd unit (entry: driver_assistant.py --source csi --headless)
│   └── install-service.sh       # Copy unit to /etc/systemd/system and enable
│
└── videos/                      # Test video files
```

---

## Appendix C: Ambiguity Resolution

The following PRD requirements required interpretation. The safest, most deterministic interpretation was chosen:

| PRD Section | Ambiguity | Resolution | Rationale |
|-------------|-----------|------------|-----------|
| 6.4 | Lane output format shows points, but 6.2 specifies polynomial fit | Output polynomial coefficients, provide method to evaluate points | Polynomials are the internal representation; points are derived |
| 7 | "Vehicle center" for lane departure not defined | Use bottom center of frame (fixed ego position assumption) | Simplest deterministic interpretation |
| 8 | Traffic light "state" alert vs detection | Alert only on red/yellow, green is informational only | Safety-conservative: warn on caution/stop |
| 9 | "One active alert at a time" vs GPIO + audio | GPIO and audio are parallel outputs for same alert | Redundancy is safety feature |
| 11 | "5-10 FPS" YOLO vs "≥15 FPS" capture | Frame skipping reconciles both requirements | Explicitly defined skip strategy |
| 14 | IR "vision takes priority" scope | IR can trigger alert only if no collision detected by vision | Vision is primary, IR is supplementary |

---

**Document End**

*This architecture document is implementation-ready and suitable for direct use as a development specification.*
