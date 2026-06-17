"""Distributed laptop AI server for the Driver Assistant project."""

from __future__ import annotations

import argparse
import base64
import binascii
import csv
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import zmq

# Ensure project root is importable when this script is launched directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.alerts import AlertDecisionEngine, AlertType
from src.capture.frame import Frame, FrameSource
from src.config import Config, load_config
from src.detection import Detection, DetectionLabel, YOLODetector
from src.lane import LaneDetectionPipeline, LaneResult
from src.overtake import OvertakeAdvisory, OvertakeStatus
from src.overtake.assistant import OvertakeAssistant, create_overtake_assistant
from src.environment import EnvironmentDetector, prepare_night_yolo_frame


LOGGER = logging.getLogger("laptop_server")
PROTOCOL_VERSION = "1.0"
DEFAULT_BIND = "tcp://*:5555"

# DATA COLLECTION UPDATE: Chapter 4 latency profiling (max rows) and monocular calibration CSV names.
LATENCY_LOG_PATH = PROJECT_ROOT / "latency_analysis_log.csv"
CALIBRATION_LOG_PATH = PROJECT_ROOT / "distance_calibration.csv"
LATENCY_LOG_MAX_FRAMES = 1000


def _to_int(value: Any) -> int:
    if isinstance(value, np.generic):
        return int(value.item())
    return int(value)


def _to_float(value: Any) -> float:
    if isinstance(value, np.generic):
        return float(value.item())
    return float(value)


def _decode_frame_from_payload(request: Dict[str, Any]) -> np.ndarray:
    frame_b64 = request.get("frame")
    if not isinstance(frame_b64, str) or not frame_b64:
        raise ValueError("Request missing valid 'frame' field")
    raw = base64.b64decode(frame_b64, validate=True)
    frame_np = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("JPEG decode failed")
    return frame


def _build_allowed_label_set(config: Config) -> Set[str]:
    return {str(name).strip().lower() for name in config.yolo.classes if str(name).strip()}


def _normalize_label_candidates(label: DetectionLabel) -> Set[str]:
    raw = label.value.lower()
    candidates = {raw, raw.replace("_", ""), raw.replace("_", "-")}
    if "traffic_light" in raw:
        compact = raw.replace("traffic_light", "trafficlight")
        candidates.add(compact)
        candidates.add(compact.replace("_", "-"))
    return candidates


def _is_detection_allowed(det: Detection, allowed: Set[str]) -> bool:
    if not allowed:
        return True
    return bool(_normalize_label_candidates(det.label).intersection(allowed))


def _check_lane_departure(lane_result: LaneResult, frame_width: int) -> Optional[str]:
    if not lane_result.valid or lane_result.left_lane is None or lane_result.right_lane is None:
        return None
    vehicle_x = frame_width / 2
    y_check = lane_result.left_lane.y_range[1] - 20
    left_boundary = lane_result.left_lane.evaluate(y_check)
    right_boundary = lane_result.right_lane.evaluate(y_check)
    margin = 20
    if vehicle_x < left_boundary + margin:
        return "left"
    if vehicle_x > right_boundary - margin:
        return "right"
    return None


def _evaluate_collision_risks(
    detections: List[Detection],
    decision_engine: AlertDecisionEngine,
    frame_shape: Tuple[int, int, int],
) -> List[Detection]:
    h, w = frame_shape[:2]
    risks: List[Detection] = []
    for det in detections:
        if det.label.is_obstacle() and decision_engine.danger_zone.intersects_bbox(det.bbox, w, h):
            risks.append(det)
    return risks


def _build_boxes_payload(detections: List[Detection]) -> List[List[Any]]:
    boxes: List[List[Any]] = []
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        boxes.append([_to_int(round(x1)), _to_int(round(y1)), _to_int(round(x2)), _to_int(round(y2)), det.label.value])
    return boxes


def _build_safe_payload(error: str) -> Dict[str, Any]:
    return {"version": PROTOCOL_VERSION, "alert": "SAFE", "boxes": [], "error": error}


def _init_overtake_assistant(config: Config) -> OvertakeAssistant:
    """Reference-project OvertakeAssistant (lane stability, clearance zone, line type)."""
    return create_overtake_assistant({"overtake_assistant": asdict(config.overtake_assistant)})


def _evaluate_overtake_for_pi(
    overtake_mode: bool,
    assistant: OvertakeAssistant,
    lane_result: LaneResult,
    detections: List[Detection],
    frame: np.ndarray,
) -> Tuple[Optional[str], OvertakeAdvisory]:
    """
    GPIO overtake switch gates evaluation (Pi sends overtake_mode).

    Switch OFF: reset assistant state, no Pi banner (overtake_status omitted).
    Switch ON: run full OvertakeAssistant.evaluate(); map SAFE -> \"SAFE\", else -> \"DANGER\".
    """
    if not overtake_mode:
        assistant.reset()
        return None, OvertakeAdvisory(
            status=OvertakeStatus.DISABLED,
            reason="Overtake mode off",
            clearance_zone=None,
            confidence=0.0,
            vehicles_in_zone=0,
        )

    advisory = assistant.evaluate(
        lane_result=lane_result,
        detections=detections,
        frame_width=frame.shape[1],
        frame_height=frame.shape[0],
        frame=frame,
    )
    if advisory.status == OvertakeStatus.SAFE:
        pi_status = "SAFE"
    else:
        pi_status = "DANGER"
    return pi_status, advisory


def _serialize_clearance_zone(
    zone: Optional[List[Tuple[int, int]]],
) -> List[List[int]]:
    """Polygon [[x, y], ...] for Pi overtake clearance overlay."""
    if not zone or len(zone) != 4:
        return []
    out: List[List[int]] = []
    for pt in zone:
        try:
            out.append([int(pt[0]), int(pt[1])])
        except (TypeError, ValueError, IndexError):
            return []
    return out


def _overtake_zone_color_bgr(advisory: OvertakeAdvisory, config: Config) -> List[int]:
    """BGR border/fill base color from reference overtake_assistant config."""
    ot = config.overtake_assistant
    if advisory.status == OvertakeStatus.SAFE:
        color = ot.zone_color_safe
    elif advisory.status == OvertakeStatus.UNSAFE:
        color = ot.zone_color_unsafe
    else:
        color = ot.zone_color_disabled
    return [int(color[0]), int(color[1]), int(color[2])]


def _lane_points_for_json(lane: Any, num_points: int = 40) -> List[List[int]]:
    """Serialize one lane boundary as [[x, y], ...] for Pi HDMI overlay."""
    if lane is None:
        return []
    try:
        return [[int(x), int(y)] for x, y in lane.get_points(num_points)]
    except Exception:
        return []


def _build_lane_payload(
    lane_result: LaneResult,
    lane_departure: Optional[str],
    lane_pipeline: LaneDetectionPipeline,
    config: Config,
) -> Dict[str, Any]:
    """Lane metadata + polyline points for distributed Pi preview drawing."""
    lc = config.lane_detection
    return {
        "valid": bool(lane_result.valid),
        "partial": bool(lane_result.partial),
        "departure": lane_departure or "none",
        "lane_lost": bool(lane_pipeline.lane_lost),
        "left_points": _lane_points_for_json(lane_result.left_lane),
        "right_points": _lane_points_for_json(lane_result.right_lane),
        "style": {
            "color_bgr": [int(c) for c in lc.lane_boundary_color_bgr],
            "thickness": int(lc.lane_line_thickness),
            "fill_alpha": float(lc.lane_fill_alpha),
            "lane_lost_message": str(lc.lane_lost_message),
        },
    }


def _extract_traffic_light(detections: List[Detection]) -> Dict[str, Any]:
    lights = [d for d in detections if d.label.is_traffic_light()]
    if not lights:
        return {"detected": False, "state": "none"}
    best = max(lights, key=lambda d: d.confidence)
    x1, y1, x2, y2 = best.bbox
    return {
        "detected": True,
        "state": best.label.value,
        "confidence": _to_float(best.confidence),
        "bbox": [_to_int(round(x1)), _to_int(round(y1)), _to_int(round(x2)), _to_int(round(y2))],
    }


def _bgr_for_detection_label(label: DetectionLabel) -> Tuple[int, int, int]:
    """Class-specific box colors (BGR). Collision override is applied separately."""
    return {
        DetectionLabel.TRAFFIC_LIGHT_GREEN: (0, 255, 0),
        DetectionLabel.TRAFFIC_LIGHT_RED: (0, 0, 255),
        DetectionLabel.PEDESTRIAN: (255, 0, 0),
        DetectionLabel.VEHICLE: (0, 255, 255),
    }.get(label, (0, 255, 255))


def _draw_debug(
    frame: np.ndarray,
    detections: List[Detection],
    collision_risks: List[Detection],
    lane_result: LaneResult,
    lane_departure: Optional[str],
    response_alert: str,
    latencies: Dict[str, float],
) -> np.ndarray:
    out = frame.copy()
    collision_color = (0, 0, 255)
    # Detection dataclass is mutable/non-hashable; list membership avoids set() errors.
    risk_list = collision_risks

    def _readable_text_color(bg_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        # OpenCV colors are BGR; convert to rough luminance for contrast.
        b, g, r = bg_color
        luminance = (0.114 * b) + (0.587 * g) + (0.299 * r)
        return (0, 0, 0) if luminance >= 128 else (255, 255, 255)

    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det.bbox]
        is_collision_risk = det in risk_list

        color = _bgr_for_detection_label(det.label)
        thickness = 2
        if is_collision_risk:
            color = collision_color
            thickness = 3

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = f"{det.label.value}:{det.confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        font_thickness = 1
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        text_y = max(label_h + baseline + 2, y1 - 6)
        bg_top_left = (x1, text_y - label_h - baseline - 2)
        bg_bottom_right = (x1 + label_w + 4, text_y + 2)
        cv2.rectangle(out, bg_top_left, bg_bottom_right, color, -1)
        cv2.putText(
            out,
            label,
            (x1 + 2, text_y - baseline),
            font,
            font_scale,
            _readable_text_color(color),
            font_thickness,
            cv2.LINE_AA,
        )

    lane_text = f"Lane valid={lane_result.valid} partial={lane_result.partial}"
    dep_text = f"Departure={lane_departure or 'none'}"
    alert_text = f"Alert={response_alert}"
    metrics = (
        f"decode={latencies['decode_ms']:.1f}ms "
        f"yolo={latencies['inference_ms']:.1f}ms "
        f"lane={latencies['lane_ms']:.1f}ms "
        f"total={latencies['total_ms']:.1f}ms"
    )
    cv2.putText(out, lane_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, dep_text, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, alert_text, (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, metrics, (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def _bbox_pixel_height(det: Detection) -> int:
    """DATA COLLECTION UPDATE: Monocular calibration uses vertical bbox extent in pixels."""
    x1, y1, x2, y2 = det.bbox
    return max(1, _to_int(round(y2 - y1)))


def _select_calibration_vehicle(
    detections: List[Detection],
    collision_risks: List[Detection],
) -> Optional[Tuple[Detection, int]]:
    """
    DATA COLLECTION UPDATE: Pick one vehicle and target_bbox_height for distance_calibration.csv.

    Priority: collision-risk vehicles (largest height h) > highest-confidence VEHICLE >
    largest height h on confidence tie.
    """
    vehicles = [d for d in detections if d.label == DetectionLabel.VEHICLE]
    if not vehicles:
        return None

    risk_vehicles = [d for d in collision_risks if d.label == DetectionLabel.VEHICLE]
    if risk_vehicles:
        target = max(risk_vehicles, key=_bbox_pixel_height)
    else:
        target = max(vehicles, key=lambda d: (float(d.confidence), _bbox_pixel_height(d)))

    return target, _bbox_pixel_height(target)


def _init_latency_logger() -> Tuple[csv.writer, Any, bool, int]:
    """DATA COLLECTION UPDATE: Open latency CSV once; cap at LATENCY_LOG_MAX_FRAMES rows."""
    latency_file = open(LATENCY_LOG_PATH, "w", newline="", encoding="utf-8")
    writer = csv.writer(latency_file)
    writer.writerow(
        [
            "Frame_ID",
            "Capture_ms",
            "Encode_Send_ms",
            "Network_ms",
            "Decode_ms",
            "YOLO_ms",
            "Lane_ms",
            "Decision_ms",
            "Send_ms",
            "GPIO_ms",
            "Total_ms",
        ]
    )
    latency_file.flush()
    LOGGER.info("DATA COLLECTION UPDATE: Latency log started at %s", LATENCY_LOG_PATH)
    return writer, latency_file, True, 0


def _write_latency_row(
    writer: csv.writer,
    file_handle: Any,
    *,
    frame_id: int,
    capture_ms: float,
    encode_send_ms: float,
    network_ms: float,
    decode_ms: float,
    yolo_ms: float,
    lane_ms: float,
    decision_ms: float,
    send_ms: float,
    gpio_ms: float,
    total_ms: float,
) -> None:
    """DATA COLLECTION UPDATE: Append one end-to-end latency sample; flush to avoid disk write stalls."""
    writer.writerow(
        [
            frame_id,
            round(capture_ms, 3),
            round(encode_send_ms, 3),
            round(network_ms, 3),
            round(decode_ms, 3),
            round(yolo_ms, 3),
            round(lane_ms, 3),
            round(decision_ms, 3),
            round(send_ms, 3),
            round(gpio_ms, 3),
            round(total_ms, 3),
        ]
    )
    file_handle.flush()


def _try_write_latency_row(
    writer: csv.writer,
    file_handle: Any,
    *,
    latency_report: Any,
    laptop_metrics: Optional[Dict[str, Any]],
    latency_logging_enabled: bool,
    latency_rows_written: int,
) -> int:
    """Merge Pi latency_report with stored laptop metrics and append one CSV row."""
    if not latency_logging_enabled or latency_rows_written >= LATENCY_LOG_MAX_FRAMES:
        return latency_rows_written
    if not isinstance(latency_report, dict) or not isinstance(laptop_metrics, dict):
        return latency_rows_written
    if latency_report.get("frame_id") != laptop_metrics.get("frame_id"):
        return latency_rows_written

    report_frame_id = int(latency_report.get("frame_id", 0))

    decode_ms = _to_float(laptop_metrics.get("decode_ms", 0.0))
    yolo_ms = _to_float(laptop_metrics.get("yolo_ms", 0.0))
    lane_ms = _to_float(laptop_metrics.get("lane_ms", 0.0))
    decision_ms = _to_float(laptop_metrics.get("decision_ms", 0.0))
    send_ms = _to_float(laptop_metrics.get("send_ms", 0.0))
    zmq_wait_ms = _to_float(latency_report.get("zmq_wait_ms", 0.0))
    laptop_sum_ms = decode_ms + yolo_ms + lane_ms + decision_ms + send_ms
    network_ms = max(0.0, zmq_wait_ms - laptop_sum_ms)

    _write_latency_row(
        writer,
        file_handle,
        frame_id=report_frame_id,
        capture_ms=_to_float(latency_report.get("capture_ms", 0.0)),
        encode_send_ms=_to_float(latency_report.get("encode_send_ms", 0.0)),
        network_ms=network_ms,
        decode_ms=decode_ms,
        yolo_ms=yolo_ms,
        lane_ms=lane_ms,
        decision_ms=decision_ms,
        send_ms=send_ms,
        gpio_ms=_to_float(latency_report.get("gpio_ms", 0.0)),
        total_ms=_to_float(latency_report.get("total_ms", 0.0)),
    )
    return latency_rows_written + 1


def _init_calibration_logger() -> Tuple[csv.writer, Any]:
    """DATA COLLECTION UPDATE: Append-only calibration log; header only when file is empty."""
    write_header = not CALIBRATION_LOG_PATH.exists() or CALIBRATION_LOG_PATH.stat().st_size == 0
    calibration_file = open(CALIBRATION_LOG_PATH, "a", newline="", encoding="utf-8")
    writer = csv.writer(calibration_file)
    if write_header:
        writer.writerow(["Actual_Distance_Meters", "Bounding_Box_Height_Pixels"])
        calibration_file.flush()
    LOGGER.info("DATA COLLECTION UPDATE: Calibration log at %s", CALIBRATION_LOG_PATH)
    return writer, calibration_file


def _handle_distance_calibration_save(
    calibration_writer: csv.writer,
    calibration_file: Any,
    detections: List[Detection],
    collision_risks: List[Detection],
    debug_frame: np.ndarray,
    window_name: str,
) -> None:
    """
    DATA COLLECTION UPDATE: 's' key — freeze preview, terminal input, append calibration row.
    """
    selection = _select_calibration_vehicle(detections, collision_risks)
    if selection is None:
        print("DATA COLLECTION UPDATE: Warning — no vehicle bbox found; calibration skipped.")
        return

    _target, target_bbox_height = selection

    frozen = debug_frame.copy()
    cv2.putText(
        frozen,
        "CALIBRATION PAUSED - Check terminal",
        (10, frozen.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow(window_name, frozen)
    cv2.waitKey(1)  # DATA COLLECTION UPDATE: Force frozen overlay to render before input().

    try:
        raw = input(">>> Enter the physical distance in meters: ").strip()
        distance_m = float(raw)
    except ValueError:
        print("DATA COLLECTION UPDATE: Invalid distance; enter a numeric value in meters.")
        return

    calibration_writer.writerow([round(distance_m, 4), target_bbox_height])
    calibration_file.flush()
    print(
        "DATA COLLECTION UPDATE: Saved calibration "
        f"distance={distance_m:.4f} m, bbox_height={target_bbox_height} px -> {CALIBRATION_LOG_PATH}"
    )


def configure_logging(verbose: bool, level_name: str) -> None:
    level = logging.DEBUG if verbose else getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def _init_models(config: Config, model_override: Optional[str]) -> Tuple[YOLODetector, LaneDetectionPipeline, AlertDecisionEngine]:
    model_path = Path(model_override) if model_override else (PROJECT_ROOT / config.yolo.model_path)
    detector = YOLODetector(
        model_path=str(model_path),
        conf_threshold=config.yolo.confidence_threshold,
        iou_threshold=config.yolo.iou_threshold,
        input_size=config.yolo.input_width,
        frame_skip=config.yolo.frame_skip,
        cache_ttl_ms=config.yolo.cache_ttl_ms,
        class_names=config.yolo.classes,
    )
    lane_pipeline = LaneDetectionPipeline(config.lane_detection)
    decision_engine = AlertDecisionEngine(
        cooldown_ms=config.alerts.cooldown_ms,
        traffic_light_cooldown_ms=config.alerts.traffic_light_cooldown_ms,
        confidence_threshold=config.yolo.confidence_threshold,
        frame_width=config.capture.resolution[0],
        frame_height=config.capture.resolution[1],
        danger_zone_config=config.danger_zone,
        lidar_required=False,
    )
    return detector, lane_pipeline, decision_engine


def run_server(bind_addr: str, config: Config, model_override: Optional[str]) -> None:
    detector, lane_pipeline, decision_engine = _init_models(config, model_override)
    overtake_assistant = _init_overtake_assistant(config)
    environment_detector: Optional[EnvironmentDetector] = None
    if config.environment_detection.enabled:
        try:
            environment_detector = EnvironmentDetector(config.environment_detection)
            LOGGER.info("Environment detector initialized (laptop server)")
        except Exception as exc:
            LOGGER.warning("Environment detector setup failed: %s", exc)
    night_yolo_active = False
    manual_night_mode = False
    manual_rain_mode = False
    manual_fog_mode = False
    last_key = -1
    if overtake_assistant.config.enabled:
        LOGGER.info(
            "Overtake assistant active when Pi switch ON (traffic_side=%s)",
            overtake_assistant.config.traffic_side,
        )
    else:
        LOGGER.info("Overtake assistant disabled in config")
    allowed_labels = _build_allowed_label_set(config)

    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, max(200, config.capture.timeout_ms))
    socket.setsockopt(zmq.SNDTIMEO, max(200, config.capture.timeout_ms))
    socket.bind(bind_addr)
    LOGGER.info("Laptop AI server listening on %s", bind_addr)

    # DATA COLLECTION UPDATE: Chapter 4 CSV loggers (handles kept open for the session).
    latency_writer = None
    latency_file = None
    latency_logging_enabled = False
    latency_rows_written = 0
    calibration_writer, calibration_file = _init_calibration_logger()
    if not config.display.enabled:
        LOGGER.info(
            "DATA COLLECTION UPDATE: Press 's' in the debug window to save distance_calibration.csv "
            "(enable display in config)."
        )

    frame_seq = 0
    prev_laptop_latency: Optional[Dict[str, Any]] = None
    # DATA COLLECTION UPDATE: Last successful frame context for 's' calibration key.
    last_calibration_detections: List[Detection] = []
    last_calibration_collision_risks: List[Detection] = []
    try:
        while True:
            if last_key in (ord("n"), ord("N")):
                manual_night_mode = not manual_night_mode
                LOGGER.info(
                    "Manual night mode %s (press N to toggle)",
                    "ON" if manual_night_mode else "OFF",
                )
            elif last_key in (ord("r"), ord("R")):
                manual_rain_mode = not manual_rain_mode
                LOGGER.info(
                    "Manual rain mode %s (press R to toggle)",
                    "ON" if manual_rain_mode else "OFF",
                )
            elif last_key in (ord("f"), ord("F")):
                manual_fog_mode = not manual_fog_mode
                LOGGER.info(
                    "Manual fog mode %s (press F to toggle)",
                    "ON" if manual_fog_mode else "OFF",
                )
            elif last_key in (ord("l"), ord("L")):
                if not latency_logging_enabled:
                    if latency_file is not None:
                        try:
                            latency_file.close()
                        except Exception:
                            pass
                    latency_writer, latency_file, latency_logging_enabled, latency_rows_written = _init_latency_logger()
                else:
                    latency_logging_enabled = False
                    if latency_file is not None:
                        try:
                            latency_file.close()
                        except Exception:
                            pass
                        latency_file = None
                    LOGGER.info(
                        "Latency collection STOPPED at %d frames (press L to start again)",
                        latency_rows_written,
                    )

            cycle_start = time.perf_counter()
            decode_ms = 0.0
            lane_ms = 0.0
            inference_ms = 0.0
            decision_ms = 0.0
            send_ms = 0.0
            response: Dict[str, Any]
            debug_frame: Optional[np.ndarray] = None
            frame_processed_ok = False

            try:
                request = socket.recv_json()
                latency_rows_written = _try_write_latency_row(
                    latency_writer,
                    latency_file,
                    latency_report=request.get("latency_report"),
                    laptop_metrics=prev_laptop_latency,
                    latency_logging_enabled=latency_logging_enabled,
                    latency_rows_written=latency_rows_written,
                )
                if (
                    latency_logging_enabled
                    and latency_rows_written >= LATENCY_LOG_MAX_FRAMES
                ):
                    latency_logging_enabled = False
                    print(
                        "DATA COLLECTION UPDATE: Latency log complete (1000 frames). "
                        f"File: {LATENCY_LOG_PATH}"
                    )
                overtake_mode = bool(request.get("overtake_mode", False))
                decode_start = time.perf_counter()
                frame = _decode_frame_from_payload(request)
                decode_ms = (time.perf_counter() - decode_start) * 1000.0

                if tuple(config.capture.resolution) != (frame.shape[1], frame.shape[0]):
                    frame = cv2.resize(frame, tuple(config.capture.resolution), interpolation=cv2.INTER_AREA)

                yolo_frame = frame
                env_cfg = config.environment_detection
                env_result = None
                if env_cfg.enabled and environment_detector is not None:
                    try:
                        env_result = environment_detector.update(frame)
                    except Exception as exc:
                        LOGGER.warning("Environment detector update failed: %s", exc)

                is_night = (bool(env_result.is_night) if env_result else False) or manual_night_mode
                is_fog = (bool(env_result.is_fog) if env_result else False) or manual_fog_mode
                is_rain = (bool(env_result.is_rain) if env_result else False) or manual_rain_mode

                if env_cfg.enabled and env_cfg.night_yolo_boost_enabled:
                    night_conf = (
                        float(env_result.confidence.get("night", 0.0))
                        if env_result is not None
                        else 0.0
                    )
                    yolo_frame, night_yolo_active, _active_conf = prepare_night_yolo_frame(
                        frame,
                        is_night=bool(env_result.is_night) if env_result else False,
                        env_cfg=env_cfg,
                        day_confidence=config.yolo.confidence_threshold,
                        detector=detector,
                        decision_engine=decision_engine,
                        previous_night_yolo_active=night_yolo_active,
                        night_confidence=night_conf,
                        manual_night=manual_night_mode,
                    )

                det_result = detector.detect(yolo_frame)
                inference_ms = 0.0 if det_result.from_cache else _to_float(det_result.latency_ms)
                detections = [d for d in det_result.detections if _is_detection_allowed(d, allowed_labels)]

                frame_obj = Frame(
                    data=frame,
                    timestamp=time.monotonic(),
                    sequence=frame_seq,
                    source=FrameSource.IP_CAMERA,
                )
                lane_start = time.perf_counter()
                lane_result = lane_pipeline.process(frame_obj)
                lane_ms = (time.perf_counter() - lane_start) * 1000.0
                frame_seq += 1

                decision_start = time.perf_counter()
                lane_departure = _check_lane_departure(lane_result, frame.shape[1])
                if lane_result.valid:
                    decision_engine.danger_zone.update_from_lanes(
                        lane_result.left_lane,
                        lane_result.right_lane,
                        frame.shape[1],
                        frame.shape[0],
                    )
                collision_risks = _evaluate_collision_risks(detections, decision_engine, frame.shape)
                alert_event = decision_engine.evaluate(
                    detections=detections,
                    lane_departure=lane_departure,
                    lane_result=lane_result,
                )
                overtake_status, overtake_advisory = _evaluate_overtake_for_pi(
                    overtake_mode=overtake_mode,
                    assistant=overtake_assistant,
                    lane_result=lane_result,
                    detections=detections,
                    frame=frame,
                )
                overtake_reason = overtake_advisory.reason
                vehicles_in_zone = int(overtake_advisory.vehicles_in_zone)
                overtake_nested_status = (
                    overtake_advisory.status.value
                    if overtake_mode
                    else "disabled"
                )
                decision_ms = (time.perf_counter() - decision_start) * 1000.0

                danger_objects = [
                    {
                        "label": det.label.value,
                        "bbox": [int(round(det.bbox[0])), int(round(det.bbox[1])), int(round(det.bbox[2])), int(round(det.bbox[3]))],
                        "confidence": _to_float(det.confidence),
                        "danger": det in collision_risks,
                    }
                    for det in detections
                ]
                alert_type_value = alert_event.alert_type.value if alert_event is not None else "none"
                alert_is_danger = bool(collision_risks) or (
                    alert_event is not None
                    and alert_event.alert_type in (
                        AlertType.COLLISION_IMMINENT,
                        AlertType.LANE_DEPARTURE_LEFT,
                        AlertType.LANE_DEPARTURE_RIGHT,
                    )
                )
                response_alert = "DANGER" if alert_is_danger else "SAFE"
                total_ms = (time.perf_counter() - cycle_start) * 1000.0
                frame_processed_ok = True
                current_frame_id = int(request.get("frame_id") or frame_seq)
                prev_laptop_latency = {
                    "frame_id": current_frame_id,
                    "decode_ms": decode_ms,
                    "yolo_ms": inference_ms,
                    "lane_ms": lane_ms,
                    "decision_ms": decision_ms,
                    "send_ms": 0.0,
                }

                last_calibration_detections = detections
                last_calibration_collision_risks = collision_risks

                response = {
                    "version": PROTOCOL_VERSION,
                    "alert": response_alert,
                    "boxes": _build_boxes_payload(detections),
                    "latency_ms": _to_float(total_ms),
                    "decode_ms": _to_float(decode_ms),
                    "inference_ms": _to_float(inference_ms),
                    "lane_ms": _to_float(lane_ms),
                    "total_ms": _to_float(total_ms),
                    "traffic_light": _extract_traffic_light(detections),
                    "lane": _build_lane_payload(
                        lane_result,
                        lane_departure,
                        lane_pipeline,
                        config,
                    ),
                    "alert_type": alert_type_value,
                    "overtake": {
                        "status": overtake_nested_status,
                        "reason": overtake_reason,
                        "confidence": _to_float(overtake_advisory.confidence),
                        "vehicles_in_zone": vehicles_in_zone,
                        "clearance_zone": _serialize_clearance_zone(
                            overtake_advisory.clearance_zone
                        ),
                        "zone_color_bgr": _overtake_zone_color_bgr(
                            overtake_advisory, config
                        ),
                        "zone_fill_alpha": 0.3,
                    },
                    "collision": {
                        "risk": bool(collision_risks),
                        "count": int(len(collision_risks)),
                        "labels": [d.label.value for d in collision_risks],
                    },
                    "danger_objects": danger_objects,
                }
                if overtake_mode and overtake_status is not None:
                    response["overtake_status"] = overtake_status

                if config.display.enabled:
                    debug_frame = _draw_debug(
                        frame=frame,
                        detections=detections,
                        collision_risks=collision_risks,
                        lane_result=lane_result,
                        lane_departure=lane_departure,
                        response_alert=response_alert,
                        latencies={
                            "decode_ms": decode_ms,
                            "inference_ms": inference_ms,
                            "lane_ms": lane_ms,
                            "total_ms": total_ms,
                        },
                    )

            except zmq.error.Again:
                continue
            except (ValueError, KeyError, TypeError, binascii.Error) as exc:
                LOGGER.warning("Bad request payload: %s", exc)
                response = _build_safe_payload("bad_frame_payload")
            except cv2.error as exc:
                LOGGER.warning("OpenCV processing error: %s", exc)
                response = _build_safe_payload("opencv_error")
            except Exception as exc:
                LOGGER.exception("Unhandled server error: %s", exc)
                response = _build_safe_payload("server_error")

            try:
                send_start = time.perf_counter()
                socket.send_json(response)
                send_ms = (time.perf_counter() - send_start) * 1000.0
                if frame_processed_ok and prev_laptop_latency is not None:
                    prev_laptop_latency["send_ms"] = send_ms
            except zmq.error.Again:
                LOGGER.warning("Send timeout while replying to Pi")
            except Exception as exc:
                LOGGER.error("Failed to send response: %s", exc)

            if config.display.enabled and debug_frame is not None:
                cv2.imshow(config.display.window_name, debug_frame)
                last_key = cv2.waitKey(1) & 0xFF
                if last_key == ord("q"):
                    LOGGER.info("Quit requested from laptop debug window")
                    break
                # DATA COLLECTION UPDATE: 's' saves ground-truth distance vs bbox height for monocular calibration.
                if last_key == ord("s") and frame_processed_ok:
                    _handle_distance_calibration_save(
                        calibration_writer,
                        calibration_file,
                        last_calibration_detections,
                        last_calibration_collision_risks,
                        debug_frame,
                        config.display.window_name,
                    )
    finally:
        # DATA COLLECTION UPDATE: Close Chapter 4 CSV handles with the ZMQ socket.
        try:
            if latency_file is not None:
                latency_file.close()
        except Exception:
            pass
        try:
            calibration_file.close()
        except Exception:
            pass
        socket.close(0)
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed Laptop Brain server")
    parser.add_argument("--bind", default=DEFAULT_BIND, help="ZMQ bind address (default: tcp://*:5555)")
    parser.add_argument("--config", default=None, help="Optional config path")
    parser.add_argument("--model-path", default=None, help="Optional model override")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    configure_logging(args.verbose, config.system.log_level)
    run_server(bind_addr=args.bind, config=config, model_override=args.model_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.info("Laptop server stopped by user")
