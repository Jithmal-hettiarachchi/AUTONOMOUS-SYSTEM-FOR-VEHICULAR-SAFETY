"""
Complete Driver Assistant Test - Detection + Alerts

Tests the full pipeline: YOLO detection with danger zone evaluation
and audio/visual alerts on Windows.

Usage:
    python test_full_system.py --video videos/test2.mp4
    python test_full_system.py --video videos/test2.mp4 --skip 3
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.detection import YOLODetector, Detection, DetectionLabel
from src.alerts import AlertType, AlertEvent, AlertDecisionEngine, AudioAlertManager


# Colors (BGR format)
COLORS = {
    DetectionLabel.TRAFFIC_LIGHT_RED: (0, 0, 255),    # Red
    DetectionLabel.TRAFFIC_LIGHT_YELLOW: (0, 255, 255),  # Yellow
    DetectionLabel.TRAFFIC_LIGHT_GREEN: (0, 255, 0),  # Green
    DetectionLabel.TRAFFIC_LIGHT: (0, 200, 200),      # Generic light
    DetectionLabel.STOP_SIGN: (0, 0, 200),            # Dark Red
    DetectionLabel.PEDESTRIAN: (255, 0, 0),           # Blue
    DetectionLabel.VEHICLE: (0, 200, 0),              # Green
    DetectionLabel.BIKER: (255, 165, 0),              # Orange
}

ALERT_COLORS = {
    AlertType.COLLISION_IMMINENT: (0, 0, 255),      # Red
    AlertType.LANE_DEPARTURE_LEFT: (0, 165, 255),   # Orange
    AlertType.LANE_DEPARTURE_RIGHT: (0, 165, 255),  # Orange
    AlertType.TRAFFIC_LIGHT_RED: (0, 0, 255),       # Red
    AlertType.TRAFFIC_LIGHT_YELLOW: (0, 255, 255),  # Yellow
    AlertType.STOP_SIGN: (0, 0, 200),               # Dark red
    AlertType.SYSTEM_WARNING: (128, 128, 128),      # Gray
}


def draw_danger_zone(frame: np.ndarray, polygon: list, alpha: float = 0.3) -> np.ndarray:
    """Draw semi-transparent danger zone overlay."""
    overlay = frame.copy()
    pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(overlay, [pts], (0, 0, 100))  # Dark red fill
    cv2.polylines(frame, [pts], True, (0, 0, 255), 2)  # Red border
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def draw_detections(frame: np.ndarray, detections: list, danger_zone_polygon: list) -> np.ndarray:
    """Draw detection boxes with danger zone highlighting."""
    output = frame.copy()
    
    # Draw danger zone first (background)
    output = draw_danger_zone(output, danger_zone_polygon, alpha=0.2)
    
    for det in detections:
        color = COLORS.get(det.label, (255, 255, 255))
        x1, y1, x2, y2 = map(int, det.bbox)
        
        # Check if in danger zone (highlight with thicker border)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        in_danger = is_point_in_polygon(cx, y2, danger_zone_polygon)  # Check bottom center
        
        thickness = 4 if in_danger else 2
        if in_danger:
            # Draw red highlight behind
            cv2.rectangle(output, (x1-2, y1-2), (x2+2, y2+2), (0, 0, 255), thickness + 2)
        
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        
        # Label
        label = f"{det.class_name}: {det.confidence:.2f}"
        if in_danger:
            label = "⚠ " + label
        
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(output, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(output, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return output


def is_point_in_polygon(x: int, y: int, polygon: list) -> bool:
    """Simple point-in-polygon test."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def draw_alert_banner(frame: np.ndarray, alert: Optional[AlertEvent]) -> np.ndarray:
    """Draw alert banner at top of frame."""
    if alert is None:
        return frame
    
    output = frame.copy()
    h, w = output.shape[:2]
    
    # Get alert color and text
    color = ALERT_COLORS.get(alert.alert_type, (255, 255, 255))
    text = alert.alert_type.display_name
    
    # Draw banner background
    banner_height = 50
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), color, -1)
    output = cv2.addWeighted(overlay, 0.7, output, 0.3, 0)
    
    # Draw text
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    tx = (w - tw) // 2
    ty = (banner_height + th) // 2
    
    # Black outline for readability
    cv2.putText(output, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
    cv2.putText(output, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Draw flashing border for high priority alerts
    if alert.priority == 1:
        # Red flashing border for collision
        if int(time.time() * 4) % 2 == 0:
            cv2.rectangle(output, (0, 0), (w-1, h-1), (0, 0, 255), 8)
    
    return output


def draw_info_panel(frame: np.ndarray, info: dict) -> np.ndarray:
    """Draw info panel in corner."""
    output = frame.copy()
    
    lines = [
        f"Frame: {info.get('frame', 0)}/{info.get('total', 0)}",
        f"Detections: {info.get('detections', 0)}",
        f"Inference: {info.get('latency', 0):.1f}ms",
        f"FPS: {info.get('fps', 0):.1f}",
    ]
    
    if info.get('from_cache'):
        lines[2] = "Inference: cached"
    
    y = 60  # Start below potential alert banner
    for line in lines:
        cv2.putText(output, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(output, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += 25
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Full Driver Assistant Test")
    parser.add_argument("--video", type=str, required=True, help="Video file path")
    parser.add_argument("--model", type=str, default="models/object.onnx", help="ONNX model path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--skip", type=int, default=3, help="YOLO frame skip (1=all, 3=every 3rd)")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio alerts")
    parser.add_argument("--cooldown", type=int, default=1000, help="Alert cooldown in ms")
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.video).exists():
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)
    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)
    
    # Initialize components
    print("=" * 60)
    print("Driver Assistant - Full System Test")
    print("=" * 60)
    
    print(f"\nLoading YOLO model: {args.model}")
    detector = YOLODetector(
        model_path=args.model,
        conf_threshold=args.conf,
        frame_skip=args.skip,
    )
    print(f"  Model loaded: {detector.model_info['output_shape']}")
    
    # Open video to get dimensions
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video: {args.video}")
        sys.exit(1)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo: {args.video}")
    print(f"  Resolution: {frame_width}x{frame_height}")
    print(f"  FPS: {video_fps:.1f}, Frames: {total_frames}")
    
    # Initialize alert system
    alert_engine = AlertDecisionEngine(
        cooldown_ms=args.cooldown,
        confidence_threshold=args.conf,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    
    audio_manager = AudioAlertManager(enabled=not args.no_audio)
    print(f"\nAlert system initialized (audio: {'enabled' if not args.no_audio else 'disabled'})")
    print(f"  Cooldown: {args.cooldown}ms")
    print(f"  Frame skip: {args.skip}")
    
    # Get danger zone for visualization
    danger_zone_polygon = alert_engine.get_danger_zone_polygon()
    
    print("\n" + "=" * 60)
    print("Starting detection... Press 'q' to quit, SPACE to pause")
    print("=" * 60 + "\n")
    
    # Processing loop
    frame_count = 0
    total_latency = 0.0
    inference_count = 0
    alert_count = 0
    fps_timer = time.time()
    fps_frame_count = 0
    current_fps = 0.0
    
    current_alert: Optional[AlertEvent] = None
    alert_display_until = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        fps_frame_count += 1
        
        # Calculate FPS
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            current_fps = fps_frame_count / elapsed
            fps_frame_count = 0
            fps_timer = time.time()
        
        # Run detection
        result = detector.detect(frame)
        
        if not result.from_cache:
            total_latency += result.latency_ms
            inference_count += 1
        
        # Evaluate alerts
        alert = alert_engine.evaluate(result.detections)
        
        if alert:
            current_alert = alert
            alert_display_until = time.time() + 1.5  # Show for 1.5 seconds
            alert_count += 1
            
            # Play audio alert
            audio_manager.play_alert(alert)
            
            # Print alert to console
            print(f"[{frame_count:5d}] ALERT: {alert.alert_type.display_name} "
                  f"(priority={alert.priority}, conf={alert.confidence:.2f})")
        
        # Clear alert display after timeout
        if time.time() > alert_display_until:
            current_alert = None
        
        # Draw frame
        output = draw_detections(frame, result.detections, danger_zone_polygon)
        output = draw_alert_banner(output, current_alert)
        output = draw_info_panel(output, {
            'frame': frame_count,
            'total': total_frames,
            'detections': len(result.detections),
            'latency': result.latency_ms,
            'from_cache': result.from_cache,
            'fps': current_fps,
        })
        
        # Show frame
        cv2.imshow("Driver Assistant", output)
        
        # Handle input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            print("PAUSED - Press any key to continue...")
            cv2.waitKey(0)
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    audio_manager.stop()
    
    # Print summary
    print("\n" + "=" * 60)
    print("Session Summary")
    print("=" * 60)
    print(f"Frames processed: {frame_count}")
    print(f"YOLO inferences: {inference_count}")
    if inference_count > 0:
        avg_latency = total_latency / inference_count
        print(f"Average inference: {avg_latency:.1f}ms")
        print(f"Effective YOLO FPS: {1000/avg_latency:.1f}")
    print(f"Total alerts fired: {alert_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
