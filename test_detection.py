"""
Test script for the detection module.

Tests the YOLODetector class with a video file and displays
detections with visualization.

Usage:
    python test_detection.py --video videos/test2.mp4
    python test_detection.py --video videos/test2.mp4 --model models/object.onnx
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.detection import YOLODetector, Detection, DetectionLabel


# Colors for different detection types (BGR format)
LABEL_COLORS = {
    DetectionLabel.TRAFFIC_LIGHT_RED: (0, 0, 255),    # Red
    DetectionLabel.TRAFFIC_LIGHT_YELLOW: (0, 255, 255),  # Yellow
    DetectionLabel.TRAFFIC_LIGHT_GREEN: (0, 255, 0),  # Green
    DetectionLabel.TRAFFIC_LIGHT: (0, 200, 200),      # Generic light
    DetectionLabel.STOP_SIGN: (0, 0, 200),            # Dark Red
    DetectionLabel.PEDESTRIAN: (255, 0, 0),           # Blue
    DetectionLabel.VEHICLE: (0, 200, 0),              # Green
    DetectionLabel.BIKER: (255, 165, 0),              # Orange
}


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw detection boxes and labels on frame."""
    output = frame.copy()
    
    for det in detections:
        # Get color for this label type
        color = LABEL_COLORS.get(det.label, (255, 255, 255))
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, det.bbox)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        
        # Create label text with class name and confidence
        label_text = f"{det.class_name}: {det.confidence:.2f}"
        
        # Draw label background
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            output,
            (x1, y1 - text_h - baseline - 5),
            (x1 + text_w, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            output,
            label_text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Test detection module")
    parser.add_argument(
        "--video", 
        type=str, 
        required=True,
        help="Path to video file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/object.onnx",
        help="Path to ONNX model file"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold (default: from config.yaml)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=None,
        help="IoU threshold for NMS (default: from config.yaml)",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=None,
        help="Frame skip (1=all frames, 2=every other, etc.)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Load YOLO thresholds/classes from config.yaml (CLI flags override)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    yolo = cfg.yolo
    model_path = args.model if args.model != "models/object.onnx" else yolo.model_path
    conf = yolo.confidence_threshold if args.conf is None else args.conf
    iou = yolo.iou_threshold if args.iou is None else args.iou
    frame_skip = yolo.frame_skip if args.skip is None else args.skip
    
    # Validate paths
    if not Path(args.video).exists():
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)
    
    if not Path(model_path).exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)
    
    # Initialize detector
    print(f"Loading model: {model_path}")
    detector = YOLODetector(
        model_path=model_path,
        conf_threshold=conf,
        iou_threshold=iou,
        frame_skip=frame_skip,
        cache_ttl_ms=yolo.cache_ttl_ms,
        class_names=yolo.classes,
    )
    print(f"Detector: {detector}")
    print(f"Model info: {detector.model_info}")
    
    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video: {args.video}")
        sys.exit(1)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {args.video}")
    print(f"FPS: {fps:.1f}, Total frames: {total_frames}")
    
    # Stats tracking
    frame_count = 0
    total_inference_time = 0.0
    inference_count = 0
    
    print("\nStarting detection... Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run detection
        result = detector.detect(frame)
        
        # Track inference time (only for non-cached results)
        if not result.from_cache:
            total_inference_time += result.latency_ms
            inference_count += 1
        
        # Draw detections
        output = draw_detections(frame, result.detections)
        
        # Draw info overlay
        cache_str = " (cached)" if result.from_cache else ""
        info_text = [
            f"Frame: {frame_count}/{total_frames}",
            f"Detections: {len(result.detections)}{cache_str}",
            f"Inference: {result.latency_ms:.1f}ms" if not result.from_cache else "Inference: cached",
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(
                output,
                text,
                (10, 25 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
        
        # Show frame
        cv2.imshow("Detection Test", output)
        
        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            # Pause on space
            cv2.waitKey(0)
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Print stats
    print("\n--- Detection Stats ---")
    print(f"Total frames processed: {frame_count}")
    print(f"Inference runs: {inference_count}")
    if inference_count > 0:
        avg_inference = total_inference_time / inference_count
        print(f"Average inference time: {avg_inference:.1f}ms")
        print(f"Estimated detection FPS: {1000/avg_inference:.1f}")


if __name__ == "__main__":
    main()
