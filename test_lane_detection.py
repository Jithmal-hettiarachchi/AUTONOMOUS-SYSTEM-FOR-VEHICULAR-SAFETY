"""
Lane Detection Module Test Script

This script tests the lane detection pipeline with:
1. A webcam feed (live testing)
2. A video file (if provided)
3. A synthetic test image

Run with:
    python test_lane_detection.py                    # Test with webcam
    python test_lane_detection.py --video path.mp4  # Test with video file
    python test_lane_detection.py --synthetic       # Test with synthetic image
    python test_lane_detection.py --synthetic --headless   # No GUI (CI / SSH)

Press 'q' to quit, 's' to save current frame.
"""

import sys
import argparse
import time
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, '.')

from src.config import load_config, LaneDetectionConfig
from src.lane.pipeline import LaneDetectionPipeline
from src.lane.result import LaneResult
from src.capture.frame import Frame, FrameSource


def create_synthetic_road_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a synthetic road image with lane markings for testing."""
    # Create dark gray road background
    image = np.ones((height, width, 3), dtype=np.uint8) * 80
    
    # Draw sky (top portion)
    image[:height//2, :] = [180, 200, 220]  # Light blue sky
    
    # Draw road (bottom portion) 
    road_top = height // 2
    pts = np.array([
        [0, height],
        [width, height],
        [int(width * 0.7), road_top],
        [int(width * 0.3), road_top]
    ], np.int32)
    cv2.fillPoly(image, [pts], (60, 60, 60))
    
    # Draw left lane line (white, dashed)
    for i in range(5):
        y1 = height - i * 60 - 20
        y2 = height - i * 60 - 50
        if y1 < road_top or y2 < road_top:
            break
        
        # Calculate x positions based on perspective
        progress1 = (height - y1) / (height - road_top)
        progress2 = (height - y2) / (height - road_top)
        
        x1 = int(width * 0.15 + progress1 * (width * 0.35 - width * 0.15))
        x2 = int(width * 0.15 + progress2 * (width * 0.35 - width * 0.15))
        
        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 4)
    
    # Draw right lane line (yellow, solid)
    for i in range(20):
        y1 = height - i * 15
        y2 = height - (i + 1) * 15
        if y1 < road_top or y2 < road_top:
            break
        
        progress1 = (height - y1) / (height - road_top)
        progress2 = (height - y2) / (height - road_top)
        
        x1 = int(width * 0.85 - progress1 * (width * 0.85 - width * 0.65))
        x2 = int(width * 0.85 - progress2 * (width * 0.85 - width * 0.65))
        
        cv2.line(image, (x1, y1), (x2, y2), (0, 200, 255), 4)
    
    return image


def draw_lane_overlay(image: np.ndarray, result: LaneResult, roi_vertices: np.ndarray = None) -> np.ndarray:
    """Draw lane detection results on the image."""
    overlay = image.copy()
    height, width = image.shape[:2]
    
    # Draw ROI region (semi-transparent)
    if roi_vertices is not None:
        roi_overlay = image.copy()
        cv2.polylines(roi_overlay, [roi_vertices], True, (255, 255, 0), 2)
        cv2.addWeighted(roi_overlay, 0.3, overlay, 0.7, 0, overlay)
    
    # Draw lanes if detected
    if result.left_lane is not None:
        points = result.left_lane.get_points(30)
        if len(points) >= 2:
            pts = np.array(points, np.int32)
            cv2.polylines(overlay, [pts], False, (0, 255, 0), 3)
    
    if result.right_lane is not None:
        points = result.right_lane.get_points(30)
        if len(points) >= 2:
            pts = np.array(points, np.int32)
            cv2.polylines(overlay, [pts], False, (0, 255, 0), 3)
    
    # Draw lane area if both lanes detected
    if result.valid and result.left_lane and result.right_lane:
        left_pts = result.left_lane.get_points(20)
        right_pts = result.right_lane.get_points(20)
        
        # Create polygon for lane area
        lane_poly = np.array(left_pts + right_pts[::-1], np.int32)
        lane_overlay = overlay.copy()
        cv2.fillPoly(lane_overlay, [lane_poly], (0, 100, 0))
        cv2.addWeighted(lane_overlay, 0.3, overlay, 0.7, 0, overlay)
    
    # Draw status text
    status_color = (0, 255, 0) if result.valid else (0, 165, 255) if result.partial else (0, 0, 255)
    status_text = "VALID" if result.valid else "PARTIAL" if result.partial else "INVALID"
    
    cv2.putText(overlay, f"Lane: {status_text}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(overlay, f"Latency: {result.latency_ms:.1f}ms", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    if result.left_lane:
        cv2.putText(overlay, f"Left conf: {result.left_lane.confidence:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    if result.right_lane:
        cv2.putText(overlay, f"Right conf: {result.right_lane.confidence:.2f}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Draw lane center if valid
    if result.valid:
        center_y = int(height * 0.8)
        lane_center = result.get_lane_center(center_y)
        if lane_center:
            frame_center = width // 2
            cv2.circle(overlay, (int(lane_center), center_y), 8, (255, 0, 255), -1)
            cv2.line(overlay, (frame_center, center_y - 20), (frame_center, center_y + 20), (255, 0, 0), 2)
            
            offset = lane_center - frame_center
            direction = "LEFT" if offset < -20 else "RIGHT" if offset > 20 else "CENTER"
            cv2.putText(overlay, f"Position: {direction} ({offset:.0f}px)", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return overlay


def test_with_webcam(pipeline: LaneDetectionPipeline, show_xray: bool = False):
    """Test lane detection with webcam feed."""
    print("\n=== Testing with Webcam ===")
    print("Press 'q' to quit, 's' to save frame")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    fps_time = time.monotonic()
    fps = 0
    
    try:
        while True:
            ret, frame_data = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Create Frame object
            frame = Frame(
                data=frame_data,
                timestamp=time.monotonic(),
                sequence=frame_count,
                source=FrameSource.WEBCAM
            )
            
            # Process through lane detection
            result = pipeline.process(frame)
            
            # Draw overlay
            display = draw_lane_overlay(frame_data, result, pipeline.get_roi_vertices())
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                now = time.monotonic()
                fps = 30 / (now - fps_time)
                fps_time = now
            
            cv2.putText(display, f"FPS: {fps:.1f}", (540, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Lane Detection Test - Webcam", display)
            if show_xray:
                xray = pipeline.get_debug_frame()
                cv2.imshow("Lane X-Ray (2x2)", xray)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"lane_capture_{frame_count}.png"
                cv2.imwrite(filename, display)
                print(f"Saved: {filename}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


def test_with_video(pipeline: LaneDetectionPipeline, video_path: str, show_xray: bool = False):
    """Test lane detection with video file."""
    print(f"\n=== Testing with Video: {video_path} ===")
    print("Press 'q' to quit, 's' to save frame, SPACE to pause")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        return
    
    frame_count = 0
    paused = False
    
    try:
        while True:
            if not paused:
                ret, frame_data = cap.read()
                if not ret:
                    print("End of video, looping...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                # Resize if needed
                if frame_data.shape[1] != 640 or frame_data.shape[0] != 480:
                    frame_data = cv2.resize(frame_data, (640, 480))
                
                # Create Frame object
                frame = Frame(
                    data=frame_data,
                    timestamp=time.monotonic(),
                    sequence=frame_count,
                    source=FrameSource.VIDEO_FILE
                )
                
                # Process through lane detection
                result = pipeline.process(frame)
                
                # Draw overlay
                display = draw_lane_overlay(frame_data, result, pipeline.get_roi_vertices())
                frame_count += 1
            
            cv2.putText(display, f"Frame: {frame_count}", (540, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if paused:
                cv2.putText(display, "PAUSED", (280, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            cv2.imshow("Lane Detection Test - Video", display)
            if show_xray:
                xray = pipeline.get_debug_frame()
                cv2.imshow("Lane X-Ray (2x2)", xray)
            
            key = cv2.waitKey(30 if not paused else 0) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"lane_video_{frame_count}.png"
                cv2.imwrite(filename, display)
                print(f"Saved: {filename}")
            elif key == ord(' '):
                paused = not paused
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


def test_with_synthetic(
    pipeline: LaneDetectionPipeline,
    show_xray: bool = False,
    headless: bool = False,
):
    """Test lane detection with synthetic image."""
    print("\n=== Testing with Synthetic Road Image ===")
    if not headless:
        print("Press any key to close, 's' to save")
    
    # Create synthetic road
    synthetic = create_synthetic_road_image(640, 480)
    
    # Create Frame object
    frame = Frame(
        data=synthetic,
        timestamp=time.monotonic(),
        sequence=0,
        source=FrameSource.VIDEO_FILE
    )
    
    # Process
    result = pipeline.process(frame)
    
    # Print results
    print(f"\nResults:")
    print(f"  Valid: {result.valid}")
    print(f"  Partial: {result.partial}")
    print(f"  Latency: {result.latency_ms:.2f} ms")
    print(f"  Left lane: {'Detected' if result.left_lane else 'Not detected'}")
    print(f"  Right lane: {'Detected' if result.right_lane else 'Not detected'}")
    
    if result.left_lane:
        print(f"    Left confidence: {result.left_lane.confidence:.2f}")
        print(f"    Left coefficients: {result.left_lane.coefficients}")
    
    if result.right_lane:
        print(f"    Right confidence: {result.right_lane.confidence:.2f}")
        print(f"    Right coefficients: {result.right_lane.coefficients}")
    
    # Draw overlay
    display = draw_lane_overlay(synthetic, result, pipeline.get_roi_vertices())
    
    # Show original and result side by side
    combined = np.hstack([synthetic, display])
    if headless:
        print("(headless: skipping GUI; writing lane_synthetic_test.png)")
        cv2.imwrite("lane_synthetic_test.png", combined)
        if show_xray:
            cv2.imwrite("lane_synthetic_xray.png", pipeline.get_debug_frame())
            print("Saved: lane_synthetic_xray.png")
    else:
        cv2.imshow("Synthetic Test: Original | Detection", combined)
        if show_xray:
            xray = pipeline.get_debug_frame()
            cv2.imshow("Lane X-Ray (2x2)", xray)
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('s'):
                cv2.imwrite("lane_synthetic_test.png", combined)
                print("Saved: lane_synthetic_test.png")
            else:
                break
        
        cv2.destroyAllWindows()
    
    return result.valid


def main():
    parser = argparse.ArgumentParser(description="Test Lane Detection Module")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--synthetic", action="store_true", help="Test with synthetic image")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument(
        "--xray",
        action="store_true",
        help="Show 2x2 pipeline debug grid (get_debug_frame) alongside main view",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Synthetic mode: no windows; save PNG(s) and exit (for CI / no display)",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("LANE DETECTION MODULE TEST")
    print("=" * 60)
    
    # Load configuration
    try:
        config = load_config(args.config)
        print(f"Configuration loaded successfully")
    except Exception as e:
        print(f"Using default configuration: {e}")
        config = load_config(None)
    
    # Create pipeline
    print("Initializing lane detection pipeline...")
    pipeline = LaneDetectionPipeline(config.lane_detection)
    print("Pipeline initialized!")
    
    # Run tests
    if args.synthetic:
        success = test_with_synthetic(
            pipeline,
            show_xray=args.xray,
            headless=args.headless,
        )
        print(f"\nSynthetic test: {'PASSED' if success else 'FAILED'}")
    elif args.video:
        test_with_video(pipeline, args.video, show_xray=args.xray)
    else:
        # Default: try synthetic first, then webcam
        print("\nRunning synthetic test first...")
        success = test_with_synthetic(
            pipeline,
            show_xray=args.xray,
            headless=args.headless,
        )
        print(f"\nSynthetic test: {'PASSED' if success else 'FAILED'}")
        
        response = input("\nTest with webcam? (y/n): ").strip().lower()
        if response == 'y':
            pipeline.reset()  # Reset temporal state
            test_with_webcam(pipeline, show_xray=args.xray)
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
