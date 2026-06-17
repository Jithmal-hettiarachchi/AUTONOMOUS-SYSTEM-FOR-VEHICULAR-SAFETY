"""
Simple ONNX Model Test Script

Tests a YOLO ONNX model to verify:
1. Model loads correctly
2. Output shape is correct for COCO (80 classes)
3. Detections work on a test image/video

Usage:
    python test_onnx_model.py
    python test_onnx_model.py --model models/object.onnx
    python test_onnx_model.py --model models/object.onnx --image test.jpg
    python test_onnx_model.py --model models/object.onnx --video test.mp4
"""

import argparse
import cv2
import numpy as np

try:
    import onnxruntime as ort
    print(f"ONNX Runtime version: {ort.__version__}")
except ImportError:
    print("ERROR: onnxruntime not installed. Run: pip install onnxruntime")
    exit(1)


# COCO class names (80 classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]


def load_model(model_path: str):
    """Load ONNX model and print info."""
    print(f"\n{'='*60}")
    print(f"Loading model: {model_path}")
    print(f"{'='*60}")
    
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    
    # Print input info
    print("\nModel Inputs:")
    for inp in session.get_inputs():
        print(f"  Name: {inp.name}")
        print(f"  Shape: {inp.shape}")
        print(f"  Type: {inp.type}")
    
    # Print output info
    print("\nModel Outputs:")
    for out in session.get_outputs():
        print(f"  Name: {out.name}")
        print(f"  Shape: {out.shape}")
        print(f"  Type: {out.type}")
    
    return session


def preprocess(image: np.ndarray, input_size: int = 640):
    """Preprocess image for YOLO inference."""
    h, w = image.shape[:2]
    
    # Calculate scale to fit in input_size while maintaining aspect ratio
    scale = min(input_size / w, input_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create padded image (letterbox)
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    pad_x = (input_size - new_w) // 2
    pad_y = (input_size - new_h) // 2
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    
    # Convert BGR to RGB, normalize, transpose to NCHW
    blob = padded[:, :, ::-1].astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)  # HWC -> CHW
    blob = np.expand_dims(blob, 0)  # Add batch dimension
    
    return blob, scale, (pad_x, pad_y)


def compute_iou(box1, boxes):
    """Compute IoU between one box and array of boxes."""
    x1 = np.maximum(box1[0], boxes[:, 0])
    y1 = np.maximum(box1[1], boxes[:, 1])
    x2 = np.minimum(box1[2], boxes[:, 2])
    y2 = np.minimum(box1[3], boxes[:, 3])
    
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    return inter / (area1 + areas - inter + 1e-6)


def nms(boxes, scores, iou_threshold=0.45):
    """Non-Maximum Suppression."""
    if len(boxes) == 0:
        return []
    
    order = np.argsort(scores)[::-1]
    keep = []
    
    while len(order) > 0:
        idx = order[0]
        keep.append(idx)
        
        if len(order) == 1:
            break
        
        ious = compute_iou(boxes[idx], boxes[order[1:]])
        order = order[1:][ious < iou_threshold]
    
    return keep


def postprocess(output: np.ndarray, conf_threshold: float = 0.25, iou_threshold: float = 0.45, verbose: bool = True):
    """
    Process YOLO output with detailed debugging.
    """
    print(f"\n{'='*40}")
    print("OUTPUT ANALYSIS")
    print(f"{'='*40}")
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output min: {output.min():.4f}, max: {output.max():.4f}")
    
    # Handle different shapes
    if len(output.shape) == 3:
        batch, dim1, dim2 = output.shape
        print(f"Batch={batch}, Dim1={dim1}, Dim2={dim2}")
        
        # Determine format
        if dim1 < dim2:
            # (1, 84, 8400) format - features x predictions
            print(f"Format: (batch, {dim1} features, {dim2} predictions)")
            data = output[0].T  # -> (8400, 84)
        else:
            # (1, 8400, 84) format - predictions x features
            print(f"Format: (batch, {dim1} predictions, {dim2} features)")
            data = output[0]
        
        print(f"Data shape after transpose: {data.shape}")
        
        # Check first few rows to understand format
        print(f"\nFirst 3 predictions (raw):")
        for i in range(min(3, len(data))):
            print(f"  [{i}]: {data[i][:10]}...")  # First 10 values
        
        # Check if first 4 values look like boxes (should be ~0-640 for coordinates)
        boxes_sample = data[:5, :4]
        print(f"\nFirst 5 box values (cols 0-3):")
        print(f"  {boxes_sample}")
        print(f"  Box range: min={boxes_sample.min():.2f}, max={boxes_sample.max():.2f}")
        
        # Check scores (should be 0-1)
        scores_sample = data[:, 4:]
        print(f"\nScore columns shape: {scores_sample.shape}")
        print(f"  Score range: min={scores_sample.min():.4f}, max={scores_sample.max():.4f}")
        
        # Find which columns have highest values
        col_maxes = scores_sample.max(axis=0)
        top_cols = np.argsort(col_maxes)[-5:][::-1]
        print(f"  Top 5 class columns by max value: {top_cols}")
        print(f"  Their max values: {col_maxes[top_cols]}")
        
        num_classes = data.shape[1] - 4
        print(f"\nAssuming {num_classes} classes")
        
        # Extract boxes and scores
        boxes = data[:, :4]
        scores = data[:, 4:]
        
        # Get best class per prediction
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        print(f"\nConfidence stats:")
        print(f"  Min: {confidences.min():.4f}")
        print(f"  Max: {confidences.max():.4f}")
        print(f"  Mean: {confidences.mean():.4f}")
        print(f"  Above {conf_threshold}: {(confidences >= conf_threshold).sum()}")
        
        # Class distribution
        unique, counts = np.unique(class_ids[confidences >= conf_threshold], return_counts=True)
        print(f"\nClass distribution (above threshold):")
        for cls, cnt in sorted(zip(unique, counts), key=lambda x: -x[1])[:10]:
            cls_name = COCO_CLASSES[cls] if cls < len(COCO_CLASSES) else f"class_{cls}"
            print(f"  {cls_name} (id={cls}): {cnt}")
        
        # Filter by confidence
        mask = confidences >= conf_threshold
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]
        
        # Convert xywh to xyxy
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        
        # Apply NMS per class
        final_boxes = []
        final_scores = []
        final_class_ids = []
        
        for cls_id in np.unique(class_ids):
            cls_mask = class_ids == cls_id
            cls_boxes = boxes_xyxy[cls_mask]
            cls_scores = confidences[cls_mask]
            
            keep = nms(cls_boxes, cls_scores, iou_threshold)
            
            for idx in keep:
                final_boxes.append(cls_boxes[idx])
                final_scores.append(cls_scores[idx])
                final_class_ids.append(cls_id)
        
        if verbose:
            print(f"\nAfter NMS: {len(final_boxes)} detections")
        
        if len(final_boxes) == 0:
            return []
        
        return list(zip(final_boxes, final_scores, final_class_ids))
    else:
        print(f"ERROR: Unexpected shape {output.shape}")
        return []


def draw_detections(image: np.ndarray, detections: list, scale: float, padding: tuple):
    """Draw detection boxes on image."""
    pad_x, pad_y = padding
    h, w = image.shape[:2]
    
    for box, conf, class_id in detections:
        # Scale box back to original image coordinates
        x1 = int((box[0] - pad_x) / scale)
        y1 = int((box[1] - pad_y) / scale)
        x2 = int((box[2] - pad_x) / scale)
        y2 = int((box[3] - pad_y) / scale)
        
        # Clip to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Get class name
        class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
        
        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{class_name}: {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(image, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return image


def test_with_image(session, image_path: str):
    """Test model with a single image."""
    print(f"\n{'='*60}")
    print(f"Testing with image: {image_path}")
    print(f"{'='*60}")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not load image: {image_path}")
        return
    
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Preprocess
    blob, scale, padding = preprocess(image)
    print(f"Input blob shape: {blob.shape}")
    
    # Run inference
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: blob})[0]
    
    # Postprocess
    detections = postprocess(output)
    
    # Print detections
    print("\nDetections:")
    for box, conf, class_id in detections[:10]:  # Show first 10
        class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
        print(f"  {class_name}: {conf:.2f}")
    
    if len(detections) > 10:
        print(f"  ... and {len(detections) - 10} more")
    
    # Draw and show
    result = draw_detections(image.copy(), detections, scale, padding)
    cv2.imshow("ONNX Model Test", result)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_with_video(session, video_path: str):
    """Test model with video."""
    print(f"\n{'='*60}")
    print(f"Testing with video: {video_path}")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        return
    
    input_name = session.get_inputs()[0].name
    frame_count = 0
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Preprocess
        blob, scale, padding = preprocess(frame)
        
        # Run inference
        output = session.run(None, {input_name: blob})[0]
        
        # Postprocess (verbose only on first frame)
        detections = postprocess(output, verbose=(frame_count == 1))
        
        # Draw
        result = draw_detections(frame, detections, scale, padding)
        
        # Show info
        cv2.putText(result, f"Frame: {frame_count} | Detections: {len(detections)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("ONNX Model Test", result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nProcessed {frame_count} frames")


def test_with_webcam(session):
    """Test model with webcam."""
    print(f"\n{'='*60}")
    print("Testing with webcam")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return
    
    input_name = session.get_inputs()[0].name
    first_frame = True
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess
        blob, scale, padding = preprocess(frame)
        
        # Run inference
        output = session.run(None, {input_name: blob})[0]
        
        # Postprocess (verbose only on first frame)
        detections = postprocess(output, verbose=first_frame)
        first_frame = False
        
        # Draw
        result = draw_detections(frame, detections, scale, padding)
        cv2.putText(result, f"Detections: {len(detections)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("ONNX Model Test", result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Test YOLO ONNX model")
    parser.add_argument("--model", type=str, default="models/object.onnx", help="Path to ONNX model")
    parser.add_argument("--image", type=str, help="Path to test image")
    parser.add_argument("--video", type=str, help="Path to test video")
    args = parser.parse_args()
    
    # Load model
    session = load_model(args.model)
    
    # Test based on input type
    if args.image:
        test_with_image(session, args.image)
    elif args.video:
        test_with_video(session, args.video)
    else:
        test_with_webcam(session)


if __name__ == "__main__":
    main()
