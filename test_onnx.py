# test_onnx.py

import os
import cv2
import numpy as np
import time
from unified_inference import UnifiedDetector, COCO_CLASSES  # Reuse your code


def evaluate_on_image(img_path, detector):
    """
    Read a single image, run ONNX inference using the detector,
    postprocess detections, and return the annotated image.
    """
    orig_img = cv2.imread(img_path)
    if orig_img is None:
        raise ValueError(f"Image not found: {img_path}")

    # Preprocessing and inference: note that the UnifiedDetector
    # already implements _predict_yolo_on_frame and postprocess_yolo_outputs.
    # Here, we call them in sequence. The _predict_yolo_on_frame method
    # will perform the necessary preprocessing and run inference via ONNX.
    preds = detector._predict_yolo_on_frame(orig_img)

    # postprocess_yolo_outputs expects the predictions (raw outputs) and
    # the original image to scale boxes appropriately.
    annotated = detector.postprocess_yolo_outputs(preds, orig_img)

    return annotated


def main():
    # Path to a sample image for testing; update the below path accordingly.
    sample_img_path = "datasets/coco128/images/val2017/000000001000.jpg"
    output_dir = "output/eval_onnx_test"
    os.makedirs(output_dir, exist_ok=True)

    # Create a UnifiedDetector instance with ONNX acceleration.
    # It will use your onnx_path, confidence threshold, and other settings from unified_inference.py.
    detector = UnifiedDetector(
        model_type="yolo",
        onnx_path="models/yolov8n.onnx",
        confidence=0.25,
        acceleration="onnx"
    )

    # Run inference on the sample image.
    start_time = time.time()
    annotated_img = evaluate_on_image(sample_img_path, detector)
    elapsed = time.time() - start_time
    print(f"Inference on image took: {elapsed:.3f} sec")

    # Save the annotated image.
    output_file = os.path.join(output_dir, os.path.basename(sample_img_path))
    cv2.imwrite(output_file, annotated_img)
    print(f"Annotated image saved to: {output_file}")


if __name__ == "__main__":
    main()