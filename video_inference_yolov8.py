# python video_inference_yolov8.py
"""
Usage:
  1. To run video inference on arbitrary videos (using native YOLOv8):
     Set mode = "video" and place your video files under the videos/ folder.

  2. To evaluate inference using ONNX and TRT on the COCO128 dataset images:
     Set mode = "eval" and ensure that your COCO128 images are under
         datasets/coco128/images/train2017

Project structure:
project_root/
│── models/            (Stores exported ONNX model & TRT engine)
│   ├── yolov8n.onnx
│   ├── yolov8n.trt
│── videos/            (Input videos for video inference)
│── datasets/
│   └── coco128
│       └── images
│           └── val2017  (COCO128 images for evaluation)
│── output/            (Outputs will be saved here)
│── unified_inference.py
│── video_inference_yolov8.py
"""

import os
import json
import time
import cv2
from unified_inference import UnifiedDetector

def process_videos_for_detector(detector, video_paths, output_prefix):
    total_frames = 0
    total_time = 0.0
    for idx, video_path in enumerate(video_paths):
        output_video = f"{output_prefix}_video{idx + 1}.mp4"
        output_json = f"{output_prefix}_video{idx + 1}.json"
        print(f"Processing {video_path} -> {output_video}")
        frames = detector.process_video(video_path, output_video, json_out=output_json)
        total_frames += frames
        total_time += sum(detector.timings)
    return total_frames, total_time

def evaluate_images_for_detector(detector, image_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    total_time = 0.0
    count = 0
    for fname in image_files:
        fpath = os.path.join(image_dir, fname)
        image = cv2.imread(fpath)
        if image is None:
            continue
        t0 = time.time()
        preds = detector._predict_yolo_on_frame(image)  # ONNX inference
        annotated = detector.postprocess_yolo_outputs(preds, image)
        dt = time.time() - t0
        total_time += dt
        count += 1
        cv2.imwrite(os.path.join(output_dir, fname), annotated)
    avg_time = total_time / count if count > 0 else 0.0
    fps = count / total_time if total_time > 0 else 0.0
    return count, total_time, avg_time, fps

def main():
    # Choose mode: "video" for video inference (native model) or "eval" for evaluation on COCO128 images (using ONNX)
    mode = "video"  # Set to "video" OR "eval"

    if mode == "video":
        print("Running video inference with native YOLOv8 model...")
        os.makedirs("output", exist_ok=True)
        video_paths = ["videos/video1.mp4", "videos/video2.mp4"]
        for v in video_paths:
            if not os.path.exists(v):
                print(f"Video {v} not found.")
                return
        detector_native = UnifiedDetector(
            model_type="yolo",
            model_path="yolov8n.pt",
            confidence=0.5,
            acceleration="none"
        )
        total_frames, total_time = process_videos_for_detector(detector_native, video_paths, "output/native")
        fps = total_frames / total_time if total_time > 0 else 0
        print(f"Native YOLOv8: Processed {total_frames} frames in {total_time:.2f} sec, avg FPS: {fps:.2f}")
    elif mode == "eval":
        print("Evaluating ONNX Inference on COCO128 images...")
        coco_img_dir = os.path.join("datasets", "coco128", "images", "val2017")
        if not os.path.exists(coco_img_dir):
            print(f"COCO128 image directory not found: {coco_img_dir}")
            return
        output_eval_dir = os.path.join("output", "eval_onnx")
        detector_onnx = UnifiedDetector(
            model_type="yolo",
            model_path="yolov8n.pt",
            confidence=0.5,
            acceleration="onnx",
            onnx_path=os.path.join("models", "yolov8n.onnx")
        )
        count, total_time, avg_time, fps = evaluate_images_for_detector(detector_onnx, coco_img_dir, output_eval_dir)
        print(f"ONNX Evaluation: Processed {count} images in {total_time:.2f} sec, avg latency: {avg_time:.3f} sec, FPS: {fps:.2f}")
        eval_results = {
            "num_images": count,
            "total_time_sec": total_time,
            "avg_latency_sec": avg_time,
            "fps": fps
        }
        with open(os.path.join("output", "eval_onnx_results.json"), "w") as f:
            json.dump(eval_results, f, indent=4)
        print("Saved evaluation results to output/eval_onnx_results.json")
    else:
        print("Invalid mode selected. Please set mode to 'video' or 'eval'.")

if __name__ == "__main__":
    main()