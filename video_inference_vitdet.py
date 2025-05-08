# python video_inference_vitdet.py --cfg configs/mask_rcnn_vitdet_b_100ep.py --w models/mask_rcnn_vitdet_b_100ep.pkl

import os
import argparse
from unified_inference import UnifiedDetector
import time, json

# Set up command-line argument parsing
parser = argparse.ArgumentParser(description="Run ViTDet inference on videos")
parser.add_argument("--cfg", required=True, help="ViTDet config (.py)")
parser.add_argument("--w", required=True, help="Weights (.pth/.pkl)")
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

# Initialize UnifiedDetector for ViTDet using the provided command-line arguments
detector = UnifiedDetector(model_type="vitdet",
                           model_path=args.w,
                           config_path=args.cfg,
                           confidence=0.5)

# Define paths to input videos
video1_path = os.path.join("videos", "video1.mp4")
video2_path = os.path.join("videos", "video2.mp4")
assert os.path.exists(video1_path), f"Input video not found: {video1_path}"
assert os.path.exists(video2_path), f"Input video not found: {video2_path}"

# Process the videos using ViTDet and save the outputs along with JSON detection results
frames1 = detector.process_video(video1_path, os.path.join("output", "video1_vitdet_result.mp4"),
                       json_out=os.path.join("output", "video1_vitdet.json"))
frames2 = detector.process_video(video2_path, os.path.join("output", "video2_vitdet_result.mp4"),
                       json_out=os.path.join("output", "video2_vitdet.json"))

print("ViTDet inference on both videos completed. Results are saved in the 'output' folder.")

total_frames = frames1 + frames2
total_time = sum(detector.timings)
fps = total_frames / total_time
latency_ms = (total_time / total_frames) * 1000

# --- ViTDet Inference Completed ---
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
import json

register_coco_instances(
    "coco_2017_val",                                   # the name you’ll use below
    {},                                                # no extra metadata
    "datasets/coco/annotations/instances_val2017.json",  # path to your JSON
    "datasets/coco/val2017"                       # path to your image folder
)

# Build evaluator & data loader (uses the same cfg as in UnifiedDetector)
cfg = detector.predictor.cfg
evaluator = COCOEvaluator("coco_2017_val", cfg, False, output_dir="output")  # COCO bbox AP metrics
val_loader = build_detection_test_loader(cfg, "coco_2017_val")
coco_metrics = inference_on_dataset(detector.predictor.model, val_loader, evaluator)

# Speed metrics (reuse unified_inference timings)
total_frames = detector.processed_frames  # if you stored this in UnifiedDetector, otherwise sum up per video
total_time = sum(detector.timings)        # sum of all per-frame times :contentReference[oaicite:11]{index=11}
fps = total_frames / total_time
latency_ms = (total_time / total_frames) * 1000

vitdet_results = {
    "accuracy": coco_metrics["bbox"],        # {'AP':…, 'AP50':…, …}
    "speed": {"fps": fps, "latency_ms": latency_ms}
}
with open("output/vitdet_eval.json", "w") as f:
    json.dump(vitdet_results, f)
print("Saved ViTDet evaluation → output/vitdet_eval.json")