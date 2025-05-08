#python train_yolov8.py
from ultralytics import YOLO
import torch
import json
import numpy as np

def main():
    # 1. Load model and move to GPU if available
    model = YOLO("yolov8n.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Using device: {device}")

    # 2. Train
    train_results = model.train(
        data="datasets/coco128.yaml",
        epochs=50,
        batch=8,
        workers=0,
        name="yolov8n-training",
        project="runs"
    )
    print("✅ Training completed.")

    # 3. Validate
    val_metrics = model.val(data="datasets/coco128.yaml", batch=8)
    # Build your results dictionary using the correct methods
    metrics_dict = {
        "Precision": float(val_metrics.box.mp),
        "Recall": float(val_metrics.box.mr),
        "mAP_50": float(val_metrics.box.map50),
        "mAP_50_95": float(val_metrics.box.map),
        # Convert the NumPy array to a plain Python list of floats
        "F1_per_class": [float(x) for x in val_metrics.box.f1]
    }

    print("🏷️ Validation Metrics:")
    for k, v in metrics_dict.items():
        # If it's an array or list, format each element
        if isinstance(v, (list, np.ndarray)):
            # ensure it's a numpy array for easy iteration
            arr = np.array(v)
            formatted = ", ".join(f"{x:.4f}" for x in arr)
            print(f"  {k}: [{formatted}]")
        else:
            # scalar case
            print(f"  {k}: {v:.4f}")

    # 4. Save metrics
    out_path = "runs/yolov8n-training/validation_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"✅ Validation metrics saved to {out_path}")

if __name__ == "__main__":
    # On Windows, this is required for multiprocessing in PyTorch / Ultralytics
    from multiprocessing import freeze_support
    freeze_support()
    main()