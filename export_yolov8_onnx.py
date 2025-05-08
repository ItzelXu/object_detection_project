# Export YOLOv8 Model to ONNX
# python export_yolov8_onnx.py
from ultralytics import YOLO
import os

# Load YOLOv8 model (auto-downloads "yolov8n.pt" if not present)
model = YOLO("yolov8n.pt")

# Define output directory and filename
output_dir = "models"
os.makedirs(output_dir, exist_ok=True)
onnx_path = os.path.join(output_dir, "yolov8n.onnx")

# Export the model with optimized settings
print("Exporting model to ONNX format...")
model.export(
    format="onnx",
    imgsz=640,  # Fixed input size for better performance
    dynamic=False,  # Disable dynamic axes for better performance
    simplify=True,  # Enable ONNX simplification
    opset=12,  # Use a stable ONNX opset version
    half=True,  # Use FP16 for better performance
    int8=False  # Disable INT8 quantization for accuracy
)

# Move the ONNX model to the desired folder
import shutil
default_onnx_path = "yolov8n.onnx"  # This is where YOLOv8 saves it by default
if os.path.exists(default_onnx_path):
    shutil.move(default_onnx_path, onnx_path)
    print(f"ONNX model saved to: {onnx_path}")
else:
    raise FileNotFoundError(f"ONNX model not found at {default_onnx_path}")

# Verify the exported model
import onnx
try:
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification passed!")
except Exception as e:
    print(f"Warning: ONNX model verification failed: {e}")

# Print model information
print("\nModel Information:")
print(f"Input shape: {[1, 3, 640, 640]}")
print(f"Number of classes: 80 (COCO standard)")
print("Using default COCO class names from unified_inference.py")

