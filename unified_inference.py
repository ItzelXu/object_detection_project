"""
revised from https://github.com/lkk688/VisionLangAnnotate/blob/main/ModelDev/multimodels.py
"""
import sys
import os
import cv2
import numpy as np
import torch
import scipy.special

# Ensure Detectron2 can be found by adding its installation path.
detectron2_path = r"C:\detectron2\detectron2"
if os.path.exists(detectron2_path) and detectron2_path not in sys.path:
    sys.path.insert(0, detectron2_path)

# For YOLOv8 (Ultralytics)
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from detectron2.config import get_cfg, LazyConfig, instantiate
from detectron2.engine.defaults import DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode

# Define COCO class names (80 classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


class UnifiedDetector:
    """
    Unified object detection interface that supports YOLOv8 (Ultralytics) and ViTDet (Detectron2).

    For YOLOv8:
      - If no model_path is supplied, the default "yolov8n.pt" is used (auto-download enabled).
      - The acceleration mode can be "none" (native), "onnx", or "trt". In ONNX mode, the YOLO model
        is exported (if needed) and an ONNX Runtime session is created. In TRT mode, a prebuilt
        TensorRT engine is loaded (placeholder implementation provided).

    For ViTDet:
      - You must supply both a valid config_path and model_path.
    """

    def __init__(self, model_type: str, model_path: str = None, config_path: str = None,
                 confidence: float = 0.5, device: str = None, acceleration: str = "none",
                 onnx_path: str = None, trt_engine_path: str = None):
        """
        Initialize the detector.

        Parameters:
            model_type: 'yolo' for YOLOv8, or 'vitdet' for ViTDet.
            model_path: Path to the model weights (for YOLOv8 this is optional; for ViTDet it’s required).
            config_path: For ViTDet; must be provided when using a custom Detectron2 model.
            confidence: Detection confidence threshold.
            device: 'cuda' or 'cpu'. Defaults to GPU if available.
            acceleration: For YOLOv8, choose "none", "onnx", or "trt".
            onnx_path: Path to save/load the exported ONNX file.
            trt_engine_path: Path to a prebuilt TensorRT engine.
        """
        self.model_type = model_type.lower()
        self.confidence = confidence
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.acceleration = acceleration.lower()

        if self.model_type == "yolo":
            if model_path is None:
                model_path = "yolov8n.pt"  # Auto-download pretrained YOLOv8n weights if needed.
            if YOLO is None:
                raise ImportError("Ultralytics YOLO is not installed. Please run: pip install ultralytics")
            self.yolo_model = YOLO(model_path)
            self.yolo_model.to(self.device)
            self.class_names = self.yolo_model.names

            # Debug: Check if the class names count matches expected 80 classes.
            print(f"Class names loaded: {self.class_names}")
            if len(self.class_names) != 80:
                print(f"[ERROR] Class count mismatch! Expected 80, got {len(self.class_names)}.")

            if self.acceleration == "onnx":
                # Export YOLOv8 model to ONNX if not already exported.
                onnx_path = onnx_path or os.path.join("models", "yolov8n.onnx")
                self.yolo_model.export(format="onnx", imgsz=640, simplify=True)
                import onnxruntime as ort
                self.ort_session = ort.InferenceSession(
                    onnx_path,
                    providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
                )
            elif self.acceleration == "trt":
                # For TRT acceleration, load a prebuilt TensorRT engine.
                if trt_engine_path is None:
                    raise ValueError("Please provide a valid trt_engine_path for TRT acceleration")
                import tensorrt as trt
                TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
                with open(trt_engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                    self.trt_engine = runtime.deserialize_cuda_engine(f.read())
            # Native inference ("none") uses the YOLO model directly.

        elif self.model_type == "vitdet":
            # For ViTDet, both config_path and model_path must be provided.
            if config_path is None or model_path is None:
                raise ValueError("For ViTDet, please provide both a valid config_path and model_path. "
                                 "Example: --cfg configs/mask_rcnn_vitdet_b_100ep.py --w models/mask_rcnn_vitdet_b_100ep.pth")
            if config_path.endswith(".py"):
                cfg_lazy = LazyConfig.load(config_path)
                cfg_lazy.train.init_checkpoint = model_path
                self.vitdet_model = instantiate(cfg_lazy.model)
                DetectionCheckpointer(self.vitdet_model).load(cfg_lazy.train.init_checkpoint)
                self.vitdet_model.to(self.device)
                self.vitdet_model.eval()
                self.predictor = None
            else:
                cfg = get_cfg()
                cfg.merge_from_file(config_path)
                cfg.MODEL.DEVICE = self.device
                cfg.MODEL.WEIGHTS = model_path
                cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence
                self.predictor = DefaultPredictor(cfg)
                self.vitdet_model = None

            # Set up metadata.
            metadata = MetadataCatalog.get("coco_2017_val")
            metadata.thing_classes = COCO_CLASSES
            self.metadata = metadata

        else:
            raise ValueError(f"Unsupported model_type '{self.model_type}'. Use 'yolo' or 'vitdet'.")

    def _predict_yolo_on_frame(self, frame):
        """
        Accelerated YOLO inference using ONNX or TRT.
        Preprocesses the input frame for YOLOv8, then runs the ONNX Runtime session.
        Returns the raw predictions.
        """
        resized = cv2.resize(frame, (640, 640))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        input_tensor = np.transpose(normalized, (2, 0, 1))  # HWC -> CHW
        input_np = np.expand_dims(input_tensor, axis=0)  # Add batch dimension

        if self.acceleration == "onnx":
            import onnxruntime as ort
            ort_inputs = {self.ort_session.get_inputs()[0].name: input_np}
            preds = self.ort_session.run(None, ort_inputs)
            # preds[0] originally has shape: (1, 84, 8400)
            if preds[0].ndim == 3:
                # Transpose to get a shape of (1, 8400, 84)
                preds_out = np.transpose(preds[0], (0, 2, 1))
                return preds_out  # Now, preds_out[0] is (8400, 84)
            else:
                return preds
        elif self.acceleration == "trt":
            return self._predict_trt_on_frame(frame)
        else:
            raise ValueError("Acceleration mode not supported in _predict_yolo_on_frame.")

    def _predict_yolo_on_frame(self, frame):
        """
        Accelerated YOLO inference using ONNX or TRT.
        Preprocesses the input frame for YOLOv8, then either runs the ONNX Runtime
        session (if acceleration is "onnx") or delegates to _predict_trt_on_frame (if "trt").
        Returns the raw predictions.
        """
        resized = cv2.resize(frame, (640, 640))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        input_tensor = np.transpose(normalized, (2, 0, 1))  # HWC -> CHW
        input_np = np.expand_dims(input_tensor, axis=0)  # Add batch dimension

        if self.acceleration == "onnx":
            import onnxruntime as ort
            ort_inputs = {self.ort_session.get_inputs()[0].name: input_np}
            preds = self.ort_session.run(None, ort_inputs)
            return preds
        elif self.acceleration == "trt":
            return self._predict_trt_on_frame(frame)
        else:
            raise ValueError("Acceleration mode not supported in _predict_yolo_on_frame.")

    def _predict_trt_on_frame(self, frame):
        """
        Runs inference on a single frame using the TensorRT engine.
        Returns the raw output predictions in a list so that postprocessing can use them.
        """
        import pycuda.driver as cuda
        import pycuda.autoinit  # Initializes CUDA driver automatically

        # Preprocess the frame: resize to 640x640, convert from BGR to RGB, normalize, and reorder axes.
        resized = cv2.resize(frame, (640, 640))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        input_tensor = np.transpose(normalized, (2, 0, 1))  # CHW format
        input_np = np.expand_dims(input_tensor, axis=0)  # Shape becomes (1, 3, 640, 640)

        # Create an execution context from the loaded TRT engine.
        context = self.trt_engine.create_execution_context()
        # Set the input binding shape (this works when the engine was built with dynamic shapes)
        context.set_binding_shape(0, input_np.shape)

        # Allocate device memory for the input.
        input_size = input_np.size * input_np.dtype.itemsize
        d_input = cuda.mem_alloc(input_size)

        # Determine output binding shape from context. For YOLOv8, this is typically [1, 84, N]
        output_shape = tuple(context.get_binding_shape(1))
        output_size = np.prod(output_shape) * np.float32().itemsize
        d_output = cuda.mem_alloc(output_size)

        # Create a CUDA stream for asynchronous execution.
        stream = cuda.Stream()

        # Transfer the input data from host to device memory.
        cuda.memcpy_htod_async(d_input, input_np, stream)

        # Prepare the bindings list for input and output.
        bindings = [int(d_input), int(d_output)]

        # Execute the network asynchronously.
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        # Allocate host memory for the output and copy the output data from device to host.
        output_np = np.empty(output_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(output_np, d_output, stream)
        stream.synchronize()

        # Return the output wrapped in a list (this matches the signature expected by postprocessing)
        return [output_np]

    def postprocess_yolo_outputs(self, preds, original_frame):
        """
        Parses YOLOv8 ONNX outputs and overlays detections.

        The raw ONNX output has shape (1, 84, 8400). We assume that:
          - The first 6400 candidates come from head1 (grid size 80x80, stride 8)
          - The next 1600 candidates come from head2 (grid size 40x40, stride 16)
          - The last 400 candidates come from head3 (grid size 20x20, stride 32)

        Each candidate provides 84 numbers in the following order:
          [tx, ty, tw, th, objectness, class logits (for 80 classes)]

        We decode each candidate as follows:
          - Apply sigmoid to tx, ty, tw, th, objectness, and class logits.
          - Decode center coordinates using:
              cx = (sigmoid(tx)*2 - 0.5 + grid_x) * stride
              cy = (sigmoid(ty)*2 - 0.5 + grid_y) * stride
            And width and height as:
              w = (sigmoid(tw)*2)**2 * stride
              h = (sigmoid(th)*2)**2 * stride
          - Convert from center format (cx, cy, w, h) to corner (x1, y1, x2, y2)
          - Multiply objectness with the highest class probability.

        Finally, boxes are scaled from the network input space (assumed 640x640)
        to the original image dimensions.
        """
        # --- Step 1. Correct tensor shape ---
        # Raw ONNX output has shape (1,84,8400)
        output = np.squeeze(preds[0], axis=0)  # shape becomes (84, 8400)
        output = output.transpose(1, 0)  # Now shape becomes (8400, 84)

        print("Raw detection (first 4 entries):", output[0, :4])

        boxes = []
        scores = []
        classes = []
        conf_thresh = self.confidence

        # --- Helper: decode one group of candidates ---
        def decode_group(group, grid_w, grid_h, stride):
            # group: shape (num_candidates, 84)
            num_candidates = group.shape[0]
            # Create a grid of shape (grid_h, grid_w)
            # The grid is generated in row-major order.
            grid_x, grid_y = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
            grid_x = grid_x.reshape(-1)  # shape (grid_w*grid_h,)
            grid_y = grid_y.reshape(-1)  # shape (grid_w*grid_h,)
            decoded_boxes = []
            decoded_scores = []
            decoded_classes = []
            for idx, det in enumerate(group):
                # Raw bbox parameters: tx, ty, tw, th
                tx, ty, tw, th = det[:4]
                # Decode center coordinates:
                cx = (scipy.special.expit(tx) * 2 - 0.5 + grid_x[idx]) * stride
                cy = (scipy.special.expit(ty) * 2 - 0.5 + grid_y[idx]) * stride
                w = (scipy.special.expit(tw) * 2) ** 2 * stride
                h = (scipy.special.expit(th) * 2) ** 2 * stride

                # Convert center coordinates (cx,cy,w,h) to corner format:
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2

                # Decode objectness:
                obj_conf = scipy.special.expit(det[4])
                if obj_conf < conf_thresh:
                    continue

                # Decode class scores:
                cls_logits = det[5:5 + len(self.class_names)]
                cls_probs = scipy.special.expit(cls_logits)
                cls_conf = np.max(cls_probs)
                cls_id = int(np.argmax(cls_probs))
                total_conf = obj_conf * cls_conf
                if total_conf < conf_thresh:
                    continue

                decoded_boxes.append([x1, y1, x2, y2])
                decoded_scores.append(total_conf)
                decoded_classes.append(cls_id)
            return decoded_boxes, decoded_scores, decoded_classes

        # --- Step 2. Split output into three groups ---
        head1_count = 80 * 80  # 6400 candidates, stride=8
        head2_count = 40 * 40  # 1600 candidates, stride=16
        head3_count = 20 * 20  # 400 candidates, stride=32

        group1 = output[0:head1_count]  # indices 0 ... 6399
        group2 = output[head1_count:head1_count + head2_count]  # indices 6400 ... 7999
        group3 = output[head1_count + head2_count:]  # indices 8000 ... 8399

        b1, s1, c1 = decode_group(group1, grid_w=80, grid_h=80, stride=8)
        b2, s2, c2 = decode_group(group2, grid_w=40, grid_h=40, stride=16)
        b3, s3, c3 = decode_group(group3, grid_w=20, grid_h=20, stride=32)

        boxes.extend(b1)
        boxes.extend(b2)
        boxes.extend(b3)
        scores.extend(s1)
        scores.extend(s2)
        scores.extend(s3)
        classes.extend(c1)
        classes.extend(c2)
        classes.extend(c3)

        boxes = np.array(boxes)
        scores = np.array(scores)
        classes = np.array(classes)

        # If no valid detections remain, return original frame.
        if boxes.size == 0:
            return original_frame

        # --- Step 3. Scale boxes to original image dimensions ---
        orig_h, orig_w, _ = original_frame.shape
        scale_x = orig_w / 640.0
        scale_y = orig_h / 640.0

        annotated = original_frame.copy()
        for i in UnifiedDetector.nms(boxes, scores, threshold=0.45):
            bx = boxes[i]
            ax1 = int(bx[0] * scale_x)
            ay1 = int(bx[1] * scale_y)
            ax2 = int(bx[2] * scale_x)
            ay2 = int(bx[3] * scale_y)
            cls_id = int(classes[i])
            conf = scores[i]

            if 0 <= cls_id < len(self.class_names):
                label = f"{self.class_names[cls_id]}: {conf:.2f}"
            else:
                label = f"Unknown Class {cls_id}: {conf:.2f}"

            cv2.rectangle(annotated, (ax1, ay1), (ax2, ay2), (0, 255, 0), 2)
            cv2.putText(annotated, label, (ax1, ay1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

        return annotated

    @staticmethod
    def nms(boxes, scores, threshold):
        """
        A simple NumPy implementation of Non-Maximum Suppression.
        boxes: numpy array of shape (N, 4)
        scores: numpy array of shape (N,)
        threshold: IoU threshold for suppression.
        Returns a list of indices of boxes to keep.
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]
        return keep

    def process_video(self, input_path: str, output_path: str, json_out: str = None):
        """
        Run object detection on a video and save the annotated video and optionally JSON detections.
        """
        import time, json
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {input_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        self.timings = []
        coco_results = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            t0 = time.perf_counter()
            if self.model_type == "yolo":
                if self.acceleration == "onnx":
                    preds = self._predict_yolo_on_frame(frame)
                    annotated_frame = self.postprocess_yolo_outputs(preds, frame)
                    # Optionally, extract detection info from preds to build coco_results.
                elif self.acceleration == "trt":
                    preds = self._predict_yolo_on_frame(frame)  # Internally calls _predict_trt_on_frame.
                    annotated_frame = self.postprocess_yolo_outputs(preds, frame)
                else:  # Native YOLO inference.
                    results = self.yolo_model.predict(source=frame, conf=self.confidence, verbose=False)
                    if len(results) > 0:
                        annotated_frame = results[0].plot()
                        for bx, cls, conf in zip(
                            results[0].boxes.xyxy.cpu().numpy(),
                            results[0].boxes.cls.cpu().numpy(),
                            results[0].boxes.conf.cpu().numpy(),
                        ):
                            coco_results.append({
                                "image_id": frame_count,
                                "category_id": int(cls),
                                "bbox": [
                                    float(bx[0]),
                                    float(bx[1]),
                                    float(bx[2] - bx[0]),
                                    float(bx[3] - bx[1])
                                ],
                                "score": float(conf),
                            })
                    else:
                        annotated_frame = frame
            elif self.model_type == "vitdet":
                instances = self._predict_vitdet_on_frame(frame)
                instances = instances.to("cpu")
                if instances.has("pred_boxes"):
                    vis = Visualizer(frame[:, :, ::-1], metadata=self.metadata, scale=1.0)
                    vis_out = vis.draw_instance_predictions(instances)
                    annotated_frame = vis_out.get_image()[:, :, ::-1]
                    for i in range(len(instances)):
                        bx = instances.pred_boxes.tensor[i].numpy()
                        coco_results.append({
                            "image_id": frame_count,
                            "category_id": int(instances.pred_classes[i]),
                            "bbox": [
                                float(bx[0]),
                                float(bx[1]),
                                float(bx[2] - bx[0]),
                                float(bx[3] - bx[1])
                            ],
                            "score": float(instances.scores[i]),
                        })
                else:
                    annotated_frame = frame

            out.write(annotated_frame)
            self.timings.append(time.perf_counter() - t0)
        cap.release()
        out.release()
        print(f"Saved output video: {output_path}")

        avg_ms = 1000 * np.mean(self.timings)
        effective_fps = 1 / np.mean(self.timings)
        summary = f"{self.model_type.upper()}  ·  frames: {frame_count}  ·  avg={avg_ms:.1f} ms  ·  {effective_fps:.1f} fps"
        print(summary)

        benchmark_file = output_path.replace('.mp4', '_benchmark.txt')
        with open(benchmark_file, "w") as f:
            f.write(summary + "\n")

        if json_out:
            json_dir = os.path.dirname(json_out)
            if json_dir and not os.path.exists(json_dir):
                os.makedirs(json_dir, exist_ok=True)
            with open(json_out, "w") as f:
                json.dump(coco_results, f)
            print(f"Saved detections JSON → {json_out}")
        return frame_count