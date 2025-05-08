# python export_yolov8_trt.py
import os
import tensorrt as trt

def build_engine(onnx_file_path, engine_file_path, fp16_mode=True):
    """
    Builds a TensorRT engine from an ONNX model and saves it to disk.
    """
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # Use explicit batch flag for TensorRT 8+.
    EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse the ONNX model.
    with open(onnx_file_path, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # Create a builder configuration.
    config = builder.create_builder_config()
    # Note: config.max_workspace_size is deprecated in favor of set_memory_pool_limit,
    # but for now it works for many setups.
    config.max_workspace_size = 1 << 28  # 256MB workspace; adjust as needed.

    # IMPORTANT: For networks with dynamic shape inputs, define an optimization profile.
    # Here we assume the input to the network is the first input tensor.
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    # Set input shape: (min_shape, optimum_shape, max_shape).
    # Adjust the dimensions if necessary (here we assume shape (1, 3, 640, 640)).
    profile.set_shape(input_tensor.name, (1, 3, 640, 640), (1, 3, 640, 640), (1, 3, 640, 640))
    config.add_optimization_profile(profile)

    # Enable FP16 mode if supported and requested.
    if fp16_mode and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Build the engine using the defined network and configuration.
    engine = builder.build_engine(network, config)
    if engine is None:
        print("Failed to build the engine!")
        return None

    # Serialize the engine and save it.
    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())

    return engine

if __name__ == "__main__":
    # Define paths
    onnx_path = os.path.join("models", "yolov8n.onnx")
    engine_path = os.path.join("models", "yolov8n.trt")

    engine = build_engine(onnx_path, engine_path)
    if engine:
        print(f"TensorRT engine successfully saved to {engine_path}")
    else:
        print("Engine build failed!")