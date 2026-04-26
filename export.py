from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom trained model

# Export the model and save to specific path
try:
    exported_model_path = model.export(format="onnx", imgsz=640)     # 'onnx' 'torchscript' 'tensorflow'等
    print(f"Model exported and saved to: {exported_model_path}")
except Exception as e:
    print(f"Error occurred during export: {e}")