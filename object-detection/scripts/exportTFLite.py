from ultralytics import YOLO

modelPath = "object-detection\\runs\\yolo26n-whiteboard\\weights\\best.pt"

try:
    model = YOLO(modelPath, task="detect")
    model.export(
        format="tflite",
        imgsz=640,
        int8=True,
        data="object-detection\\whiteboards\\data.yaml"
    )
    print("Model exported to 'object-detection\\runs\\yolo26n-whiteboard\\weights\\best.tflite'.")
except Exception as e:
    print(f"Export failed: {e}")