from ultralytics import YOLO
import torch

if __name__ == "__main__":
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dataYAML = "object-detection/whiteboards/data.yaml"

    model = YOLO("yolo26n.pt")

    results = model.train(
        data = dataYAML,
        epochs = 100,
        imgsz = 640,
        batch = 16,
        device = 0 if torch.cuda.is_available() else "cpu",
        patience = 20,
        cache = True,
        amp = True,
        perspective = 0.0001,
        flipud = 0.0,
        project = "object-detection/runs",
        name = "yolo26n-whiteboard"
    )

    print("Training complete. Results have been saved to 'object-detection/runs/yolo26n-whiteboard'.")
