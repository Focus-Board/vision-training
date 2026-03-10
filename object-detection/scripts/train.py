from ultralytics import YOLO
import torch
import yaml

with open("object-detection/config.yaml", "r") as f:
    config = yaml.safe_load(f)

version = config["datasetVersion"]
dataset = config["datasets"][version]

if __name__ == "__main__":
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nTraining on dataset: {version}")

    model = YOLO("yolo26n.pt")

    results = model.train(
        data = dataset["dataYaml"],
        epochs = 100,
        imgsz = 640,
        batch = 16,
        device = 0 if torch.cuda.is_available() else "cpu",
        patience = 20,
        cache = dataset.get("cache", True),  # From config: true (RAM), "disk", or false
        amp = True,
        perspective = 0.0001,
        flipud = 0.0,
        project = "object-detection/runs",
        name = dataset["runName"],
    )

    print(f"\nTraining complete. Results have been saved to 'object-detection/runs/{dataset['runName']}'.")
