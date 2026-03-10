from ultralytics import YOLO
import yaml

with open("object-detection/config.yaml", "r") as f:
    config = yaml.safe_load(f)

version = config["datasetVersion"]
dataset = config["datasets"][version]

modelPath = f"object-detection/runs/{dataset['runName']}/weights/best.pt"
dataYaml = dataset["dataYaml"]

try:
    model = YOLO(modelPath, task = "detect")
    model.export(
        format = "tflite",
        imgsz = 640,
        int8 = True,
        data = dataYaml,
    )
    print(f"Model exported to 'object-detection/runs/{dataset['runName']}/weights/best_saved_model/best_int8.tflite'.")
except Exception as e:
    print(f"Export failed: {e}")