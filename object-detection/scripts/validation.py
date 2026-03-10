from ultralytics import YOLO
import yaml

with open("object-detection/config.yaml", "r") as f:
    config = yaml.safe_load(f)

version = config["datasetVersion"]
dataset = config["datasets"][version]

modelPath = f"object-detection/runs/{dataset['runName']}/weights/best.pt"
dataYaml = dataset["dataYaml"]

if __name__ == "__main__":
    model = YOLO(modelPath)

    results = model.val(
        data = dataYaml,
        imgsz = 640,
        split = "val",
    )

    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
