from ultralytics import YOLO
import yaml
import cv2

with open("object-detection/config.yaml", "r") as f:
    config = yaml.safe_load(f)

version = config["datasetVersion"]
dataset = config["datasets"][version]
runsDir = config["runsDir"]

modelPath = f"{runsDir}/{dataset['runName']}/weights/best.pt"

# Change this to your image path
imagePath = "image.jpeg"
outputPath = "inference_output.png"

if __name__ == "__main__":
    model = YOLO(modelPath)
    results = model(imagePath)
    plot = results[0].plot()
    cv2.imwrite(outputPath, plot)
    print(f"Inference saved to '{outputPath}'")