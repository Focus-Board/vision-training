from ultralytics import YOLO
if __name__ == '__main__':
    
    modelPath = "object-detection\\runs\\yolo26n-whiteboard\\weights\\best.pt"
    dataYAML = "object-detection\\whiteboards\\data.yaml"

    model = YOLO(modelPath)

    results = model.val(
        data=dataYAML,
        imgsz=640,
        split="val")

    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
