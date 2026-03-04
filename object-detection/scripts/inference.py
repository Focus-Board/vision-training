import cv2

from ultralytics import YOLO
model = YOLO("object-detection\\runs\\yolo26n-whiteboard\\weights\\best.pt")
results = model("image.jpeg")
plot = results[0].plot()
cv2.imwrite("inferenceimage.png", plot)