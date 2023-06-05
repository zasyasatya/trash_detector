from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2

model = YOLO("D:/Job/trash_detector/runs/detect/train/weights/best.pt")

results = model.predict(source="0", show=True)

print(results)