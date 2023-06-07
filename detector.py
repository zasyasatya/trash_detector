from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2

model = YOLO("detect_train_test/train/weights/best_v2.pt")

results = model.predict(source="0", show=True, conf=0.25)

print(results)