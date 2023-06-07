from ultralytics import YOLO

# Load a model

model = YOLO('detect_train_test/train/weights/best copy.pt')  # load a custom trained

# Export the model
model.export(format='onnx')