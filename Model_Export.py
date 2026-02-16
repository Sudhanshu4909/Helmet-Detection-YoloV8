from ultralytics import YOLO

model = YOLO('models/best.pt')

# Export to ONNX
model.export(format='onnx')

# Export to TensorRT
model.export(format='engine')

# Export to CoreML (iOS)
model.export(format='coreml')