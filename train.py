from ultralytics import YOLO


MODEL_PATH = 'yolov10n.pt'
model = YOLO(MODEL_PATH)

YAML_PATH = 'D:\\projects\\helmet_safety_detection\\dataset\\data.yaml'
EPOCHS = 10
IMG_SIZE = 640
BATCH_SIZE = 256

model.train(data=YAML_PATH, epochs=EPOCHS, batch=BATCH_SIZE, imgsz=IMG_SIZE)
