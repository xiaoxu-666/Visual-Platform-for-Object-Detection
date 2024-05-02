from ultralytics import YOLO
from ultralytics import RTDETR

# model = YOLO('yolov8s.pt')
# model = YOLO('yolov8s.yaml')
# model = YOLO('yolov9c.yaml')
# model = YOLO('yolov8s-cbam.yaml')
model = YOLO('yolov8s-ema.yaml')
# model = YOLO('yolov8s-worldv2.pt')
# model = YOLO('yolov8s-ta2.yaml')
# model = YOLO('yolov9c.pt')
# model = YOLO('yolov5su.pt')
# model = RTDETR('rtdetr-l.pt')
# model.load('yolov8s.pt')
# results = model.train(data='data/coco128.yaml', epochs=500, batch=16, imgsz=640, workers=0)
results = model.train(data='data/VOC.yaml', epochs=30, batch=16, imgsz=640, workers=0)
# results = model.train(data='data/mytrain.yaml
# ', epochs=100, batch=16, imgsz=640, workers=0)