# detectors/yolo_detector.py
import os
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_path="best.pt"):
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, '..', 'models', 'best.pt')
        self.model = YOLO(model_path)
        
    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            confidence = box.conf[0].item()
            if confidence > 0.5:
                x_center, y_center, w, h = box.xywh[0].tolist()
                class_id = int(box.cls[0].item())
                detections.append({
                    'class_id': class_id,
                    'x': x_center,
                    'y': y_center
                })
        return detections