import os
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_path=None): 
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, '..', 'models', 'best.pt')
        self.model = YOLO(model_path)
        
    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        
        for box in results.boxes:
            confidence = box.conf[0].item()
            if confidence > 0.5:
                x_center, y_center, w, h = box.xywh[0].tolist()
                
                
                return (x_center, y_center)
                
       
        return None