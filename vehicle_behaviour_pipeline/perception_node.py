from ultralytics import YOLO
import logging

class YOLOPerceptor:
    def __init__(self, model_path):
        logging.getLogger('ultralytics').setLevel(logging.ERROR)
        self.model = YOLO(model_path)
        
    def process_frame(self, frame, conf=0.25):
        """Takes a BGR frame and returns a list of boxes."""
        results = self.model.predict(frame, conf=conf, verbose=False)
        return results[0].boxes
