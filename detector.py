# detector.py

from ultralytics import YOLO

class ObjectDetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.vehicle_classes = [2, 3, 5, 7]
    def detect(self, frame):
        results = self.model(frame)
        
        boxes = results[0].boxes

        vehicle_count = 0

        # 👇 FILTER ONLY VEHICLES
        for box in boxes:
            cls = int(box.cls[0])  # class id

            if cls in self.vehicle_classes:
                vehicle_count += 1

        annotated_frame = results[0].plot()

        return annotated_frame, vehicle_count