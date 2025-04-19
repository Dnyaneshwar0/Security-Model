# modules/anomaly_detector/inference.py
import cv2
import numpy as np
import time
from datetime import datetime
import os

class AnomalyDetector:
    def __init__(self, config=None):
        self.MIN_CONFIDENCE = 0.5
        self.ACTIVE_START_HOUR = 16  # 4 PM
        self.ACTIVE_END_HOUR = 5     # 5 AM

        model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
        self.net = cv2.dnn.readNetFromCaffe(
            os.path.join(model_dir, "deploy.prototxt"),
            os.path.join(model_dir, "mobilenet_iter_73000.caffemodel")
        )

        self.class_labels = [
            "background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ]

    def is_active_time(self):
        current_hour = datetime.now().hour
        if self.ACTIVE_START_HOUR < self.ACTIVE_END_HOUR:
            return self.ACTIVE_START_HOUR <= current_hour < self.ACTIVE_END_HOUR
        else:
            return current_hour >= self.ACTIVE_START_HOUR or current_hour < self.ACTIVE_END_HOUR

    def run(self, frame, timestamp=None):
        (h, w) = frame.shape[:2]
        if not self.is_active_time():
            return {
                "status": "inactive",
                "confidence": 0.0,
                "details": "System outside active hours",
                "module": "anomaly_detector"
            }

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

        anomaly_detected = False
        person_count = 0

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.MIN_CONFIDENCE:
                class_id = int(detections[0, 0, i, 1])
                label = self.class_labels[class_id]

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw detection
                color = (0, 255, 0) if class_id == 15 else (0, 0, 255)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}", (startX, startY - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if class_id == 15:  # person
                    anomaly_detected = True
                    person_count += 1

        if anomaly_detected:
            return {
                "status": "anomaly",
                "confidence": 1.0,
                "details": f"{person_count} person(s) detected during restricted hours",
                "module": "anomaly_detector"
            }

        return {
            "status": "normal",
            "confidence": 0.0,
            "details": "No human detected",
            "module": "anomaly_detector"
        }
