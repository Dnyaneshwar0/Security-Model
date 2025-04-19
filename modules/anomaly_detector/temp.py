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

        model_dir = "models"
        self.net = cv2.dnn.readNetFromCaffe(
            os.path.join(model_dir, "deploy.prototxt"),
            os.path.join(model_dir, "mobilenet_iter_73000.caffemodel")
        )

    def is_active_time(self):
        current_hour = datetime.now().hour
        if self.ACTIVE_START_HOUR < self.ACTIVE_END_HOUR:
            return self.ACTIVE_START_HOUR <= current_hour < self.ACTIVE_END_HOUR
        else:
            return current_hour >= self.ACTIVE_START_HOUR or current_hour < self.ACTIVE_END_HOUR

    def run(self, frame, timestamp=None) -> dict:
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

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.MIN_CONFIDENCE and int(detections[0, 0, i, 1]) == 15:
                return {
                    "status": "anomaly",
                    "confidence": confidence,
                    "details": "Human detected during restricted hours",
                    "module": "anomaly_detector"
                }

        return {
            "status": "normal",
            "confidence": 0.0,
            "details": "No human detected",
            "module": "anomaly_detector"
        }
