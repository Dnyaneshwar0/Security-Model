from core.module_interface import MonitoringModule
import numpy as np
import cv2
from ultralytics import YOLO
import datetime

class UnauthorizedAccessModule(MonitoringModule):
    def __init__(self, config=None):
        super().__init__(config)
        self.model = YOLO("yolov8n-pose.pt")
        self.window_area = config.get("window_area", (100, 500, 50, 400))
        self.alerts = []
        self.crawlers = []
        self.weapon_first_seen = {}
        self.prev_frame = None
        self.frame_count = 0
        self.fps = 30
        self.weapon_keywords = ['knife', 'gun', 'pistol', 'rifle']

    def detect_motion(self, frame):
        motion_detected = False
        if self.prev_frame is not None:
            frame_diff = cv2.absdiff(self.prev_frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            motion_score = np.sum(frame_diff) / (frame_diff.shape[0] * frame_diff.shape[1])
            motion_detected = motion_score > 0.2
        self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return motion_detected

    def analyze_frame(self, results, height, frame):
        people_count = 0
        timestamp = str(datetime.timedelta(seconds=int(self.frame_count / self.fps)))

        for r in results:
            for box, keypoints in zip(r.boxes, r.keypoints):
                cls = int(box.cls[0])
                name = self.model.names[cls]

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if name == 'person':
                    people_count += 1

                    # Crawling Detection
                    kp = keypoints.xy[0].cpu().numpy()
                    if kp.shape[0] >= 6:
                        head_y = kp[0][1]
                        hip_y = kp[5][1]
                        if head_y - hip_y < 20 and y2 > height * 0.7:
                            if timestamp not in self.crawlers:
                                self.crawlers.append(timestamp)
                                self.alerts.append(f"[CRAWLING DETECTED] at {timestamp}")

                    # Unusual Entry
                    if (x1 < self.window_area[0] or x2 > self.window_area[1]) and \
                       (y1 < self.window_area[2] or y2 > self.window_area[3]):
                        msg = f"[UNUSUAL ENTRY DETECTED] at {timestamp} (Window/Fence)"
                        if msg not in self.alerts:
                            self.alerts.append(msg)

                for weapon in self.weapon_keywords:
                    if weapon in name.lower() and weapon not in self.weapon_first_seen:
                        self.weapon_first_seen[weapon] = timestamp
                        self.alerts.append(f"[WEAPON: {weapon.upper()}] at {timestamp}")

        return people_count

    def run(self, frame: np.ndarray, timestamp: str = None) -> dict:
        result = {
            "status": "normal",
            "confidence": 1.0,
            "details": "No unauthorized access detected",
            "module": "unauthorized_access"
        }

        if frame is None:
            result.update({
                "status": "error",
                "confidence": 0.0,
                "details": "Empty frame received"
            })
            return result

        self.frame_count += 1

        try:
            height = frame.shape[0]
            yolo_results = self.model(frame)
            self.analyze_frame(yolo_results, height, frame)
            motion = self.detect_motion(frame)

            if self.alerts:
                result.update({
                    "status": "alert",
                    "confidence": 0.95,
                    "details": "; ".join(self.alerts[-3:])  # show last 3 alerts
                })

        except Exception as e:
            result.update({
                "status": "error",
                "confidence": 0.0,
                "details": f"Processing error: {str(e)}"
            })

        return result
