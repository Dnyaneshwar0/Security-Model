# modules/unattended_object_touch/inference.py
import cv2
import torch
import numpy as np

class UnattendedObjectTouchModule:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
        self.target_objects = ['backpack', 'bottle', 'laptop', 'handbag']

    def run(self, frame: np.ndarray, timestamp: str = None) -> dict:
        results = self.model(frame)
        detections = results.pandas().xyxy[0]

        persons = detections[detections['name'] == 'person']
        objects = detections[detections['name'].isin(self.target_objects)]
        touched_objects = set()

        for _, person in persons.iterrows():
            px1, py1, px2, py2 = int(person.xmin), int(person.ymin), int(person.xmax), int(person.ymax)
            pw, ph = px2 - px1, py2 - py1
            left_hand = (px1 + int(0.1 * pw), py1 + int(0.4 * ph))
            right_hand = (px2 - int(0.1 * pw), py1 + int(0.4 * ph))

            # Draw person box and hands
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
            cv2.circle(frame, left_hand, 5, (255, 0, 0), -1)
            cv2.circle(frame, right_hand, 5, (0, 0, 255), -1)

            for _, obj in objects.iterrows():
                ox1, oy1, ox2, oy2 = int(obj.xmin), int(obj.ymin), int(obj.xmax), int(obj.ymax)
                label = obj['name']
                # Draw object box and label
                cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (255, 255, 0), 2)
                cv2.putText(frame, label, (ox1, oy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                if ox1 < left_hand[0] < ox2 and oy1 < left_hand[1] < oy2 or \
                   ox1 < right_hand[0] < ox2 and oy1 < right_hand[1] < oy2:
                    touched_objects.add(label)

        status = "touching" if touched_objects else "no_touch"
        details = f"Touched: {', '.join(touched_objects) if touched_objects else 'None'}"

        return {
            "status": status,
            "confidence": len(touched_objects) / len(self.target_objects) if self.target_objects else 0.0,
            "details": details,
            "module": "unattended_object_touch"
        }
