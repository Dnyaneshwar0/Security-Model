# modules/unattended_object_touch/test.py
import sys
import os
import cv2
import numpy as np
import time
import platform

if platform.system() == "Windows":
    import winsound

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from inference import UnattendedObjectTouchModule

module = UnattendedObjectTouchModule()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam not accessible.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Couldn't read frame.")
        break

    result = module.run(frame)
    status = result["status"]
    confidence = result["confidence"]
    details = result["details"]

    # Rerun model inference to draw boxes and hand circles
    detections = module.model(frame).pandas().xyxy[0]
    persons = detections[detections['name'] == 'person']
    objects = detections[detections['name'].isin(module.target_objects)]

    touched_labels = set()
    for _, person in persons.iterrows():
        px1, py1, px2, py2 = int(person.xmin), int(person.ymin), int(person.xmax), int(person.ymax)
        pw, ph = px2 - px1, py2 - py1
        left_hand = (px1 + int(0.1 * pw), py1 + int(0.4 * ph))
        right_hand = (px2 - int(0.1 * pw), py1 + int(0.4 * ph))

        cv2.circle(frame, left_hand, 5, (255, 0, 0), -1)
        cv2.circle(frame, right_hand, 5, (255, 0, 0), -1)

        for _, obj in objects.iterrows():
            ox1, oy1, ox2, oy2 = int(obj.xmin), int(obj.ymin), int(obj.xmax), int(obj.ymax)
            label = obj['name']
            is_touching = ox1 < left_hand[0] < ox2 and oy1 < left_hand[1] < oy2 or \
                          ox1 < right_hand[0] < ox2 and oy1 < right_hand[1] < oy2
            touched_labels.add(label) if is_touching else None

            color = (0, 255, 0) if is_touching else (0, 0, 255)
            txt = f"{label} (Touching)" if is_touching else label
            cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), color, 2)
            cv2.putText(frame, txt, (ox1, oy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    color_map = {
        "touching": (0, 255, 255),
        "no_touch": (0, 255, 0)
    }
    label = f"{status.upper()} ({confidence:.2f})"
    color = color_map.get(status, (255, 255, 255))

    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, details, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

    cv2.imshow("Unattended Object Touch Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
