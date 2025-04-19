# modules/anomaly_detector/test.py
import sys
import os
import cv2
import time
import platform
from datetime import datetime

if platform.system() == "Windows":
    import winsound

# Add root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from inference import AnomalyDetector

module = AnomalyDetector()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam not accessible.")
    exit()

alert_triggered = False
alert_timer = None
ALERT_THRESHOLD_SECONDS = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_copy = frame.copy()
    result = module.run(frame)
    status = result["status"]
    confidence = result["confidence"]
    details = result["details"]

    now = time.time()
    height, width = frame.shape[:2]

    if status == "anomaly":
        if not alert_triggered:
            alert_timer = now
            alert_triggered = True
        elif now - alert_timer >= ALERT_THRESHOLD_SECONDS:
            print(f"🚨 ALERT: {details} (Confidence: {confidence:.2f})")
            if platform.system() == "Windows":
                winsound.Beep(1200, 600)
            else:
                print("\a")
            alert_triggered = False
    else:
        alert_triggered = False
        alert_timer = None

    # Visualize detections (draw red box for anomaly, green for normal)
    color = (0, 0, 255) if status == "anomaly" else (0, 255, 0)
    label = "Person (Anomaly)" if status == "anomaly" else "Person"

    # Only draw if status is not inactive
    if status != "inactive":
        cv2.rectangle(frame_copy, (50, 50), (width - 50, height - 50), color, 2)
        cv2.putText(frame_copy, label, (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Status details
    cv2.putText(frame_copy, f"{details} | Conf: {confidence:.2f}", (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Anomaly Detector", frame_copy)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()