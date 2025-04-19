import sys
import os
import cv2
import numpy as np

# Add project root to path so we can import correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Local import of the guard module
from inference import GuardVigilanceModule

# Initialize the module
module = GuardVigilanceModule()

# Webcam capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam not accessible.")
    exit()

print("📸 Press 'q' to quit...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Couldn't read frame.")
            break

        result = module.run(frame)

        status = result["status"]
        confidence = result["confidence"]
        details = result["details"]

        color_map = {
            "attentive": (0, 255, 0),
            "sleeping": (0, 0, 255),
            "distracted": (0, 255, 255),
            "absent": (128, 128, 128),
            "unknown": (255, 255, 255)
        }
        color = color_map.get(status, (255, 255, 255))
        label = f"{status.upper()} ({confidence:.2f})"

        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, details, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

        cv2.imshow("Guard Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n⛔ Exiting on user interrupt.")

cap.release()
cv2.destroyAllWindows()
