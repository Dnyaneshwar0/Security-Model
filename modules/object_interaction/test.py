# modules/unattended_object_touch/test.py
import sys
import os
import cv2
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
        break

    result = module.run(frame)  # frame is modified in-place
    status = result["status"]
    details = result["details"]

    color = (0, 255, 0) if status == "touching" else (255, 255, 255)
    cv2.putText(frame, status.upper(), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, details, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

    if status == "touching" and platform.system() == "Windows":
        winsound.Beep(1200, 300)

    cv2.imshow("Unattended Object Touch Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
