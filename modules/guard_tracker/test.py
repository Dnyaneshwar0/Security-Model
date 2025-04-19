import sys
import os
import cv2
import numpy as np
import time
import platform


# Optional: for audio cue on Windows
if platform.system() == "Windows":
    import winsound

# Ensure the root directory is in sys.path to import core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from inference import GuardVigilanceModule

module = GuardVigilanceModule()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam not accessible.")
    exit()

# === Alert tracking variables ===
alert_states = {"distracted", "sleeping", "absent"}
alert_timer = None
current_alert_state = None
ALERT_THRESHOLD_SECONDS = 5
alert_triggered = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Couldn't read frame.")
        break

    result = module.run(frame)
    status = result["status"]
    confidence = result["confidence"]
    details = result["details"]

    now = time.time()

    if status in alert_states:
        if current_alert_state != status:
            current_alert_state = status
            alert_timer = now
            alert_triggered = False
        elif now - alert_timer >= ALERT_THRESHOLD_SECONDS and not alert_triggered:
            print(f"🚨 ALERT: Person has been '{status}' for more than 5 seconds!")

            # Play audio alert
            if platform.system() == "Windows":
                winsound.Beep(1000, 500)  # freq=1000Hz, duration=500ms
            else:
                print("\a")  # fallback for UNIX-based systems (may or may not beep)

            alert_triggered = True
    else:
        current_alert_state = None
        alert_timer = None
        alert_triggered = False

    # Display output
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

cap.release()
cv2.destroyAllWindows()
