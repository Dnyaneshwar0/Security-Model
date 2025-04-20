# main.py

import cv2
import platform
from datetime import datetime

from modules.guard_tracker.inference import GuardVigilanceModule
from modules.altercation_detector.inference import AltercationDetector
from modules.unauthorized_access.inference import UnauthorizedAccessModule
from modules.anomaly_detector.inference import AnomalyDetector
from modules.object_interaction.inference import UnattendedObjectTouchModule

if platform.system() == "Windows":
    import winsound

modules = [
    GuardVigilanceModule(),
    AltercationDetector(),
    UnauthorizedAccessModule(),
    AnomalyDetector(),
    UnattendedObjectTouchModule()
]

ALERT_DURATION = 5  # seconds
alert_timers = {}

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def check_and_trigger_alert(result, timestamp):
    status = result["status"]
    module_name = result["module"]

    if status in ["distracted", "sleeping", "absent", "touching", "alert"]:
        if module_name not in alert_timers:
            alert_timers[module_name] = {"start": datetime.now(), "alerted": False}
        else:
            elapsed = (datetime.now() - alert_timers[module_name]["start"]).total_seconds()
            if elapsed > ALERT_DURATION and not alert_timers[module_name]["alerted"]:
                print(f"[{timestamp}] ⚠️ ALERT: {module_name.upper()} - {result['details']}")
                if platform.system() == "Windows":
                    winsound.Beep(1000, 400)
                alert_timers[module_name]["alerted"] = True
    else:
        alert_timers[module_name] = {"start": datetime.now(), "alerted": False}

def draw_status_overlay(frame, result, idx):
    text = f"{result['module'].upper()}: {result['status'].upper()} - {result['details']} (Conf: {result['confidence']:.2f})"
    y_offset = 30 + (idx * 25)
    color = (0, 255, 0) if result["status"] in ["attentive", "no_touch"] else (0, 0, 255)
    cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Failed to open video stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = get_timestamp()

        for idx, module in enumerate(modules):
            result = module.run(frame.copy(), timestamp)
            check_and_trigger_alert(result, timestamp)
            draw_status_overlay(frame, result, idx)

        cv2.imshow("Security Monitoring", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
