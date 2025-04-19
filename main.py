 
# main.py
import cv2
from modules.guard_tracker.inference import GuardVigilanceModule
# from modules.altercation_detector.inference import AltercationModule
# from modules.unauthorized_access.inference import UnauthorizedAccessModule
# from modules.anomaly_detector.inference import AnomalyModule
from datetime import datetime

modules = [
    GuardVigilanceModule(),
    # AltercationModule(),
    # UnauthorizedAccessModule(),
    # AnomalyModule()
]

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def main():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or path to video file

    if not cap.isOpened():
        print("Failed to open video stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = get_timestamp()

        for module in modules:
            result = module.run(frame, timestamp)
            if result["status"] == "alert":
                print(f"[{result['module']}] ALERT: {result['details']} ({result['confidence']})")

        cv2.imshow("Video Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
