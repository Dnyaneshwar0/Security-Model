# modules/<feature_name>/test.py

import cv2
from inference import YourFeatureModule  # Replace with your actual class name
from datetime import datetime

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def main():
    module = YourFeatureModule()

    # Use webcam (0), or replace with video file path for testing
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open video stream.")
        return

    print("[INFO] Starting dummy test for <feature_name>...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = get_timestamp()
        result = module.run(frame, timestamp)

        # Print result to console
        print(f"[{result['module']}] {result['status']} | {result['details']} | Confidence: {result['confidence']:.2f}")

        # Optional: display the feed
        cv2.imshow("<feature_name> Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Test ended.")

if __name__ == "__main__":
    main()
