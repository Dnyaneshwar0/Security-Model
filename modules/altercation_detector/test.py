# modules/altercation_detector/test.py
import cv2
import os
import sys

# Ensure the root directory is in sys.path to import core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# modules/altercation_detector/test.py
from inference import AltercationDetector  # Import the class from inference.py
from datetime import datetime

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def main():
    module = AltercationDetector()

    # Select input source: webcam or video file
    input_source = module.select_input_source()

    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Or use the file path if testing with a video file
    if not cap.isOpened():
        print("[ERROR] Could not open video stream.")
        return
    else:
        print("[INFO] Video stream opened successfully.")

    print("[INFO] Starting altercation detection test...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame")
            break
        else:
            print("[INFO] Captured frame.")

        timestamp = get_timestamp()
        result = module.run(frame, timestamp)

        # Print result to console
        print(f"[{result['module']}] {result['status']} | {result['details']} | Confidence: {result['confidence']:.2f}")

        # Optional: display the frame with motion status
        cv2.imshow("Altercation Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Test ended.")

if __name__ == "__main__":
    main()
