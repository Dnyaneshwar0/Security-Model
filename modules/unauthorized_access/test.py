import sys
import os

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import cv2
import numpy as np
from modules.unauthorized_access.inference import UnauthorizedAccessModule

def run_test_with_video(video_path=None):
    print("🔍 Initializing Unauthorized Access Monitoring Module...")
    module = UnauthorizedAccessModule()

    if video_path:
        cap = cv2.VideoCapture(video_path)
        print(f"📽️ Processing video: {video_path}")
    else:
        cap = cv2.VideoCapture(0)
        print("🎥 Using webcam...")

    if not cap.isOpened():
        print("❌ Failed to open video source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("🔚 End of video stream.")
            break

        # Run the module's detection
        result = module.run(frame)
        print(f"[{result['status'].upper()}] - {result['details']}")

        # Annotate frame for visualization (optional)
        cv2.imshow("Unauthorized Access Monitoring", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("1. Use Webcam")
    print("2. Test with video file")
    choice = input("Enter your choice: ")

    if choice == '2':
        video_file = input("Enter path to video file: ")
        if os.path.exists(video_file):
            run_test_with_video(video_file)
        else:
            print("❌ File not found.")
    else:
        run_test_with_video()
