# modules/altercation_detector/inference.py
import cv2
import numpy as np
import time
from collections import deque

class AltercationDetector:
    def __init__(self, config=None):
        # Parameters
        self.REQUIRED_DURATION = 5    # Must detect fighting for 5+ seconds
        self.COOLDOWN_DURATION = 10   # Seconds before next alert can trigger
        self.VIOLENT_MOTION_THRESHOLD = 10  # For more aggressive movements
        self.SMOOTHING_WINDOW = 15    # Number of frames for motion smoothing

        self.motion_history = deque(maxlen=self.SMOOTHING_WINDOW)
        self.prev_frame = None
        self.fighting_start_time = 0
        self.is_currently_fighting = False
        self.last_alert_time = 0

    def select_input_source(self):
        """ Select input source (webcam or video file) """
        return "webcam"  # Default to webcam (could be extended to support video files)

    def detect_motion(self, prev_frame, current_frame, threshold=25):
        """Detects motion between two frames using simple frame differencing"""
        if prev_frame is None or current_frame is None:
            return 0
        
        gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)
        gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)
        
        frame_diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        
        motion_pixels = np.sum(thresh) / 255
        total_pixels = thresh.size
        return (motion_pixels / total_pixels) * 100

    def run(self, frame: np.ndarray, timestamp: str = None) -> dict:
        """Runs the fighting detection logic"""
        current_time = time.time()

        motion_level = self.detect_motion(self.prev_frame, frame) if self.prev_frame is not None else 0
        self.prev_frame = frame.copy()
        self.motion_history.append(motion_level)

        smoothed_motion = sum(self.motion_history) / len(self.motion_history) if self.motion_history else 0
        is_fighting = smoothed_motion > self.VIOLENT_MOTION_THRESHOLD

        status = "No violence"
        details = "No fighting detected"
        confidence = smoothed_motion  # Confidence is based on the motion level

        if is_fighting:
            if not self.is_currently_fighting:
                self.fighting_start_time = current_time
                self.is_currently_fighting = True
            else:
                fighting_duration = current_time - self.fighting_start_time
                if fighting_duration >= self.REQUIRED_DURATION:
                    if current_time - self.last_alert_time > self.COOLDOWN_DURATION:
                        self.last_alert_time = current_time
                        status = "Violence detected"
                        details = f"Fighting detected for {fighting_duration:.1f} seconds (Motion: {smoothed_motion:.1f}%)"

        else:
            self.is_currently_fighting = False

        # Set text color based on violence detection status
        text_color = (0, 255, 0) if confidence < self.VIOLENT_MOTION_THRESHOLD else (0, 0, 255)
        status_text = "No violence" if confidence < self.VIOLENT_MOTION_THRESHOLD else "Violence detected"
        
        # Add text to frame (showing status and confidence)
        label = f"Violence: {status_text} | Confidence: {confidence:.2f}%"
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)

        return {
            "status": status,
            "confidence": confidence,
            "details": details,
            "module": "altercation_detector"
        }
