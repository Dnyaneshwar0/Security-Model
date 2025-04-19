 
# core/module_interface.py
import numpy as np

class MonitoringModule:
    def __init__(self, config=None):
        self.config = config or {}

    def run(self, frame: np.ndarray, timestamp: str = None) -> dict:
        """
        Processes a video frame and returns detection results.

        Returns:
            dict: {
                "status": str,        # e.g., "alert", "normal"
                "confidence": float,  # e.g., 0.92
                "details": str,       # optional message
                "module": str         # e.g., "guard_tracker"
            }
        """
        raise NotImplementedError("Subclasses must implement the run() method.")
