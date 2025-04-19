# modules/<feature_name>/inference.py
from core.module_interface import MonitoringModule
import numpy as np

# Replace YourFeatureModule with module name
class YourFeatureModule(MonitoringModule):
    def __init__(self, config=None):
        super().__init__(config)
        # Load models or config if needed

    def run(self, frame: np.ndarray, timestamp: str = None) -> dict:
        # Dummy logic (replace with actual detection code)
        status = "normal"
        confidence = 1.0

        return {
            "status": status,
            "confidence": confidence,
            "details": "No threat detected",
            "module": "your_feature_name"
        }
