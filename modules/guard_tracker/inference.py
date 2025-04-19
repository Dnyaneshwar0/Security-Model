# modules/guard_vigilance/inference.py
from core.module_interface import MonitoringModule
import numpy as np
import cv2
import dlib
import os
from scipy.spatial import distance as dist

class GuardVigilanceModule(MonitoringModule):
    def __init__(self, config=None):
        super().__init__(config)
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models'))

        self.face_net = cv2.dnn.readNetFromCaffe(
            os.path.join(base, "deploy.prototxt"),
            os.path.join(base, "res10_300x300_ssd_iter_140000.caffemodel")
        )

        self.predictor = dlib.shape_predictor(os.path.join(base, "shape_predictor_68_face_landmarks.dat"))
        self.EAR_THRESHOLD = 0.2
        self.YAW_DISTRACT_THRESHOLD = 30  # degrees

        self.cam_matrix = np.array([[650, 0, 320],
                                    [0, 650, 240],
                                    [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros((4,1))  # Assuming no lens distortion

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def get_eye_landmarks(self, shape):
        left_eye = [(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]
        right_eye = [(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]
        return np.array(left_eye), np.array(right_eye)

    def estimate_yaw(self, shape):
        image_points = np.array([
            (shape.part(30).x, shape.part(30).y),  # Nose tip
            (shape.part(8).x, shape.part(8).y),    # Chin
            (shape.part(36).x, shape.part(36).y),  # Left eye
            (shape.part(45).x, shape.part(45).y),  # Right eye
            (shape.part(48).x, shape.part(48).y),  # Left mouth
            (shape.part(54).x, shape.part(54).y),  # Right mouth
        ], dtype="double")

        model_points = np.array([
            (0.0, 0.0, 0.0),               # Nose tip
            (0.0, -63.6, -12.5),           # Chin
            (-43.3, 32.7, -26.0),          # Left eye
            (43.3, 32.7, -26.0),           # Right eye
            (-28.9, -28.9, -24.1),         # Left mouth
            (28.9, -28.9, -24.1),          # Right mouth
        ])

        success, rotation_vec, _ = cv2.solvePnP(model_points, image_points, self.cam_matrix, self.dist_coeffs)
        if not success:
            return None
        rmat, _ = cv2.Rodrigues(rotation_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        return angles[1]  # Yaw

    def run(self, frame: np.ndarray, timestamp: str = None) -> dict:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        best_conf = 0
        face_box = None
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > 0.5 and conf > best_conf:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                face_box = box.astype("int")
                best_conf = conf

        if face_box is None:
            return {
                "status": "absent",
                "confidence": 1.0,
                "details": "No person detected",
                "module": "guard_vigilance"
            }

        x1, y1, x2, y2 = face_box
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rect = dlib.rectangle(x1, y1, x2, y2)
        shape = self.predictor(gray, rect)

        left_eye, right_eye = self.get_eye_landmarks(shape)
        ear = (self.eye_aspect_ratio(left_eye) + self.eye_aspect_ratio(right_eye)) / 2.0
        yaw = self.estimate_yaw(shape)

        if yaw is None:
            return {
                "status": "unknown",
                "confidence": 0.0,
                "details": "Pose estimation failed",
                "module": "guard_vigilance"
            }

        status = "attentive"
        if abs(yaw) > self.YAW_DISTRACT_THRESHOLD:
            status = "distracted"
        if ear < self.EAR_THRESHOLD and abs(yaw) <= self.YAW_DISTRACT_THRESHOLD:
            status = "sleeping"

        return {
            "status": status,
            "confidence": round(ear, 3),
            "details": f"EAR: {ear:.3f}, Yaw: {yaw:.1f}deg",
            "module": "guard_vigilance"
        }
