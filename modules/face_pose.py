import torch
import numpy as np
import cv2
from facenet_pytorch import MTCNN

class FacePoseDetector:
    def __init__(self, device='cpu'):
        """Initialize face pose detector using MTCNN."""
        self.device = torch.device(device)
        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device
        )

    def npAngle(self, a, b, c):
        """Calculate angle between three points."""
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def classify_face_pose(self, landmarks):
        """Classify face pose based on eye landmarks."""
        # landmarks: [left_eye, right_eye, nose, left_mouth, right_mouth]
        angR = self.npAngle(landmarks[0], landmarks[1], landmarks[2])  # Right eye angle
        angL = self.npAngle(landmarks[1], landmarks[0], landmarks[2])  # Left eye angle

        if ((int(angR) in range(35, 57)) and (int(angL) in range(35, 58))):
            return 'Frontal', angR, angL
        else:
            if angR < angL:
                return 'Left Profile', angR, angL
            else:
                return 'Right Profile', angR, angL

    def detect_face_pose(self, frame):
        """Detect face pose from frame."""
        bbox_, prob_, landmarks_ = self.mtcnn.detect(frame, landmarks=True)

        poses = []
        angles_R = []
        angles_L = []
        valid_landmarks = []

        if bbox_ is not None and landmarks_ is not None:
            for bbox, landmarks, prob in zip(bbox_, landmarks_, prob_):
                if bbox is not None and prob > 0.9:  # High confidence detection
                    pose, angR, angL = self.classify_face_pose(landmarks)
                    poses.append(pose)
                    angles_R.append(angR)
                    angles_L.append(angL)
                    valid_landmarks.append(landmarks)

        return valid_landmarks, angles_R, angles_L, poses

    def draw_face_pose_overlay(self, frame, landmarks_list, angles_R, angles_L, poses):
        """Draw face pose overlay on frame."""
        for landmarks, angle_R, angle_L, pose in zip(landmarks_list, angles_R, angles_L, poses):
            # Color coding for pose
            if pose == 'Frontal':
                color = (255, 255, 255)  # White
            elif pose == 'Right Profile':
                color = (255, 0, 0)      # Blue
            else:  # Left Profile
                color = (0, 0, 255)      # Red

            # Draw facial landmarks
            for land in landmarks:
                cv2.circle(frame, (int(land[0]), int(land[1])), radius=3, color=(0, 255, 255), thickness=-1)

            # Draw triangle connecting eyes and nose
            pts = np.array([
                [int(landmarks[0][0]), int(landmarks[0][1])],  # Left eye
                [int(landmarks[1][0]), int(landmarks[1][1])],  # Right eye
                [int(landmarks[2][0]), int(landmarks[2][1])]   # Nose
            ], np.int32)

            cv2.polylines(frame, [pts], True, (255, 255, 0), 2)

            # Draw pose label
            text_pos = (int(landmarks[0][0]), int(landmarks[0][1]) - 20)
            cv2.putText(frame, f"{pose}", text_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

            # Draw angles (smaller text)
            angle_text = f"L:{int(angle_L)} R:{int(angle_R)}"
            angle_pos = (int(landmarks[0][0]), int(landmarks[0][1]) - 5)
            cv2.putText(frame, angle_text, angle_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        return frame