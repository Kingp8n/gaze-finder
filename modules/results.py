from dataclasses import dataclass
import numpy as np
from typing import List, Optional

@dataclass
class GazeResultContainer:

    pitch: np.ndarray
    yaw: np.ndarray
    bboxes: np.ndarray
    landmarks: np.ndarray
    scores: np.ndarray
    face_poses: Optional[List[str]] = None
    pose_angles_R: Optional[List[float]] = None
    pose_angles_L: Optional[List[float]] = None
    pose_landmarks: Optional[List] = None
