import pathlib
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from face_detection import RetinaFace

from .utils import prep_input_numpy, getArch
from .results import GazeResultContainer
from .face_pose import FacePoseDetector
from .smoothing import TemporalSmoother
from .calibration import FivePointCalibration


class GazePipeline:
    def __init__(
        self,
        weights: pathlib.Path,
        arch: str,
        device: str = 'cpu',
        include_detector:bool = True,
        confidence_threshold:float = 0.5,
        include_face_pose:bool = True,
        enable_smoothing:bool = False,
        smoothing_type:str = 'moving_average',
        smoothing_buffer_size:int = 5,
        smoothing_alpha:float = 0.3
        ):

        # Save input parameters
        self.weights = weights
        self.include_detector = include_detector
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.include_face_pose = include_face_pose
        self.enable_smoothing = enable_smoothing

        # Initialize temporal smoother if enabled
        self.smoother = None
        if enable_smoothing:
            self.smoother = TemporalSmoother(
                buffer_size=smoothing_buffer_size,
                alpha=smoothing_alpha,
                smoothing_type=smoothing_type
            )

        # Initialize calibration system
        self.calibration = FivePointCalibration()

        # Create L2CS model
        self.model = getArch(arch, 90)
        self.model.load_state_dict(torch.load(self.weights, map_location=device))
        self.model.to(self.device)
        self.model.eval()

        # Create RetinaFace if requested
        if self.include_detector:

            if device.type == 'cpu':
                self.detector = RetinaFace()
            else:
                self.detector = RetinaFace(gpu_id=device.index)

            self.softmax = nn.Softmax(dim=1)
            self.idx_tensor = [idx for idx in range(90)]
            self.idx_tensor = torch.FloatTensor(self.idx_tensor).to(self.device)

        # Create face pose detector if requested
        if self.include_face_pose:
            device_str = 'cpu' if device.type == 'cpu' else f'cuda:{device.index}'
            self.face_pose_detector = FacePoseDetector(device=device_str)

    def step(self, frame: np.ndarray) -> GazeResultContainer:

        # Creating containers
        face_imgs = []
        bboxes = []
        landmarks = []
        scores = []

        if self.include_detector:
            faces = self.detector(frame)

            if faces is not None: 
                for box, landmark, score in faces:

                    # Apply threshold
                    if score < self.confidence_threshold:
                        continue

                    # Extract safe min and max of x,y
                    x_min=int(box[0])
                    if x_min < 0:
                        x_min = 0
                    y_min=int(box[1])
                    if y_min < 0:
                        y_min = 0
                    x_max=int(box[2])
                    y_max=int(box[3])
                    
                    # Crop image
                    img = frame[y_min:y_max, x_min:x_max]
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    face_imgs.append(img)

                    # Save data
                    bboxes.append(box)
                    landmarks.append(landmark)
                    scores.append(score)

                # Predict gaze
                if face_imgs:
                    pitch, yaw = self.predict_gaze(np.stack(face_imgs))

                    # Apply calibration correction if available
                    if pitch.size > 0 and yaw.size > 0:
                        corrected_results = []
                        for i in range(len(pitch)):
                            corrected_pitch, corrected_yaw = self.calibration.apply_correction(pitch[i], yaw[i])
                            corrected_results.append((corrected_pitch, corrected_yaw))

                        if corrected_results:
                            pitch = np.array([r[0] for r in corrected_results])
                            yaw = np.array([r[1] for r in corrected_results])

                    # Apply temporal smoothing if enabled
                    if self.smoother and pitch.size > 0 and yaw.size > 0:
                        smoothed_results = []
                        for i in range(len(pitch)):
                            smoothed_pitch, smoothed_yaw = self.smoother.smooth(pitch[i], yaw[i])
                            smoothed_results.append((smoothed_pitch, smoothed_yaw))

                        if smoothed_results:
                            pitch = np.array([r[0] for r in smoothed_results])
                            yaw = np.array([r[1] for r in smoothed_results])
                else:
                    pitch = np.empty((0,1))
                    yaw = np.empty((0,1))

            else:

                pitch = np.empty((0,1))
                yaw = np.empty((0,1))

        else:
            pitch, yaw = self.predict_gaze(frame)

            # Apply calibration correction if available
            if pitch.size > 0 and yaw.size > 0:
                corrected_pitch, corrected_yaw = self.calibration.apply_correction(pitch[0], yaw[0])
                pitch = np.array([corrected_pitch])
                yaw = np.array([corrected_yaw])

            # Apply temporal smoothing if enabled
            if self.smoother and pitch.size > 0 and yaw.size > 0:
                smoothed_pitch, smoothed_yaw = self.smoother.smooth(pitch[0], yaw[0])
                pitch = np.array([smoothed_pitch])
                yaw = np.array([smoothed_yaw])

        # Face pose detection
        face_poses = []
        pose_angles_R = []
        pose_angles_L = []
        pose_landmarks = []

        if self.include_face_pose and face_imgs:
            try:
                pose_landmarks_detected, angles_R, angles_L, poses = self.face_pose_detector.detect_face_pose(frame)
                pose_landmarks = pose_landmarks_detected
                pose_angles_R = angles_R
                pose_angles_L = angles_L
                face_poses = poses
            except Exception as e:
                print(f"Face pose detection failed: {e}")

        # Save data
        results = GazeResultContainer(
            pitch=pitch,
            yaw=yaw,
            bboxes=np.stack(bboxes) if bboxes else np.empty((0, 4)),
            landmarks=np.stack(landmarks) if landmarks else np.empty((0, 10)),
            scores=np.stack(scores) if scores else np.empty((0,)),
            face_poses=face_poses,
            pose_angles_R=pose_angles_R,
            pose_angles_L=pose_angles_L,
            pose_landmarks=pose_landmarks
        )

        return results

    def predict_gaze(self, frame: Union[np.ndarray, torch.Tensor]):
        
        # Prepare input
        if isinstance(frame, np.ndarray):
            img = prep_input_numpy(frame, self.device)
        elif isinstance(frame, torch.Tensor):
            img = frame
        else:
            raise RuntimeError("Invalid dtype for input")
    
        # Predict 
        gaze_pitch, gaze_yaw = self.model(img)
        pitch_predicted = self.softmax(gaze_pitch)
        yaw_predicted = self.softmax(gaze_yaw)
        
        # Get continuous predictions in degrees.
        pitch_predicted = torch.sum(pitch_predicted.data * self.idx_tensor, dim=1) * 4 - 180
        yaw_predicted = torch.sum(yaw_predicted.data * self.idx_tensor, dim=1) * 4 - 180
        
        pitch_predicted= pitch_predicted.cpu().detach().numpy()* np.pi/180.0
        yaw_predicted= yaw_predicted.cpu().detach().numpy()* np.pi/180.0

        return pitch_predicted, yaw_predicted
