import cv2
import numpy as np
import time
import json
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class CalibrationPoint:
    """Represents a calibration point with screen and gaze coordinates."""
    screen_x: float
    screen_y: float
    gaze_pitch: float
    gaze_yaw: float
    timestamp: float

class FivePointCalibration:
    """Five-point calibration system for gaze estimation accuracy improvement."""

    def __init__(self, screen_width: int = 1920, screen_height: int = 1080):
        """
        Initialize calibration system.

        Args:
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.calibration_points = []
        self.is_calibrated = False

        # Calibration parameters
        self.point_radius = 20
        self.collection_time = 3.0  # seconds to collect data per point
        self.samples_per_point = 30

        # Correction matrices
        self.pitch_correction = None
        self.yaw_correction = None

        # Define 5 calibration points (center + 4 corners offset)
        margin = 0.15  # 15% margin from edges
        self.target_points = [
            (0.5, 0.5),  # Center
            (margin, margin),  # Top-left
            (1.0 - margin, margin),  # Top-right
            (margin, 1.0 - margin),  # Bottom-left
            (1.0 - margin, 1.0 - margin),  # Bottom-right
        ]

    def run_calibration(self, gaze_pipeline) -> bool:
        """
        Run the full calibration routine.

        Args:
            gaze_pipeline: GazePipeline instance for gaze prediction

        Returns:
            True if calibration successful, False otherwise
        """
        print("Starting 5-point calibration...")
        print("Look at each point when it appears and hold your gaze steady.")
        print("Press SPACE to start, ESC to cancel")

        # Create calibration window
        cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Wait for user to start
        if not self._wait_for_start():
            cv2.destroyWindow('Calibration')
            return False

        # Collect data for each point
        self.calibration_points = []

        for i, (rel_x, rel_y) in enumerate(self.target_points):
            print(f"Calibrating point {i+1}/5...")

            # Convert relative coordinates to screen coordinates
            screen_x = int(rel_x * self.screen_width)
            screen_y = int(rel_y * self.screen_height)

            # Collect gaze data for this point
            point_data = self._collect_point_data(
                screen_x, screen_y, gaze_pipeline
            )

            if point_data:
                self.calibration_points.extend(point_data)
            else:
                print("Calibration cancelled or failed")
                cv2.destroyWindow('Calibration')
                return False

        cv2.destroyWindow('Calibration')

        # Calculate correction matrices
        success = self._calculate_correction()
        if success:
            self.is_calibrated = True
            print("Calibration completed successfully!")
            self._save_calibration()
        else:
            print("Calibration failed - insufficient data")

        return success

    def _wait_for_start(self) -> bool:
        """Wait for user to press SPACE to start calibration."""
        while True:
            # Create start screen
            frame = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)

            # Draw instructions
            text_lines = [
                "GAZE CALIBRATION",
                "",
                "Instructions:",
                "1. Look directly at each red circle when it appears",
                "2. Keep your head still and gaze steady",
                "3. Each point will be collected for 3 seconds",
                "",
                "Press SPACE to start calibration",
                "Press ESC to cancel"
            ]

            y_start = self.screen_height // 2 - len(text_lines) * 20
            for i, line in enumerate(text_lines):
                if line == "GAZE CALIBRATION":
                    font_scale = 2.0
                    color = (0, 255, 255)
                    thickness = 3
                else:
                    font_scale = 1.0
                    color = (255, 255, 255)
                    thickness = 2

                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                x = (self.screen_width - text_size[0]) // 2
                y = y_start + i * 40

                cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                           font_scale, color, thickness, cv2.LINE_AA)

            cv2.imshow('Calibration', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                return True
            elif key == 27:  # ESC
                return False

    def _collect_point_data(self, screen_x: int, screen_y: int, gaze_pipeline) -> List[CalibrationPoint]:
        """Collect gaze data for a specific calibration point."""
        point_data = []
        start_time = time.time()

        # Open camera
        cap = cv2.VideoCapture(0)

        while time.time() - start_time < self.collection_time:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                continue

            # Get gaze prediction
            results = gaze_pipeline.step(frame)

            # Create calibration display
            cal_frame = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)

            # Draw calibration point
            progress = (time.time() - start_time) / self.collection_time
            radius = int(self.point_radius * (1.0 + 0.5 * np.sin(progress * 4 * np.pi)))

            cv2.circle(cal_frame, (screen_x, screen_y), radius, (0, 0, 255), -1)
            cv2.circle(cal_frame, (screen_x, screen_y), self.point_radius + 5, (255, 255, 255), 2)

            # Draw progress bar
            bar_width = 200
            bar_height = 20
            bar_x = (self.screen_width - bar_width) // 2
            bar_y = screen_y + 100

            cv2.rectangle(cal_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
            cv2.rectangle(cal_frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), (0, 255, 0), -1)

            # Draw countdown
            remaining = self.collection_time - (time.time() - start_time)
            countdown_text = f"{remaining:.1f}s"
            text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_x = screen_x - text_size[0] // 2
            text_y = screen_y - 50
            cv2.putText(cal_frame, countdown_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                       1.0, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Calibration', cal_frame)

            # Store gaze data if available
            if len(results.pitch) > 0:
                point_data.append(CalibrationPoint(
                    screen_x=screen_x,
                    screen_y=screen_y,
                    gaze_pitch=float(results.pitch[0]),
                    gaze_yaw=float(results.yaw[0]),
                    timestamp=time.time()
                ))

            # Check for ESC key
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                cap.release()
                return None

        cap.release()
        return point_data

    def _calculate_correction(self) -> bool:
        """Calculate correction matrices from calibration data."""
        if len(self.calibration_points) < 10:
            return False

        # Convert to arrays
        screen_coords = np.array([[p.screen_x, p.screen_y] for p in self.calibration_points])
        gaze_coords = np.array([[p.gaze_pitch, p.gaze_yaw] for p in self.calibration_points])

        # Normalize screen coordinates to [-1, 1]
        screen_coords_norm = np.array([
            (screen_coords[:, 0] / self.screen_width - 0.5) * 2,
            (screen_coords[:, 1] / self.screen_height - 0.5) * 2
        ]).T

        try:
            # Calculate linear transformation matrices using least squares
            # screen_coords = A * gaze_coords + B
            ones = np.ones((len(gaze_coords), 1))
            gaze_augmented = np.hstack([gaze_coords, ones])

            # Solve for pitch correction: screen_x = a1*pitch + b1*yaw + c1
            self.pitch_correction = np.linalg.lstsq(gaze_augmented, screen_coords_norm[:, 0], rcond=None)[0]

            # Solve for yaw correction: screen_y = a2*pitch + b2*yaw + c2
            self.yaw_correction = np.linalg.lstsq(gaze_augmented, screen_coords_norm[:, 1], rcond=None)[0]

            return True

        except np.linalg.LinAlgError:
            return False

    def apply_correction(self, pitch: float, yaw: float) -> Tuple[float, float]:
        """
        Apply calibration correction to gaze angles.

        Args:
            pitch: Raw pitch angle in radians
            yaw: Raw yaw angle in radians

        Returns:
            Corrected (pitch, yaw) tuple
        """
        if not self.is_calibrated:
            return pitch, yaw

        # Convert predicted gaze to screen coordinates
        gaze_vector = np.array([pitch, yaw, 1.0])

        # Apply correction matrices
        corrected_x = np.dot(self.pitch_correction, gaze_vector)
        corrected_y = np.dot(self.yaw_correction, gaze_vector)

        # Convert back to gaze angles (inverse of normalization)
        corrected_pitch = corrected_x / 2.0 + 0.5
        corrected_yaw = corrected_y / 2.0 + 0.5

        # Convert back to radians (approximate)
        corrected_pitch = (corrected_pitch - 0.5) * np.pi / 2
        corrected_yaw = (corrected_yaw - 0.5) * np.pi / 2

        return corrected_pitch, corrected_yaw

    def _save_calibration(self, filename: str = "calibration.json"):
        """Save calibration data to file."""
        calibration_data = {
            'screen_width': self.screen_width,
            'screen_height': self.screen_height,
            'pitch_correction': self.pitch_correction.tolist() if self.pitch_correction is not None else None,
            'yaw_correction': self.yaw_correction.tolist() if self.yaw_correction is not None else None,
            'calibration_points': [
                {
                    'screen_x': p.screen_x,
                    'screen_y': p.screen_y,
                    'gaze_pitch': p.gaze_pitch,
                    'gaze_yaw': p.gaze_yaw,
                    'timestamp': p.timestamp
                } for p in self.calibration_points
            ]
        }

        with open(filename, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        print(f"Calibration saved to {filename}")

    def load_calibration(self, filename: str = "calibration.json") -> bool:
        """Load calibration data from file."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            self.screen_width = data['screen_width']
            self.screen_height = data['screen_height']
            self.pitch_correction = np.array(data['pitch_correction']) if data['pitch_correction'] else None
            self.yaw_correction = np.array(data['yaw_correction']) if data['yaw_correction'] else None

            self.calibration_points = [
                CalibrationPoint(
                    screen_x=p['screen_x'],
                    screen_y=p['screen_y'],
                    gaze_pitch=p['gaze_pitch'],
                    gaze_yaw=p['gaze_yaw'],
                    timestamp=p['timestamp']
                ) for p in data['calibration_points']
            ]

            self.is_calibrated = (self.pitch_correction is not None and
                                self.yaw_correction is not None)

            if self.is_calibrated:
                print(f"Calibration loaded from {filename}")

            return self.is_calibrated

        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            return False

    def reset_calibration(self):
        """Reset calibration data."""
        self.calibration_points = []
        self.is_calibrated = False
        self.pitch_correction = None
        self.yaw_correction = None
        print("Calibration reset")