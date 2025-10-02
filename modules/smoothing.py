import numpy as np
from collections import deque
from typing import Tuple, Optional

class TemporalSmoother:
    """Temporal smoothing for gaze angles using moving average and exponential smoothing."""

    def __init__(self,
                 buffer_size: int = 5,
                 alpha: float = 0.3,
                 smoothing_type: str = 'moving_average'):
        """
        Initialize temporal smoother.

        Args:
            buffer_size: Size of moving average buffer
            alpha: Exponential smoothing factor (0.0 = no smoothing, 1.0 = no history)
            smoothing_type: 'moving_average', 'exponential', or 'both'
        """
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.smoothing_type = smoothing_type

        # Buffers for moving average
        self.pitch_buffer = deque(maxlen=buffer_size)
        self.yaw_buffer = deque(maxlen=buffer_size)

        # Previous values for exponential smoothing
        self.prev_pitch = None
        self.prev_yaw = None

        # Statistics
        self.total_samples = 0
        self.variance_pitch = 0.0
        self.variance_yaw = 0.0

    def smooth(self, pitch: float, yaw: float) -> Tuple[float, float]:
        """
        Apply temporal smoothing to gaze angles.

        Args:
            pitch: Raw pitch angle in radians
            yaw: Raw yaw angle in radians

        Returns:
            Smoothed (pitch, yaw) tuple
        """
        self.total_samples += 1

        if self.smoothing_type == 'moving_average':
            return self._moving_average_smooth(pitch, yaw)
        elif self.smoothing_type == 'exponential':
            return self._exponential_smooth(pitch, yaw)
        elif self.smoothing_type == 'both':
            # Apply moving average first, then exponential
            ma_pitch, ma_yaw = self._moving_average_smooth(pitch, yaw)
            return self._exponential_smooth(ma_pitch, ma_yaw)
        else:
            return pitch, yaw

    def _moving_average_smooth(self, pitch: float, yaw: float) -> Tuple[float, float]:
        """Apply moving average smoothing."""
        self.pitch_buffer.append(pitch)
        self.yaw_buffer.append(yaw)

        # Calculate moving average
        smooth_pitch = np.mean(self.pitch_buffer)
        smooth_yaw = np.mean(self.yaw_buffer)

        # Update variance statistics
        if len(self.pitch_buffer) > 1:
            self.variance_pitch = np.var(self.pitch_buffer)
            self.variance_yaw = np.var(self.yaw_buffer)

        return smooth_pitch, smooth_yaw

    def _exponential_smooth(self, pitch: float, yaw: float) -> Tuple[float, float]:
        """Apply exponential smoothing."""
        if self.prev_pitch is None:
            # First sample
            self.prev_pitch = pitch
            self.prev_yaw = yaw
            return pitch, yaw

        # Exponential smoothing: S_t = α * X_t + (1-α) * S_{t-1}
        smooth_pitch = self.alpha * pitch + (1 - self.alpha) * self.prev_pitch
        smooth_yaw = self.alpha * yaw + (1 - self.alpha) * self.prev_yaw

        self.prev_pitch = smooth_pitch
        self.prev_yaw = smooth_yaw

        return smooth_pitch, smooth_yaw

    def reset(self):
        """Reset smoothing buffers."""
        self.pitch_buffer.clear()
        self.yaw_buffer.clear()
        self.prev_pitch = None
        self.prev_yaw = None
        self.total_samples = 0
        self.variance_pitch = 0.0
        self.variance_yaw = 0.0

    def get_smoothing_stats(self) -> dict:
        """Get smoothing statistics."""
        return {
            'total_samples': self.total_samples,
            'buffer_size': len(self.pitch_buffer),
            'variance_pitch': self.variance_pitch,
            'variance_yaw': self.variance_yaw,
            'smoothing_type': self.smoothing_type,
            'alpha': self.alpha
        }

    def is_stable(self, threshold: float = 0.01) -> bool:
        """Check if gaze is stable based on variance."""
        return (self.variance_pitch < threshold and
                self.variance_yaw < threshold and
                len(self.pitch_buffer) >= self.buffer_size)