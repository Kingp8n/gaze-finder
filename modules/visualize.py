import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from .results import GazeResultContainer

# Global variables for data logging
gaze_data = []
start_time = time.time()

def draw_gaze(a,b,c,d,image_in, pitchyaw, thickness=2, color=(255, 255, 0),sclae=2.0):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    (h, w) = image_in.shape[:2]
    length = c
    pos = (int(a+c / 2.0), int(b+d / 2.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[0]) * np.cos(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[1])
    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.18)
    return image_out

def draw_bbox(frame: np.ndarray, bbox: np.ndarray):
    
    x_min=int(bbox[0])
    if x_min < 0:
        x_min = 0
    y_min=int(bbox[1])
    if y_min < 0:
        y_min = 0
    x_max=int(bbox[2])
    y_max=int(bbox[3])

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)

    return frame

def draw_crosshair(frame: np.ndarray):
    """Draw a green crosshair in the center of the frame."""
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2

    # Draw crosshair lines
    cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 2)
    cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 2)

    # Draw center circle
    cv2.circle(frame, (center_x, center_y), 3, (0, 255, 0), -1)

    return frame

def draw_angle_display(frame: np.ndarray, pitch: float, yaw: float):
    """Draw angle information in the top right corner."""
    h, w = frame.shape[:2]

    # Convert from radians to degrees
    pitch_deg = pitch * 180 / np.pi
    yaw_deg = yaw * 180 / np.pi

    # Draw background rectangle for better visibility
    cv2.rectangle(frame, (w - 200, 10), (w - 10, 80), (0, 0, 0), -1)
    cv2.rectangle(frame, (w - 200, 10), (w - 10, 80), (255, 255, 255), 2)

    # Draw angle text
    cv2.putText(frame, f'Pitch: {pitch_deg:.1f}°', (w - 190, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'Yaw: {yaw_deg:.1f}°', (w - 190, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame

def log_gaze_data(pitch: float, yaw: float):
    """Log gaze data for plotting."""
    global gaze_data, start_time
    current_time = time.time() - start_time

    # Convert from radians to degrees
    pitch_deg = pitch * 180 / np.pi
    yaw_deg = yaw * 180 / np.pi

    gaze_data.append({
        'time': current_time,
        'pitch': pitch_deg,
        'yaw': yaw_deg
    })

def save_gaze_plot():
    """Save the gaze data plot."""
    global gaze_data

    if not gaze_data:
        return

    times = [d['time'] for d in gaze_data]
    pitches = [d['pitch'] for d in gaze_data]
    yaws = [d['yaw'] for d in gaze_data]

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(times, pitches, 'b-', label='Pitch')
    plt.ylabel('Pitch (degrees)')
    plt.ylim(-180, 180)
    plt.grid(True)
    plt.legend()
    plt.title('Gaze Angles Over Time')

    plt.subplot(2, 1, 2)
    plt.plot(times, yaws, 'r-', label='Yaw')
    plt.ylabel('Yaw (degrees)')
    plt.xlabel('Time (seconds)')
    plt.ylim(-180, 180)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('gaze_angles_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gaze plot saved as 'gaze_angles_plot.png' with {len(gaze_data)} data points")

def render(frame: np.ndarray, results: GazeResultContainer):

    # Draw crosshair in the center
    frame = draw_crosshair(frame)

    # Draw bounding boxes
    for bbox in results.bboxes:
        frame = draw_bbox(frame, bbox)

    # Draw Gaze and collect angle data
    for i in range(results.pitch.shape[0]):

        bbox = results.bboxes[i]
        pitch = results.pitch[i]
        yaw = results.yaw[i]

        # Log gaze data for the first detected face
        if i == 0:
            log_gaze_data(pitch, yaw)
            frame = draw_angle_display(frame, pitch, yaw)

        # Extract safe min and max of x,y
        x_min=int(bbox[0])
        if x_min < 0:
            x_min = 0
        y_min=int(bbox[1])
        if y_min < 0:
            y_min = 0
        x_max=int(bbox[2])
        y_max=int(bbox[3])

        # Compute sizes
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        draw_gaze(x_min,y_min,bbox_width, bbox_height,frame,(pitch,yaw),color=(0,0,255))

    return frame
