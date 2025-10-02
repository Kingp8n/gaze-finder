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
    """Draw angle information and gaze vector in the top right corner."""
    h, w = frame.shape[:2]

    # Convert from radians to degrees
    pitch_deg = pitch * 180 / np.pi
    yaw_deg = yaw * 180 / np.pi

    # Calculate 3D gaze vector using spherical coordinates
    gaze_x = -np.cos(pitch) * np.sin(yaw)
    gaze_y = -np.sin(pitch)
    gaze_z = -np.cos(pitch) * np.cos(yaw)

    # Draw background rectangle for better visibility (expanded for vector)
    cv2.rectangle(frame, (w - 220, 10), (w - 10, 130), (0, 0, 0), -1)
    cv2.rectangle(frame, (w - 220, 10), (w - 10, 130), (255, 255, 255), 2)

    # Draw angle text
    cv2.putText(frame, f'Pitch: {pitch_deg:.1f}°', (w - 210, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f'Yaw: {yaw_deg:.1f}°', (w - 210, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw gaze vector
    cv2.putText(frame, 'Gaze Vector:', (w - 210, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(frame, f'X: {gaze_x:.2f}', (w - 210, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, f'Y: {gaze_y:.2f}', (w - 210, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, f'Z: {gaze_z:.2f}', (w - 210, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return frame

def log_gaze_data(pitch: float, yaw: float):
    """Log gaze data for plotting."""
    global gaze_data, start_time
    current_time = time.time() - start_time

    # Convert from radians to degrees
    pitch_deg = pitch * 180 / np.pi
    yaw_deg = yaw * 180 / np.pi

    # Calculate 3D gaze vector
    gaze_x = -np.cos(pitch) * np.sin(yaw)
    gaze_y = -np.sin(pitch)
    gaze_z = -np.cos(pitch) * np.cos(yaw)

    gaze_data.append({
        'time': current_time,
        'pitch': pitch_deg,
        'yaw': yaw_deg,
        'gaze_x': gaze_x,
        'gaze_y': gaze_y,
        'gaze_z': gaze_z
    })

def save_gaze_plot():
    """Save the gaze data plot including angles and 3D vector components."""
    global gaze_data

    if not gaze_data:
        return

    times = [d['time'] for d in gaze_data]
    pitches = [d['pitch'] for d in gaze_data]
    yaws = [d['yaw'] for d in gaze_data]
    gaze_x = [d['gaze_x'] for d in gaze_data]
    gaze_y = [d['gaze_y'] for d in gaze_data]
    gaze_z = [d['gaze_z'] for d in gaze_data]

    plt.figure(figsize=(15, 10))

    # Gaze angles
    plt.subplot(3, 1, 1)
    plt.plot(times, pitches, 'b-', label='Pitch', linewidth=1.5)
    plt.plot(times, yaws, 'r-', label='Yaw', linewidth=1.5)
    plt.ylabel('Angle (degrees)')
    plt.ylim(-180, 180)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Gaze Angles and 3D Vector Components Over Time')

    # Gaze vector X and Y components
    plt.subplot(3, 1, 2)
    plt.plot(times, gaze_x, 'g-', label='X (left/right)', linewidth=1.5)
    plt.plot(times, gaze_y, 'm-', label='Y (up/down)', linewidth=1.5)
    plt.ylabel('Vector Component')
    plt.ylim(-1.1, 1.1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Gaze Vector X,Y Components')

    # Gaze vector Z component
    plt.subplot(3, 1, 3)
    plt.plot(times, gaze_z, 'c-', label='Z (forward/back)', linewidth=1.5)
    plt.ylabel('Vector Component')
    plt.xlabel('Time (seconds)')
    plt.ylim(-1.1, 1.1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Gaze Vector Z Component')

    plt.tight_layout()
    plt.savefig('images/gaze_vector_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gaze plot saved as 'gaze_vector_plot.png' with {len(gaze_data)} data points")
    print("Plot includes: Pitch/Yaw angles and 3D gaze vector components (X,Y,Z)")

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
