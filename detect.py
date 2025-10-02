import argparse
import pathlib
import cv2
import time
import json
import sys

import torch
import torch.backends.cudnn as cudnn
import numpy as np

from PIL import Image
from PIL import Image, ImageOps

from face_detection import RetinaFace

from modules import select_device, draw_gaze, getArch, GazePipeline, render, save_gaze_plot, GazeWebSocketServer

CWD = pathlib.Path.cwd()

def output_gaze_json(results, timestamp, fps):
    """Output gaze data as NDJSON to console."""
    for i in range(len(results.pitch)):
        gaze_data = {
            "timestamp": timestamp,
            "fps": fps,
            "face_id": i,
            "bbox": {
                "x": float(results.bboxes[i][0]),
                "y": float(results.bboxes[i][1]),
                "width": float(results.bboxes[i][2] - results.bboxes[i][0]),
                "height": float(results.bboxes[i][3] - results.bboxes[i][1])
            },
            "gaze": {
                "pitch_rad": float(results.pitch[i]),
                "yaw_rad": float(results.yaw[i]),
                "pitch_deg": float(results.pitch[i] * 180 / 3.14159),
                "yaw_deg": float(results.yaw[i] * 180 / 3.14159),
                "vector": {
                    "x": float(-np.cos(results.pitch[i]) * np.sin(results.yaw[i])),
                    "y": float(-np.sin(results.pitch[i])),
                    "z": float(-np.cos(results.pitch[i]) * np.cos(results.yaw[i]))
                }
            },
            "confidence": float(results.scores[i]) if len(results.scores) > i else None,
            "face_pose": {
                "pose": results.face_poses[i] if results.face_poses and len(results.face_poses) > i else None,
                "angle_left": float(results.pose_angles_L[i]) if results.pose_angles_L and len(results.pose_angles_L) > i else None,
                "angle_right": float(results.pose_angles_R[i]) if results.pose_angles_R and len(results.pose_angles_R) > i else None
            }
        }
        print(json.dumps(gaze_data), flush=True)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--device',dest='device', help='Device to run model: cpu or gpu:0',
        default="cpu", type=str)
    parser.add_argument(
        '--snapshot',dest='snapshot', help='Path of model snapshot.', 
        default='output/snapshots/L2CS-gaze360-_loader-180-4/_epoch_55.pkl', type=str)
    parser.add_argument(
        '--cam',dest='cam_id', help='Camera device id to use [0]',  
        default=0, type=int)
    parser.add_argument(
        '--arch',dest='arch',help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)
    parser.add_argument(
        '--output-json',dest='output_json',help='Enable NDJSON console output',
        action='store_true')
    parser.add_argument(
        '--websocket',dest='websocket',help='Enable WebSocket server on specified port',
        type=int, default=None)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    arch=args.arch
    cam = args.cam_id

    # Initialize WebSocket server if requested
    websocket_server = None
    if args.websocket:
        websocket_server = GazeWebSocketServer(args.websocket)
        websocket_server.start_server()

    gaze_GazePipeline = GazePipeline(
        weights=CWD / 'models' / 'Gaze_Tracker.pkl',
        arch='ResNet50',
        device = select_device(args.device, batch_size=1)
    )
    cap = cv2.VideoCapture(cam)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    with torch.no_grad():
        while True:

            # Get frame
            success, frame = cap.read()
            start_fps = time.time()

            if not success:
                print("Failed to obtain frame", file=sys.stderr if args.output_json else sys.stdout)
                time.sleep(0.1)
                continue

            # Process frame
            results = gaze_GazePipeline.step(frame)

            myFPS = 1.0 / (time.time() - start_fps)

            # Output JSON if requested
            if args.output_json and len(results.pitch) > 0:
                output_gaze_json(results, time.time(), myFPS)

            # Send to WebSocket clients if server is running
            if websocket_server and len(results.pitch) > 0:
                gaze_data = {
                    "timestamp": time.time(),
                    "fps": myFPS,
                    "face_id": 0,
                    "bbox": {
                        "x": float(results.bboxes[0][0]),
                        "y": float(results.bboxes[0][1]),
                        "width": float(results.bboxes[0][2] - results.bboxes[0][0]),
                        "height": float(results.bboxes[0][3] - results.bboxes[0][1])
                    },
                    "gaze": {
                        "pitch_rad": float(results.pitch[0]),
                        "yaw_rad": float(results.yaw[0]),
                        "pitch_deg": float(results.pitch[0] * 180 / np.pi),
                        "yaw_deg": float(results.yaw[0] * 180 / np.pi),
                        "vector": {
                            "x": float(-np.cos(results.pitch[0]) * np.sin(results.yaw[0])),
                            "y": float(-np.sin(results.pitch[0])),
                            "z": float(-np.cos(results.pitch[0]) * np.cos(results.yaw[0]))
                        }
                    },
                    "confidence": float(results.scores[0]) if len(results.scores) > 0 else None,
                    "face_pose": {
                        "pose": results.face_poses[0] if results.face_poses and len(results.face_poses) > 0 else None,
                        "angle_left": float(results.pose_angles_L[0]) if results.pose_angles_L and len(results.pose_angles_L) > 0 else None,
                        "angle_right": float(results.pose_angles_R[0]) if results.pose_angles_R and len(results.pose_angles_R) > 0 else None
                    }
                }
                websocket_server.broadcast_gaze_data(gaze_data)

            # Visualize output (debug overlay)
            frame = render(frame, results)
            cv2.putText(frame, 'FPS: {:.1f}'.format(myFPS), (10, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("Demo",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            success,frame = cap.read()

    # Save the gaze plot when exiting
    save_gaze_plot()
    cap.release()
    cv2.destroyAllWindows()  
    
