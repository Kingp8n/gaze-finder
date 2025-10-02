# Gaze Direction Finder

Build a lightweight, local-only gaze prototype for Cluely that uses a standard webcam to infer eye/pupil position and head pose, classify gaze (left/right/up/down/center) and emit a continuous gaze vector at ~15-30 FPS. Ship a tiny library plus sample app with a swappable face/landmark detector, quick five-point calibration (optional), temporal smoothing and confidence scores, and two outputs: a debug overlay (face/eye ROI, ray, label) and a programmatic stream (console NDJSON and optional local WebSocket with a minimal JS client). Keep it privacy-first (no frame storage or network by default), configurable (FPS, resolution, model, smoothing), and replaceable (clean interfaces). Include a brief README (setup, architecture, calibration, limits), a simple demo clip or screenshots, and an example consumer (browser or Electron). Aim for CPU-only performance, <100 ms latency, and easy integration so anyone can run a demo and see live gaze events in minutes.

## Overview

This project implements a ResNet-based gaze estimation model that predicts 2D gaze direction from facial images. It supports both MPIIFaceGaze and Gaze360 datasets for training and evaluation.

### Architecture
- **Model**: GazeTracker (ResNet-based CNN)
- **Input**: 448x448 RGB face images
- **Output**: Pitch and yaw gaze angles (binned classification + continuous regression)
- **Datasets**: MPIIFaceGaze (leave-one-out), Gaze360 (train/test split)

## Quick Start

### Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Training
```bash
# MPIIFaceGaze dataset
python train.py --dataset mpiigaze --num_epochs 50 --batch_size 16 --lr 0.00001

# Gaze360 dataset
python train.py --dataset gaze360 --num_epochs 50 --batch_size 16 --lr 0.00001
```

### Testing
```bash
python test.py --dataset mpiigaze --snapshot output/snapshots/model.pkl
```

### Pipeline Usage
```python
from modules import GazePipeline
import cv2

# Initialize pipeline
pipeline = GazePipeline(
    weights='path/to/model.pkl',
    arch='ResNet50',
    device='cpu'
)

# Process frame
frame = cv2.imread('image.jpg')
results = pipeline.step(frame)
```

## Project Structure
```
├── modules/
│   ├── model.py        # GazeTracker CNN architecture
│   ├── datasets.py     # Data loaders for MPIIFaceGaze/Gaze360
│   ├── pipeline.py     # Inference pipeline with face detection
│   ├── utils.py        # Utility functions and device selection
│   └── visualize.py    # Visualization and rendering tools
├── train.py           # Training script
├── test.py           # Evaluation script
└── requirements.txt  # Dependencies
```

## Key Features
- **CPU/GPU Support**: Automatic device detection with CPU fallback
- **Multiple Architectures**: ResNet18/34/50/101/152 support
- **Face Detection**: Integrated RetinaFace detector
- **Cross-Platform**: Works on Ubuntu, macOS, Windows
- **Efficient**: ~1400 lines of code, minimal dependencies

## Datasets
- **MPIIFaceGaze**: 15-fold cross-validation, ±42° gaze range
- **Gaze360**: Full 360° gaze coverage, larger scale

## Performance
- **Input Resolution**: 448×448 pixels
- **Angular Range**: ±42° (MPII), ±180° (Gaze360)
- **Inference**: Real-time capable on modern CPUs