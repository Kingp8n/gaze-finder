# Gaze Direction Finder

Build a lightweight, local-only gaze prototype for Cluely that uses a standard webcam to infer eye/pupil position and head pose, classify gaze (left/right/up/down/center) and emit a continuous gaze vector at *~15-30 FPS*. Ship a tiny library plus sample app with a swappable face/landmark detector, quick five-point calibration (optional), temporal smoothing and confidence scores, and two outputs: a debug overlay (face/eye ROI, ray, label) and a programmatic stream (console NDJSON and optional local WebSocket with a minimal JS client). Keep it privacy-first (no frame storage or network by default), configurable (FPS, resolution, model, smoothing), and replaceable (clean interfaces). Include a brief README (setup, architecture, calibration, limits), a simple demo clip or screenshots, and an example consumer (browser or Electron). Aim for CPU-only performance, *<100 ms latency*, and easy integration so anyone can run a demo and see live gaze events in minutes.

## Overview

This project implements a ResNet-based gaze estimation model that predicts 2D gaze direction from facial images. It supports both MPIIFaceGaze and Gaze360 datasets for training and evaluation.

![Gaze Angles Plot](images/gazehub.png)

### Architecture
- **Model**: GazeTracker (ResNet-based CNN)
- **Input**: 448x448 RGB face images
- **Output**: Pitch and yaw gaze angles (binned classification + continuous regression)
  - **Pitch**: Vertical gaze angle (up/down, θ ∈ [-42°, +42°])
  - **Yaw**: Horizontal gaze angle (left/right, φ ∈ [-42°, +42°])
  - **Gaze Vector**: 3D direction = (-cos(θ)sin(φ), -sin(θ), -cos(θ)cos(φ))
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
```bash
python detect.py --snapshot models/Gaze_Tracker.pkl --cam 0
```
- download the trained model from here: [Google Drive](https://drive.google.com/file/d/1sR9gdm9MOB_uCAFiMMRyV5Q2u2I8-QMr/view?usp=sharing)

![Gaze Angles Plot](images/gaze_angles_plot.png)
![Gaze Angles Plot](images/gaze_vector_plot.png)
*Real-time gaze tracking output showing pitch/yaw angles (top), X/Y vector components for horizontal/vertical gaze direction (middle), and Z component for depth perception (bottom). The plot demonstrates smooth gaze transitions and 3D vector representation over a 37-second capture session.*
- Top panel: Pitch/yaw angles in degrees
- Middle panel: X/Y vector components (horizontal/vertical gaze direction)
- Bottom panel: Z component (depth/forward-back direction)

## Key Features
- **CPU/GPU Support**: Automatic device detection with CPU fallback
- **Multiple Architectures**: ResNet18/34/50/101/152 support
- **Face Detection**: Integrated RetinaFace detector
- **Cross-Platform**: Works on Ubuntu, macOS, Windows

## Datasets
- **MPIIFaceGaze**: 15-fold cross-validation, ±42° gaze range
- **Gaze360**: Full 360° gaze coverage, larger scale

## Performance
- **Input Resolution**: 448×448 pixels
- **Angular Range**: ±42° (MPII), ±180° (Gaze360)
- **Inference**: Real-time capable on modern CPUs