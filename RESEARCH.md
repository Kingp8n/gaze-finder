# Research Documentation

## Core Technology

### L2CS-Net Architecture
- **Base Model**: ResNet (18/34/50/101/152) backbone
- **Task**: Appearance-based gaze estimation
- **Approach**: Dual-head network for pitch/yaw prediction
- **Loss**: Combined cross-entropy (binned) + MSE (continuous)

### Key Innovation
L2CS-Net uses a hybrid classification-regression approach:
1. **Binned Classification**: Discretized gaze angles for robust learning
2. **Continuous Regression**: Fine-grained angle prediction
3. **Combined Loss**: α-weighted sum for optimal performance

## Implementation Details

### Model Architecture
```
Input (448×448×3) → ResNet Backbone → Global Average Pool → FC Layers
                                                         |→ Pitch Head (num_bins)
                                                         |→ Yaw Head (num_bins)
```

### Training Strategy
- **MPIIFaceGaze**: Leave-one-person-out (15 folds)
- **Gaze360**: Standard train/validation/test split
- **Data Augmentation**: Resize, normalize (ImageNet stats)
- **Optimization**: Adam optimizer, learning rate scheduling

### Technical Challenges Addressed
1. **Device Compatibility**: CPU/GPU fallback mechanism
2. **Dataset Filtering**: Fixed angle threshold bug in Mpiigaze class
3. **Model Loading**: Proper state dict filtering for pre-trained weights
4. **Memory Efficiency**: Removed unnecessary DataParallel for CPU inference

## Dataset Characteristics

### MPIIFaceGaze
- **Size**: ~37K images across 15 subjects
- **Range**: ±42° pitch/yaw angles
- **Evaluation**: Cross-validation (leave-one-person-out)
- **Challenge**: Person-independent generalization

### Gaze360
- **Size**: Larger scale dataset
- **Range**: Full 360° gaze coverage
- **Evaluation**: Standard train/test split
- **Challenge**: Extreme gaze angles

## Performance Considerations
- **Input Resolution**: 448×448 optimized for accuracy/speed trade-off
- **Inference Speed**: Real-time capable on modern CPUs
- **Memory Usage**: Minimal footprint for deployment
- **Angular Accuracy**: Sub-degree precision on calibrated systems

## References
- [L2CS-Net Paper](https://arxiv.org/abs/2203.03339)
- [MPIIFaceGaze Dataset](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation)
- [Gaze Estimation Survey](https://arxiv.org/abs/2104.12668)
- [Data Processing Pipeline](https://phi-ai.buaa.edu.cn/Gazehub/rectification/)
- [Gaze360 Dataset](https://gaze360.csail.mit.edu/)
- [Gaze360  Dataset (Huggingface)](https://huggingface.co/datasets/Morning5/Gaze360)
- [Face Detection](https://github.com/elliottzheng/face-detection)
- [Face Orientation](https://nawafalageel.medium.com/detect-face-poses-f4910dea4a6a)
- [Face Orientation MTCNN](https://github.com/ipazc/mtcnn)