# 1D CNN Audio Generation

A deep learning project implementing a 1D Convolutional Neural Network (CNN) architecture for toot sound effect audio generation. This model uses an encoder-decoder architecture with mel spectrograms for audio processing and generation.

## Architecture Overview

The model consists of three main components:

### Input Processing
The input processing pipeline transforms raw audio into a format suitable for neural network training:
1. Raw audio is converted to the frequency domain using Short-Time Fourier Transform (STFT)
2. STFT output is converted to mel spectrograms for more efficient processing
3. The spectrograms are normalized to ensure consistent training

### Encoder (1D CNN)
The encoder uses a series of 1D convolutional layers to extract features from the input:

- **Layer 1**: 32 filters with kernel size 3, followed by ReLU activation and batch normalization
- **Layer 2**: 64 filters with kernel size 3, followed by ReLU activation and batch normalization
- **Layer 3**: 128 filters with kernel size 3, followed by ReLU activation and batch normalization
- **Final Layer**: Global Average Pooling to reduce spatial dimensions

### Decoder (Transpose CNN)
The decoder reconstructs the audio using transpose convolutions:

- **Layer 1**: 64 filters with transpose convolution, followed by ReLU activation and batch normalization
- **Layer 2**: 32 filters with transpose convolution, followed by ReLU activation and batch normalization
- **Output Layer**: Linear activation for final output generation

## Training Process

The training pipeline includes:

1. **Input Processing**:
   - Target audio is processed through STFT
   - Conversion to mel spectrograms
   - Normalization

2. **Forward Pass**:
   - Input passes through encoder
   - Encoded features pass through decoder
   - Output generated

3. **Optimization**:
   - Loss calculation between generated and target spectrograms
   - Backpropagation
   - Weight updates

## Dependencies

Required libraries:
- TensorFlow/PyTorch (primary deep learning framework)
- librosa (audio processing)
- numpy (numerical operations)
- scipy (signal processing)

## Installation

```bash
pip install -r requirements.txt
```

## Development Roadmap

Our development roadmap outlines the planned phases and milestones for enhancing the audio generation system. Each phase builds upon the previous one to create a more sophisticated and capable model.

### Phase 1: Core Architecture Implementation
We are currently focused on establishing the fundamental components of our system:
- Implementation of the basic 1D CNN architecture with encoder-decoder structure
- Development of the audio preprocessing pipeline using STFT and mel spectrograms
- Creation of basic training loops and loss functions
- Setting up the project structure and development environment

### Phase 2: Model Enhancement
The second phase will focus on improving the model's architecture and performance:
- Integration of residual connections to improve gradient flow
- Implementation of attention mechanisms for better temporal coherence
- Exploration of alternative activation functions beyond ReLU
- Addition of dropout layers for better generalization
- Development of a more sophisticated loss function incorporating spectral and temporal components

### Phase 3: Training Pipeline Optimization
This phase will enhance the training process and data handling:
- Implementation of mixed-precision training for faster computation
- Development of a data augmentation pipeline for audio
- Creation of a distributed training setup for handling larger datasets
- Integration of automated hyperparameter optimization
- Implementation of model checkpointing and experiment tracking

### Phase 4: Audio Quality Improvements
Focus on enhancing the quality of generated audio:
- Integration of perceptual loss functions
- Implementation of phase reconstruction techniques
- Development of post-processing filters for noise reduction
- Addition of conditioning signals for controlled generation
- Creation of evaluation metrics for audio quality assessment


## Contact

For questions and feedback, please open an issue in the repository.
