# 1D CNN Audio Generation

A deep learning project implementing a 1D Convolutional Neural Network (CNN) architecture for audio generation. This model uses an encoder-decoder architecture with mel spectrograms for audio processing and generation.

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

## Contact

For questions and feedback, please open an issue in the repository.
