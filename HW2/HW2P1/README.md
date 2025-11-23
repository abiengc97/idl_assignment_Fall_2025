# HW2P1: Convolutional Neural Networks

## Overview

This assignment focuses on implementing convolutional neural network (CNN) components from scratch using NumPy. You will build 1D and 2D convolutional layers, pooling operations, resampling layers, and use them to construct complete CNN architectures for sequence classification tasks.

## Learning Objectives

- Understand the mathematical operations behind convolutions
- Implement 1D and 2D convolutional layers with forward and backward passes
- Implement pooling operations (MaxPool, AveragePool)
- Understand the relationship between MLPs and CNNs through scanning MLP architectures
- Build complete CNN models for classification tasks

## Assignment Structure

```
HW2P1/
├── hw2p1_f25_autolab/
│   └── hw2p1_handout/
│       ├── models/
│       │   ├── cnn.py              # Main CNN model implementation
│       │   ├── mlp_scan.py         # Scanning MLP architectures
│       │   └── mlp.py              # MLP reference implementation
│       ├── mytorch/
│       │   └── nn/
│       │       ├── Conv1d.py        # 1D Convolutional layer
│       │       ├── Conv2d.py       # 2D Convolutional layer
│       │       ├── ConvTranspose.py # Transposed convolution
│       │       ├── pool.py         # Pooling operations
│       │       ├── resampling.py   # Upsampling/Downsampling
│       │       ├── linear.py       # Linear layer (from HW1)
│       │       ├── activation.py   # Activation functions
│       │       └── loss.py         # Loss functions
│       ├── sandbox/                # Testing scripts for individual components
│       ├── autograder/             # Autograder tests
│       └── requirements.txt
└── HW2P1_F25_Writeup.pdf          # Detailed assignment writeup
```

## Components to Implement

### 1. 1D Convolution (`mytorch/nn/Conv1d.py`)

Implement `Conv1d` class with:
- **Forward pass**: 
  - Convolve input with filters
  - Handle stride > 1
  - Apply padding if needed
- **Backward pass**:
  - Compute gradients for filters and bias
  - Compute input gradients
- **Key operations**:
  - Im2col transformation for efficient computation
  - Matrix multiplication for convolution
  - Handle different stride values

### 2. 2D Convolution (`mytorch/nn/Conv2d.py`)

Implement `Conv2d` class with:
- **Forward pass**: 2D convolution operation
- **Backward pass**: Gradient computation
- Similar structure to Conv1d but for 2D inputs

### 3. Transposed Convolution (`mytorch/nn/ConvTranspose.py`)

Implement transposed (deconvolution) layers:
- Used for upsampling
- Reverse operation of convolution
- Important for generative models and segmentation

### 4. Pooling Operations (`mytorch/nn/pool.py`)

Implement:
- **MaxPool1d/2d**: Maximum pooling
- **AveragePool1d/2d**: Average pooling
- Forward and backward passes
- Handle stride and kernel size

### 5. Resampling (`mytorch/nn/resampling.py`)

Implement:
- **Upsampling**: Increase spatial dimensions
- **Downsampling**: Decrease spatial dimensions
- Various methods (nearest neighbor, bilinear, etc.)

### 6. CNN Model (`models/cnn.py`)

Build a complete CNN architecture:
- Stack convolutional layers
- Apply activations between layers
- Flatten before final linear layer
- Handle variable input sizes

### 7. Scanning MLPs (`models/mlp_scan.py`)

Implement two scanning MLP architectures:
- **CNN_SimpleScanningMLP**: Simple scanning pattern
- **CNN_DistributedScanningMLP**: Distributed scanning pattern
- Convert MLP weights to CNN weights
- Demonstrate equivalence between MLPs and CNNs

## Setup Instructions

### Prerequisites
- Python 3.9+
- NumPy 2.2.6+
- SciPy 1.15.3+
- PyTorch 2.7.0+ (for reference implementations)
- Jupyter Notebook

### Installation

1. **Create conda environment**:
```bash
conda create -n hw2p1 python=3.9
conda activate hw2p1
```

2. **Install dependencies**:
```bash
cd HW2P1/hw2p1_f25_autolab/hw2p1_handout
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python -c "import numpy; import torch; print('Setup complete!')"
```

## Running Tests

### Sandbox Scripts

Test individual components using sandbox scripts:

```bash
cd sandbox
python conv1d_sandbox.py      # Test Conv1d
python conv2d_sandbox.py      # Test Conv2d
python convtranspose_sandbox.py  # Test ConvTranspose
python pool_sandbox.py        # Test Pooling
python resampling_sandbox.py  # Test Resampling
python cnn_sandbox.py         # Test CNN model
```

### Autograder

Run the full autograder:

```bash
cd hw2p1_handout
python autograder/runner.py
```

## Implementation Details

### Convolution Operation

The core convolution operation can be implemented using:
1. **Direct convolution**: Nested loops (slow but clear)
2. **Im2col method**: Unfold input into columns, use matrix multiplication (faster)
3. **FFT-based**: Fast Fourier Transform (fastest but more complex)

For this assignment, focus on the im2col method for efficiency.

### Key Formulas

**1D Convolution Output Size**:
```
output_width = (input_width - kernel_size) // stride + 1
```

**2D Convolution Output Size**:
```
output_height = (input_height - kernel_height) // stride_h + 1
output_width = (input_width - kernel_width) // stride_w + 1
```

**Gradient Computation**:
- Filter gradients: Convolve input with output gradients
- Input gradients: Full convolution of output gradients with rotated filters

### CNN Architecture Example

From the autograder reference:
```python
Conv1d(128, 56, kernel=5, stride=1) → Tanh
Conv1d(56, 28, kernel=6, stride=2) → ReLU
Conv1d(28, 14, kernel=2, stride=2) → Sigmoid
Flatten()
Linear(14 * 30, 10)
```

## Implementation Tips

1. **Shape Management**:
   - Carefully track tensor shapes at each layer
   - Use `np.zeros()` and `np.ones()` for initialization
   - Verify output shapes match expected dimensions

2. **Efficiency**:
   - Use im2col for faster convolution
   - Vectorize operations where possible
   - Avoid unnecessary loops

3. **Gradient Checking**:
   - Implement numerical gradient checking
   - Compare with analytical gradients
   - Use small epsilon values

4. **Testing Strategy**:
   - Test each component individually
   - Compare with PyTorch reference
   - Test edge cases (stride=1, stride>1, padding)

## Key Concepts

### Convolution vs. Correlation
- Convolution: Flip the kernel before applying
- Correlation: Apply kernel directly
- In deep learning, we often use correlation (but call it convolution)

### Padding
- **Valid**: No padding (output smaller than input)
- **Same**: Padding to keep output same size as input
- **Full**: Padding to make output larger than input

### Stride
- Controls how much the kernel moves
- Stride > 1 reduces spatial dimensions
- Trade-off between computation and spatial resolution

### Pooling
- **Max Pooling**: Takes maximum value in window
- **Average Pooling**: Takes average value in window
- Reduces spatial dimensions
- Provides translation invariance

## Submission

1. **Create submission tarball**:
```bash
cd hw2p1_handout
bash create_tarball.sh
```

2. **Submit `handin.tar`** to Autolab

## Common Issues and Solutions

1. **Shape Mismatches**:
   - Double-check output size calculations
   - Verify padding and stride values
   - Print shapes at each layer for debugging

2. **Gradient Issues**:
   - Verify gradient computation matches mathematical formulas
   - Check that gradients are properly accumulated
   - Use gradient checking to verify correctness

3. **Performance**:
   - Use im2col for faster computation
   - Vectorize operations
   - Avoid Python loops in hot paths

4. **Memory Issues**:
   - Be careful with large intermediate arrays
   - Consider in-place operations where possible
   - Monitor memory usage

## Resources

- Assignment Writeup: `HW2P1_F25_Writeup.pdf`
- Convolution Tutorial: http://cs231n.github.io/convolutional-networks/
- NumPy Documentation: https://numpy.org/doc/
- PyTorch Conv1d Reference: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html

## Grading

The autograder evaluates:
- Correctness of forward pass outputs
- Correctness of backward pass gradients
- Numerical accuracy (within tolerance thresholds)
- Implementation completeness for all components

Make sure all tests pass before submission!

