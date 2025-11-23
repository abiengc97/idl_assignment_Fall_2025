


# HW1P1: Neural Network Fundamentals

## Overview

This assignment focuses on implementing fundamental neural network components from scratch using NumPy. You will build a custom deep learning framework called `mytorch` that includes core building blocks for neural networks, including linear layers, activation functions, loss functions, batch normalization, and optimization algorithms.

## Learning Objectives

- Understand the mathematical foundations of neural networks
- Implement forward and backward propagation from scratch
- Build a working autograd system for automatic differentiation
- Create reusable neural network components
- Apply these components to build multi-layer perceptrons (MLPs)

## Assignment Structure

```
HW1P1/
├── hw1p1_handout/
│   ├── models/
│   │   └── mlp.py              # MLP model implementations (MLP0, MLP1, MLP4)
│   ├── mytorch/
│   │   ├── nn/
│   │   │   ├── activation.py   # Activation functions (ReLU, Sigmoid, Tanh, GELU, Swish, Softmax)
│   │   │   ├── batchnorm.py    # Batch normalization layer
│   │   │   ├── linear.py       # Linear/fully-connected layer
│   │   │   └── loss.py         # Loss functions (MSE, CrossEntropy)
│   │   └── optim/
│   │       └── sgd.py          # Stochastic Gradient Descent optimizer
│   ├── autograder/
│   │   └── hw1p1_autograder.py # Autograder for testing implementations
│   └── requirements.txt
└── hw1p1_Writeup__F25_3.pdf    # Detailed assignment writeup
```

## Components to Implement

### 1. Linear Layer (`mytorch/nn/linear.py`)
- **Forward pass**: Matrix multiplication with weights and bias
- **Backward pass**: Compute gradients for weights, bias, and input
- **Key operations**: 
  - `Z = A @ W.T + ones @ b.T`
  - Gradient computation for `dLdW`, `dLdb`, and `dLdA`

### 2. Activation Functions (`mytorch/nn/activation.py`)
Implement the following activation functions with forward and backward passes:
- **ReLU**: Rectified Linear Unit
- **Sigmoid**: Sigmoid activation
- **Tanh**: Hyperbolic tangent
- **GELU**: Gaussian Error Linear Unit
- **Swish**: Swish activation function
- **Softmax**: Softmax activation (for multi-class classification)

### 3. Loss Functions (`mytorch/nn/loss.py`)
- **MSELoss**: Mean Squared Error loss
- **CrossEntropyLoss**: Cross-entropy loss for classification

### 4. Batch Normalization (`mytorch/nn/batchnorm.py`)
- Normalize inputs during training
- Maintain running statistics for inference
- Handle both training and evaluation modes

### 5. Optimizer (`mytorch/optim/sgd.py`)
- **SGD**: Stochastic Gradient Descent with momentum support
- Update weights and biases using computed gradients

### 6. MLP Models (`models/mlp.py`)
Build three multi-layer perceptrons:
- **MLP0**: Single linear layer with ReLU
- **MLP1**: Two linear layers with ReLU activations
- **MLP4**: Four-layer MLP with various activations

## Setup Instructions

### Prerequisites
- Python 3.9+
- NumPy 2.2.6+
- SciPy 1.15.3+
- PyTorch 2.7.0+ (for reference implementations)
- Jupyter Notebook (for testing)

### Installation

1. **Create a conda environment** (recommended):
```bash
conda create -n hw1p1 python=3.9
conda activate hw1p1
```

2. **Install dependencies**:
```bash
cd HW1P1/hw1p1_handout
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python -c "import numpy; import torch; print('Setup complete!')"
```

## Running the Autograder

The autograder tests your implementations against reference solutions:

```bash
cd hw1p1_handout
python autograder/hw1p1_autograder.py
```

### Autograder Flags

You can control which components are tested by modifying `autograder/hw1p1_autograder_flags.py`:
- `DEBUG_AND_GRADE_LINEAR`: Test linear layer
- `DEBUG_AND_GRADE_ACTIVATION`: Test activation functions
- `DEBUG_AND_GRADE_LOSS`: Test loss functions
- `DEBUG_AND_GRADE_BATCHNORM`: Test batch normalization
- `DEBUG_AND_GRADE_MLP`: Test MLP models

## Implementation Tips

1. **Shape Management**: Pay careful attention to tensor shapes. Use `np.zeros()` and `np.ones()` appropriately for broadcasting.

2. **Gradient Computation**: 
   - Store intermediate values during forward pass for backward pass
   - Use chain rule for gradient computation
   - Verify gradients numerically if needed

3. **Batch Normalization**:
   - Track running mean and variance during training
   - Use these statistics during inference (eval mode)

4. **Testing**: 
   - Test each component individually before combining
   - Compare outputs with PyTorch reference implementations
   - Use the autograder to verify correctness

## Key Concepts

### Forward Propagation
- Data flows from input through layers to output
- Each layer transforms the input: `output = layer.forward(input)`
- Store intermediate values needed for backpropagation

### Backward Propagation
- Compute gradients using chain rule
- Propagate gradients from output to input
- Update parameters using computed gradients

### Autograd System
- Automatic differentiation through computational graph
- Track operations and compute gradients automatically
- Essential for deep learning frameworks

## Submission

1. **Create submission tarball**:
```bash
cd hw1p1_handout
bash create_tarball.sh
```

2. **Submit `handin.tar`** to Autolab

## Resources

- Assignment Writeup: `hw1p1_Writeup__F25_3.pdf`
- NumPy Documentation: https://numpy.org/doc/
- Deep Learning Book (Chapter 6): http://www.deeplearningbook.org/

## Common Issues and Solutions

1. **Shape Mismatch Errors**: 
   - Check input/output shapes at each layer
   - Use `.shape` attribute to debug

2. **Gradient Issues**:
   - Verify gradient computation matches mathematical formulas
   - Check that gradients are being accumulated correctly

3. **Numerical Stability**:
   - Use appropriate data types (float32/float64)
   - Add small epsilon values where needed (e.g., in division)

## Grading

The autograder evaluates:
- Correctness of forward pass outputs
- Correctness of backward pass gradients
- Numerical accuracy (within tolerance thresholds)
- Implementation completeness

Make sure all tests pass before submission!

