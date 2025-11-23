# IDL Fall 2025 Assignments

This repository contains all assignments for the **Introduction Deep Learning (IDL)** course at Carnegie Mellon University, Fall 2025.

## Overview

This course covers fundamental and advanced topics in deep learning, with hands-on implementation of neural network components from scratch. The assignments progress from basic neural networks to state-of-the-art transformer architectures, covering both theoretical understanding and practical implementation.

## Repository Structure

```
idl_assignment_Fall_2025/
├── HW1/          # Neural Networks Fundamentals
│   ├── HW1P1/    # Neural Network Components from Scratch
│   └── HW1P2/    # Frame-Level Speech Recognition
├── HW2/          # Convolutional Neural Networks
│   ├── HW2P1/    # CNN Components Implementation
│   └── HW2P2/    # Face Recognition with Metric Learning
├── HW3/          # Recurrent Neural Networks
│   ├── HW3P1/    # RNNs, GRUs, and CTC
│   └── HW3P2/    # Sequence-to-Sequence ASR
├── HW4/          # Transformers
│   ├── HW4P1/    # Decoder-Only Transformer (Language Modeling)
│   └── HW4P2/    # Encoder-Decoder Transformer (ASR)
└── environment.yml  # Conda environment configuration
```

## Assignments

### Homework 1: Neural Networks Fundamentals

**HW1P1**: [Neural Network Components from Scratch](HW1/HW1P1/README.md)
- Implement linear layers, activation functions, loss functions, and optimizers using NumPy
- Build multi-layer perceptrons (MLPs) from scratch
- Understand forward and backward propagation

**HW1P2**: [Frame-Level Speech Recognition](HW1/HW1P2/README.md)
- Build neural networks for phoneme classification from MFCC features
- Participate in Kaggle competition
- Apply deep learning to speech processing

### Homework 2: Convolutional Neural Networks

**HW2P1**: [CNN Components Implementation](HW2/HW2P1/README.md)
- Implement 1D and 2D convolutional layers from scratch
- Build pooling, resampling, and transposed convolution operations
- Construct complete CNN architectures

**HW2P2**: [Face Recognition with Metric Learning](HW2/HW2P2/README.md)
- Implement ResNet architecture from scratch
- Apply ArcFace loss for face recognition
- Build face verification system evaluated on EER metric

### Homework 3: Recurrent Neural Networks

**HW3P1**: [RNNs, GRUs, and CTC](HW3/HW3P1/README.md)
- Implement RNN and GRU cells from scratch
- Build Connectionist Temporal Classification (CTC) loss and decoding
- Create sequence classification models

**HW3P2**: [Sequence-to-Sequence ASR](HW3/HW3P2/README.md)
- Build encoder-decoder RNN architecture for speech recognition
- Apply CTC loss for end-to-end training
- Implement beam search decoding

### Homework 4: Transformers

**HW4P1**: [Decoder-Only Transformer](HW4/HW4P1/README.md)
- Implement transformer components (attention, positional encoding, decoder layers)
- Build GPT-style language model
- Train on large-scale text data

**HW4P2**: [Encoder-Decoder Transformer for ASR](HW4/HW4P2/README.md)
- Build full encoder-decoder transformer architecture
- Apply to automatic speech recognition
- Implement cross-attention, CTC auxiliary loss, and beam search

## Learning Progression

The assignments are designed to build upon each other:

1. **HW1**: Foundation in neural networks and backpropagation
2. **HW2**: Convolutional operations and computer vision applications
3. **HW3**: Sequential modeling with RNNs and sequence-to-sequence tasks
4. **HW4**: Modern transformer architectures and attention mechanisms

## Key Technologies

- **NumPy**: For implementing neural network components from scratch
- **PyTorch**: For building and training deep learning models
- **Jupyter Notebooks**: For interactive development and experimentation
- **WandB**: For experiment tracking and visualization
- **Kaggle**: For competition-based evaluation

## Setup

### Prerequisites

- Python 3.9+ (or 3.12.4 for HW4)
- Conda (recommended) or pip
- CUDA-capable GPU (recommended for P2 assignments)

### Environment Setup

1. **Create conda environment**:
```bash
conda env create -f environment.yml
conda activate IDL_F25
```

2. **Or install manually**:
```bash
conda create -n IDL_F25 python=3.9
conda activate IDL_F25
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install matplotlib scikit-learn tqdm wandb torchsummary tensorflow==2.12
```

### Individual Assignment Setup

Each assignment has its own setup instructions. Please refer to the specific README files:
- [HW1P1 Setup](HW1/HW1P1/README.md#setup-instructions)
- [HW1P2 Setup](HW1/HW1P2/README.md#setup-instructions)
- [HW2P1 Setup](HW2/HW2P1/README.md#setup-instructions)
- [HW2P2 Setup](HW2/HW2P2/README.md#setup-instructions)
- [HW3P1 Setup](HW3/HW3P1/README.md#setup-instructions)
- [HW3P2 Setup](HW3/HW3P2/README.md#setup-instructions)
- [HW4P1 Setup](HW4/HW4P1/README.md#setup-instructions)
- [HW4P2 Setup](HW4/HW4P2/README.md#setup-instructions)

## Assignment Format

Each homework typically consists of two parts:

- **Part 1 (P1)**: Implementation-focused assignment with autograder testing
  - Implement neural network components from scratch
  - Test against reference implementations
  - Submit code to Autolab

- **Part 2 (P2)**: Application-focused assignment with Kaggle competition
  - Build complete models for real-world tasks
  - Participate in Kaggle competitions
  - Submit both code and predictions

## Resources

### Course Materials
- Assignment writeups (PDFs in each homework directory)
- Autograder test suites
- Reference implementations

### External Resources

- [NumPy Documentation](https://numpy.org/doc/)
- [Deep Learning Book](http://www.deeplearningbook.org/)
- [CS231n Course Notes](https://deeplearning.cs.cmu.edu/F25/index.html)

## Academic Integrity

All assignments follow strict academic integrity policies:
- **No pre-trained models**: Cannot use models from Hugging Face or similar libraries
- **No external data**: Only use provided datasets
- **Own implementation**: Must implement components from scratch using fundamental operations
- **Own results**: Must submit your own code and results

Please refer to individual assignment READMEs for specific requirements.

## Contact

For questions about specific assignments, please refer to:
- Course Piazza (if available)
- Assignment writeups
- Individual README files in each homework directory

## License

This repository contains course assignments and is intended for educational purposes only.

---

**Course**: Introduction Deep Learning (IDL)  
**Institution**: Carnegie Mellon University  
**Semester**: Fall 2025  
**Repository**: IDL Assignment Solutions

