# HW3P1: Recurrent Neural Networks and CTC

## Overview

This assignment focuses on implementing Recurrent Neural Network (RNN) components from scratch, including RNN cells, GRU cells, and the Connectionist Temporal Classification (CTC) algorithm. You will build sequence-to-sequence models for phoneme classification and character prediction tasks.

## Learning Objectives

- Understand the mathematical foundations of RNNs and GRUs
- Implement RNN and GRU cells with forward and backward passes
- Understand and implement the CTC loss function for sequence alignment
- Build sequence classification models using RNNs
- Implement CTC decoding algorithms (greedy and beam search)

## Assignment Structure

```
HW3P1/
├── standard/                    # Standard version (no autograd)
│   ├── models/
│   │   ├── rnn_classifier.py   # RNN-based phoneme classifier
│   │   └── char_predictor.py    # GRU-based character predictor
│   ├── mytorch/
│   │   ├── rnn_cell.py          # RNN cell implementation
│   │   ├── gru_cell.py          # GRU cell implementation
│   │   └── nn/                  # Neural network components
│   ├── CTC/
│   │   ├── CTC.py               # CTC loss and forward-backward algorithm
│   │   └── CTCDecoding.py       # CTC decoding (greedy, beam search)
│   └── autograder/              # Test suite
├── autograd/                    # Autograd version (with autograd engine)
│   └── [similar structure with autograd integration]
└── HW3P1_WriteUp_Standard_VersionF25.pdf
```

## Components to Implement

### 1. RNN Cell (`mytorch/rnn_cell.py`)

Implement Elman RNN cell:
- **Forward pass**: 
  ```
  h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)
  ```
- **Backward pass**: Compute gradients for weights, biases, and hidden states
- Handle input-to-hidden and hidden-to-hidden transformations

### 2. GRU Cell (`mytorch/gru_cell.py`)

Implement Gated Recurrent Unit:
- **Forward pass**:
  ```
  r_t = sigmoid(W_rx * x_t + W_rh * h_{t-1} + b_r)
  z_t = sigmoid(W_zx * x_t + W_zh * h_{t-1} + b_z)
  n_t = tanh(W_nx * x_t + b_nx + r_t * (W_nh * h_{t-1} + b_nh))
  h_t = (1 - z_t) * n_t + z_t * h_{t-1}
  ```
- **Gates**: Reset gate (r), update gate (z), new gate (n)
- **Backward pass**: Gradient computation through gates

### 3. CTC Loss (`CTC/CTC.py`)

Implement Connectionist Temporal Classification:
- **Forward-Backward Algorithm**: Compute alignment probabilities
- **CTC Loss**: Negative log-likelihood of correct alignment
- Handle blank tokens and repetitions
- Efficient dynamic programming implementation

### 4. CTC Decoding (`CTC/CTCDecoding.py`)

Implement decoding algorithms:
- **Greedy Decoding**: Take most likely token at each timestep
- **Beam Search**: Maintain top-k hypotheses
- Handle blank token removal and repetition collapsing

### 5. RNN Classifier (`models/rnn_classifier.py`)

Build multi-layer RNN for phoneme classification:
- Stack RNN cells
- Process variable-length sequences
- Output phoneme predictions at each timestep

### 6. Character Predictor (`models/char_predictor.py`)

Build GRU-based character-level language model:
- Single GRU cell
- Linear projection to vocabulary
- Predict next character given previous characters

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
conda create -n hw3p1 python=3.9
conda activate hw3p1
```

2. **Install dependencies**:
```bash
cd HW3P1/standard  # or autograd version
pip install numpy scipy torch ipython notebook
```

3. **Verify installation**:
```bash
python -c "import numpy; import torch; print('Setup complete!')"
```

## Running Tests

### Standard Version

```bash
cd standard
python autograder/runner.py        # Full test suite
python autograder/toy_runner.py    # Toy examples
```

### Autograd Version

```bash
cd autograd
python autograder/runner.py
```

### Individual Component Tests

```bash
python autograder/test_rnn.py          # Test RNN cell
python autograder/test_gru.py          # Test GRU cell
python autograder/test_ctc.py          # Test CTC loss
python autograder/test_ctc_decoding.py # Test CTC decoding
```

## Implementation Details

### RNN Cell

**Forward Pass**:
1. Compute input transformation: `W_ih * x_t + b_ih`
2. Compute hidden transformation: `W_hh * h_{t-1} + b_hh`
3. Combine and apply activation: `tanh(input + hidden)`

**Backward Pass**:
1. Compute gradient w.r.t. activation input
2. Backpropagate through tanh
3. Compute gradients for W_ih, b_ih, W_hh, b_hh
4. Compute gradients for x_t and h_{t-1}

### GRU Cell

**Key Components**:
- **Reset Gate (r)**: Controls how much previous hidden state to forget
- **Update Gate (z)**: Controls how much new information to incorporate
- **New Gate (n)**: Computes candidate hidden state

**Gradient Flow**:
- Gradients flow through gates
- Need to handle element-wise multiplications
- Accumulate gradients from multiple paths

### CTC Algorithm

**Forward-Backward Algorithm**:
1. **Forward Pass**: Compute probability of reaching each state
2. **Backward Pass**: Compute probability from each state to end
3. **Alignment Probability**: Product of forward and backward probabilities

**Key Concepts**:
- **Blank Token**: Represents "no output" at a timestep
- **Repetition Handling**: Same token can appear multiple times
- **Alignment**: Multiple valid alignments for same label sequence

**CTC Loss**:
```
Loss = -log(P(alignment | input_sequence))
```

### CTC Decoding

**Greedy Decoding**:
1. At each timestep, select most likely token
2. Remove blank tokens
3. Collapse repetitions

**Beam Search**:
1. Maintain top-k hypotheses at each timestep
2. Expand each hypothesis
3. Prune to top-k
4. Handle blank tokens and repetitions

## Implementation Tips

1. **Sequence Processing**:
   - Handle variable-length sequences
   - Use masking for padding
   - Process sequences in batches

2. **Gradient Computation**:
   - Carefully track gradients through time
   - Handle gradient accumulation
   - Use numerical gradient checking

3. **CTC Implementation**:
   - Use dynamic programming for efficiency
   - Handle numerical stability (log-space computations)
   - Test with simple examples first

4. **Memory Management**:
   - RNNs can be memory-intensive
   - Consider truncating backpropagation through time
   - Use efficient data structures

## Key Concepts

### Recurrent Neural Networks

- **Sequential Processing**: Process sequences one element at a time
- **Hidden State**: Maintains information about previous inputs
- **Vanishing Gradients**: Problem with long sequences (addressed by GRU/LSTM)

### Gated Recurrent Units (GRU)

- **Gates**: Control information flow
- **Reset Gate**: Decides what to forget
- **Update Gate**: Decides what to remember
- **Simpler than LSTM**: Fewer parameters, often similar performance

### Connectionist Temporal Classification

- **Alignment-Free**: No need for frame-level alignments
- **Handles Variable Lengths**: Input and output can have different lengths
- **Blank Token**: Allows model to "skip" output at timesteps
- **Common in ASR**: Used in speech recognition systems

## Common Issues and Solutions

1. **Gradient Explosion/Vanishing**:
   - Use gradient clipping
   - Initialize weights carefully
   - Consider using GRU/LSTM instead of vanilla RNN

2. **CTC Numerical Stability**:
   - Use log-space computations
   - Add small epsilon values
   - Normalize probabilities

3. **Sequence Length Issues**:
   - Handle padding correctly
   - Mask padded positions
   - Process sequences in order

4. **Memory Issues**:
   - Use smaller batch sizes
   - Truncate sequences if needed
   - Clear intermediate values

## Submission

1. **Create submission tarball**:
```bash
cd standard  # or autograd
bash create_tarball.sh
```

2. **Submit `handin.tar`** to Autolab

## Resources

- Assignment Writeup: `HW3P1_WriteUp_Standard_VersionF25.pdf`
- CTC Paper: https://www.cs.toronto.edu/~graves/icml_2006.pdf
- GRU Paper: https://arxiv.org/abs/1412.3555
- RNN Tutorial: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
- NumPy Documentation: https://numpy.org/doc/

## Grading

The autograder evaluates:
- Correctness of RNN/GRU forward and backward passes
- Correctness of CTC loss computation
- Correctness of CTC decoding algorithms
- Numerical accuracy (within tolerance thresholds)
- Implementation completeness

Make sure all tests pass before submission!

