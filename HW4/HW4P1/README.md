# HW4P1: Decoder-Only Transformer for Language Modeling

## Overview

This assignment focuses on implementing a Decoder-Only Transformer model for causal language modeling. You will build transformer components from scratch, including attention mechanisms, positional encoding, and decoder layers, then train the model on text data to predict the next token in a sequence.

## Learning Objectives

- Understand transformer architecture and attention mechanisms
- Implement scaled dot-product attention and multi-head attention
- Build decoder-only transformer for language modeling
- Understand causal masking for autoregressive generation
- Train transformer models on large-scale text data

## Task Description

Given a sequence of tokens, predict the next token at each position. The model processes text sequences autoregressively, using previous tokens to predict the next one. This is the architecture used in GPT models.

### Input
- Tokenized text sequences
- Variable-length sequences

### Output
- Probability distribution over vocabulary for next token
- Can be used for text generation

## Dataset

The dataset consists of text data for language modeling:
- **train/**: Training text data
- **valid/**: Validation text data
- **test/**: Test text data

### Data Structure
```
hw4_data_subset/
└── hw4p1_data/           # For causal language modeling
    ├── train/
    ├── valid/
    └── test/
```

## Assignment Structure

```
HW4P1/
└── IDL-HW4/
    └── IDL-HW4/
        ├── HW4P1_Student_Notebook.ipynb  # Main notebook
        ├── README.md                      # Setup instructions
        ├── hw4lib/                        # Main library
        │   ├── data/
        │   │   ├── lm_dataset.py          # Language modeling dataset
        │   │   └── tokenizer.py           # Tokenizer implementation
        │   ├── model/
        │   │   ├── masks.py               # Padding and causal masks
        │   │   ├── positional_encoding.py # Positional encoding
        │   │   ├── sublayers.py           # Attention and FFN sublayers
        │   │   ├── decoder_layers.py      # Decoder layer
        │   │   └── transformers.py        # Decoder-only transformer
        │   └── trainers/
        │       └── lm_trainer.py          # Language modeling trainer
        ├── mytorch/                       # Custom attention implementation
        │   └── nn/
        │       ├── scaled_dot_product_attention.py
        │       └── multi_head_attention.py
        ├── tests/                         # Test suite
        └── requirements.txt
```

## Components to Implement

### 1. Masks (`hw4lib/model/masks.py`)

Implement masking functions:
- **PadMask**: Mask padding tokens in attention
- **CausalMask**: Mask future tokens for autoregressive generation

### 2. Positional Encoding (`hw4lib/model/positional_encoding.py`)

Implement positional encoding:
- Add position information to token embeddings
- Use sinusoidal or learned positional encodings
- Handle sequences up to max_len

### 3. Transformer Sublayers (`hw4lib/model/sublayers.py`)

Implement:
- **SelfAttentionLayer**: Self-attention mechanism
- **FeedForwardLayer**: Position-wise feed-forward network
- Both with residual connections and layer normalization

### 4. Decoder Layer (`hw4lib/model/decoder_layers.py`)

Implement `SelfAttentionDecoderLayer`:
- Combines self-attention and feed-forward sublayers
- Pre-LN architecture (layer norm before sublayer)
- Residual connections

### 5. Decoder-Only Transformer (`hw4lib/model/transformers.py`)

Implement `DecoderOnlyTransformer`:
- Token embedding layer
- Positional encoding
- Stack of decoder layers
- Output projection to vocabulary
- Optional weight tying

### 6. Custom Attention (`mytorch/nn/`)

Implement from scratch:
- **ScaledDotProductAttention**: Core attention mechanism
- **MultiHeadAttention**: Multi-head attention module
- Match PyTorch's interface

### 7. Dataset (`hw4lib/data/lm_dataset.py`)

Implement language modeling dataset:
- Load and tokenize text data
- Create sequences for next-token prediction
- Handle batching and padding

## Setup Instructions

### Local Setup

1. **Create conda environment**:
```bash
conda create -n hw4p1 python=3.12.4
conda activate hw4p1
```

2. **Install dependencies**:
```bash
cd HW4P1/IDL-HW4/IDL-HW4
pip install --no-cache-dir --ignore-installed -r requirements.txt
```

3. **Verify setup**:
```bash
ls  # Should see hw4lib/, mytorch/, tests/, etc.
```

### Colab Setup

See `README.md` in the handout for detailed Colab setup instructions.

## Implementation Details

### Decoder-Only Architecture

```
Input Tokens
    ↓
Token Embedding
    ↓
Positional Encoding
    ↓
[Decoder Layer 1]
    ↓
[Decoder Layer 2]
    ↓
...
    ↓
[Decoder Layer N]
    ↓
Layer Norm
    ↓
Output Projection
    ↓
Logits (vocab_size)
```

### Pre-LN Architecture

- Layer normalization applied **before** each sublayer
- Residual connections wrap around sublayers
- Final layer norm before output projection

### Causal Masking

- Prevents attending to future tokens
- Upper triangular mask (all values above diagonal are -inf)
- Allows autoregressive generation

### Multi-Head Attention

- Split into multiple attention heads
- Each head operates on different representation subspaces
- Concatenate and project outputs

## Running Tests

Test individual components:

```bash
# Test masks
python -m tests.test_mask_causal
python -m tests.test_mask_padding

# Test positional encoding
python -m tests.test_positional_encoding

# Test sublayers
python -m tests.test_sublayer_selfattention
python -m tests.test_sublayer_feedforward

# Test decoder layers
python -m tests.test_decoderlayer_selfattention

# Test transformer
python -m tests.test_transformer_decoder_only

# Test dataset
python -m tests.test_dataset_lm

# Test custom attention
python -m tests.test_mytorch_scaled_dot_product_attention
python -m tests.test_mytorch_multi_head_attention
```

## Training

### Hyperparameters

Typical configuration:
- **d_model**: 256-512
- **num_layers**: 4-8
- **num_heads**: 8
- **d_ff**: 4 * d_model (1024-2048)
- **dropout**: 0.1-0.2
- **learning_rate**: 1e-4 to 1e-3
- **batch_size**: 16-64
- **max_len**: 512-1024

### Training Tips

1. **Start Small**:
   - Use small model initially
   - Train on subset of data
   - Verify implementation works

2. **Monitor Metrics**:
   - Perplexity (lower is better)
   - Training/validation loss
   - Generation quality

3. **Optimization**:
   - Use AdamW optimizer
   - Learning rate scheduling (cosine annealing)
   - Gradient clipping

## Key Concepts

### Attention Mechanism

- **Query (Q)**: What am I looking for?
- **Key (K)**: What do I contain?
- **Value (V)**: What information do I provide?
- **Attention**: Weighted combination of values based on query-key similarity

### Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

- Scale by sqrt(d_k) to prevent large dot products
- Softmax gives attention weights
- Weighted sum of values

### Causal Masking

- Ensures model can only see previous tokens
- Critical for autoregressive generation
- Prevents information leakage from future

### Positional Encoding

- Adds position information to embeddings
- Sinusoidal: Fixed, deterministic
- Learned: Trainable embeddings
- Helps model understand sequence order

## Common Issues and Solutions

1. **Shape Mismatches**:
   - Check tensor shapes at each layer
   - Verify attention mask shapes
   - Ensure batch dimension is correct

2. **Attention Issues**:
   - Verify masking is applied correctly
   - Check scaling factor (sqrt(d_k))
   - Ensure softmax is applied correctly

3. **Training Instability**:
   - Use gradient clipping
   - Adjust learning rate
   - Check initialization
   - Use warmup

4. **Memory Issues**:
   - Reduce batch size
   - Use gradient accumulation
   - Reduce sequence length
   - Use mixed precision training

## Submission

1. **Complete implementation**:
   - All components implemented
   - All tests passing
   - Model trains successfully

2. **Submit to Autolab**:
   - Follow submission instructions in writeup
   - Include all required files

## Resources

- Assignment README: `IDL-HW4/IDL-HW4/README.md`
- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- GPT Paper: https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
- PyTorch Transformer Tutorial: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
- The Illustrated Transformer: http://jalammar.github.io/illustrated-transformer/

## Notes

- This is Part 1 of HW4
- Many components will be reused in HW4P2 (Encoder-Decoder Transformer)
- Focus on understanding attention mechanisms
- Test thoroughly before training on full dataset

