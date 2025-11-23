# HW4P2: Encoder-Decoder Transformer for Automatic Speech Recognition

## Overview

This assignment focuses on building an end-to-end automatic speech recognition (ASR) system using an Encoder-Decoder Transformer architecture. You will process speech features (log mel filterbank features) and generate text transcriptions using attention-based sequence-to-sequence modeling with CTC auxiliary loss.

## Learning Objectives

- Understand encoder-decoder transformer architecture
- Implement cross-attention mechanisms
- Build speech embedding layers with time reduction
- Apply CTC loss for joint training
- Implement beam search decoding for ASR
- Optimize transformer models for speech recognition

## Task Description

Given log mel filterbank features extracted from speech audio, predict the corresponding text transcription. The model uses an encoder to process speech features and a decoder to generate text tokens, with cross-attention connecting the two.

### Input
- Speech features: 80-dimensional log mel filterbank features per time frame
- Variable-length audio sequences

### Output
- Text transcriptions
- Variable-length sequences of subword tokens

## Dataset

The dataset consists of LibriSpeech data:
- **train-clean-100**: Training data with speech features and text transcripts
- **dev-clean**: Validation data for hyperparameter tuning
- **test-clean**: Test data (no transcripts) for Kaggle submission

### Data Structure
```
hw4_data_subset/
└── hw4p2_data/           # For end-to-end speech recognition
    ├── dev-clean/
    │   ├── fbank/        # Log mel filterbank features
    │   └── text/         # Text transcripts
    ├── test-clean/
    │   └── fbank/
    └── train-clean-100/
        ├── fbank/
        └── text/
```

## Assignment Structure

```
HW4P2/
├── HW4P2_Student_Starter_Notebook.ipynb      # Main notebook
├── HW4P2_Student_Checkpoint_Notebook.ipynb   # Checkpoint submission
├── F25_HW4P2.pdf                             # Detailed writeup
├── config.yaml                                # Configuration file
├── hw4lib/                                    # Main library
│   ├── data/
│   │   ├── asr_dataset.py                    # ASR dataset
│   │   └── tokenizer.py                      # Subword tokenizer
│   ├── model/
│   │   ├── masks.py                          # Padding and causal masks
│   │   ├── positional_encoding.py            # Positional encoding
│   │   ├── speech_embedding.py               # Speech feature embedding
│   │   ├── sublayers.py                      # Attention and FFN sublayers
│   │   ├── encoder_layers.py                 # Encoder layer
│   │   ├── decoder_layers.py                 # Decoder layer (with cross-attention)
│   │   └── transformers.py                   # Encoder-decoder transformer
│   ├── decoding/
│   │   └── sequence_generator.py             # Beam search decoder
│   └── trainers/
│       └── asr_trainer.py                    # ASR trainer
├── mytorch/                                   # Custom attention (from HW4P1)
├── tests/                                     # Test suite
└── README.txt                                 # Implementation notes
```

## Model Architecture

### Encoder-Decoder Transformer

Based on the implementation notes and config.yaml:

**Encoder**:
- Speech embedding with time reduction (factor 2)
- 4 encoder layers
- 8 attention heads per layer
- d_model = 256, d_ff = 1024
- Pre-LN architecture

**Decoder**:
- Token embedding
- 4 decoder layers with cross-attention
- 8 attention heads per layer
- d_model = 256, d_ff = 1024
- Causal masking for autoregressive generation

**Auxiliary Components**:
- CTC head for joint training
- Output projection to vocabulary
- Positional encoding (optional for encoder/decoder)

### Architecture Details (from config.yaml)

```yaml
model:
  input_dim: 80                    # Speech feature dimension
  time_reduction: 2                 # Time dimension downsampling
  reduction_method: 'conv'          # 'lstm', 'conv', or 'both'
  d_model: 256
  num_encoder_layers: 4
  num_decoder_layers: 4
  num_encoder_heads: 8
  num_decoder_heads: 8
  d_ff_encoder: 1024
  d_ff_decoder: 1024
  dropout: 0.05
```

## Components to Implement

### 1. Speech Embedding (`hw4lib/model/speech_embedding.py`)

Implement `SpeechEmbedding`:
- Convolutional time reduction (downsample time dimension)
- Project to d_model dimension
- Options: 'conv', 'lstm', or 'both'

### 2. Encoder Layer (`hw4lib/model/encoder_layers.py`)

Implement `SelfAttentionEncoderLayer`:
- Self-attention sublayer
- Feed-forward sublayer
- Pre-LN with residual connections

### 3. Decoder Layer (`hw4lib/model/decoder_layers.py`)

Implement `CrossAttentionDecoderLayer`:
- Self-attention sublayer (causal masking)
- Cross-attention sublayer (encoder-decoder attention)
- Feed-forward sublayer
- Pre-LN with residual connections

### 4. Encoder-Decoder Transformer (`hw4lib/model/transformers.py`)

Implement `EncoderDecoderTransformer`:
- Speech embedding
- Encoder stack
- Decoder stack
- CTC head (optional)
- Output projection

### 5. Sequence Generator (`hw4lib/decoding/sequence_generator.py`)

Implement beam search decoding:
- Maintain top-k hypotheses
- Expand hypotheses token by token
- Handle end-of-sequence tokens
- Return best hypothesis

### 6. ASR Dataset (`hw4lib/data/asr_dataset.py`)

Implement ASR dataset:
- Load speech features and transcripts
- Apply SpecAugment (training only)
- Tokenize text using subword tokenizer
- Handle variable-length sequences

## Setup Instructions

### Local Setup

1. **Create conda environment**:
```bash
conda create -n hw4p2 python=3.12.4
conda activate hw4p2
```

2. **Install dependencies**:
```bash
cd HW4P2
pip install --no-cache-dir --ignore-installed -r requirements.txt
```

3. **Verify setup**:
```bash
ls  # Should see hw4lib/, mytorch/, tests/, etc.
```

### Colab/PSC Setup

See the main README.md in HW4P1 handout for detailed setup instructions.

## Training Strategy

### Hyperparameters (from config.yaml)

**Optimizer**:
- AdamW with lr=0.0003
- Weight decay: 0.0001
- Betas: [0.9, 0.999]

**Scheduler**:
- Cosine annealing with warmup
- Warmup epochs: 3
- T_max: 50
- eta_min: 0.00001

**Loss**:
- Cross-entropy with label smoothing (0.05)
- CTC auxiliary loss (weight: 0.2)
- Joint training: `loss = ce_loss + 0.2 * ctc_loss`

**Training**:
- Batch size: 16
- Gradient accumulation: 1
- Mixed precision: Enabled
- Epochs: 50+

### Data Augmentation

**SpecAugment** (from config.yaml):
- Frequency masking: 2 masks, width range 5
- Time masking: 2 masks, width range 30
- Applied only during training

### Training Tips

1. **Progressive Training**:
   - Start with smaller model
   - Train on subset of data
   - Gradually increase complexity

2. **Loss Balancing**:
   - Start with higher CTC weight
   - Gradually reduce CTC weight
   - Monitor both losses

3. **Learning Rate**:
   - Use warmup for stability
   - Cosine annealing for convergence
   - Monitor validation loss

4. **Checkpointing**:
   - Save best model based on validation WER
   - Resume from checkpoints
   - Experiment with different hyperparameters

## Implementation Details

### Speech Embedding

**Time Reduction**:
- Convolutional: Conv1d with stride=2
- LSTM: Bidirectional LSTM with pooling
- Both: Combine both methods

**Purpose**:
- Reduce sequence length for efficiency
- Extract higher-level features
- Reduce computational cost

### Cross-Attention

**Mechanism**:
- Query (Q) from decoder
- Key (K) and Value (V) from encoder
- Allows decoder to attend to encoder outputs
- Critical for sequence-to-sequence tasks

**Implementation**:
```python
# In decoder layer
cross_attn_output = cross_attention(
    query=decoder_hidden,
    key=encoder_output,
    value=encoder_output,
    mask=encoder_padding_mask
)
```

### CTC Auxiliary Loss

**Purpose**:
- Helps with alignment learning
- Provides additional supervision
- Improves convergence

**Implementation**:
- Compute CTC loss from encoder outputs
- Combine with cross-entropy loss
- Weight: 0.2 (from config)

### Beam Search Decoding

**Algorithm**:
1. Initialize with start token
2. Expand top-k hypotheses
3. Score each hypothesis
4. Prune to top-k
5. Continue until end token or max length

**Parameters** (from config.yaml):
- Beam width: 4
- Temperature: 1.0
- Repeat penalty: 1.2

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
python -m tests.test_sublayer_crossattention
python -m tests.test_sublayer_feedforward

# Test encoder/decoder layers
python -m tests.test_encoderlayer_selfattention
python -m tests.test_decoderlayer_selfattention
python -m tests.test_decoderlayer_crossattention

# Test transformer
python -m tests.test_transformer_encoder_decoder

# Test dataset
python -m tests.test_dataset_asr

# Test decoding
python -m tests.test_decoding
```

## Evaluation

### Metrics

- **Word Error Rate (WER)**: Primary metric for ASR
- **Character Error Rate (CER)**: Alternative metric
- **Kaggle Score**: Private leaderboard score

### Decoding

- **Greedy**: Fast but suboptimal
- **Beam Search**: Better accuracy, slower
- **Beam Width**: 4-6 typically

## Common Issues and Solutions

1. **Poor WER**:
   - Adjust CTC weight
   - Try different architectures
   - Experiment with data augmentation
   - Use beam search instead of greedy

2. **Training Instability**:
   - Use gradient clipping
   - Adjust learning rate
   - Check loss balancing
   - Use warmup

3. **Memory Issues**:
   - Reduce batch size
   - Use gradient accumulation
   - Reduce sequence length
   - Use mixed precision

4. **Slow Training**:
   - Use time reduction
   - Optimize attention implementation
   - Use efficient data loading
   - Consider model parallelism

## Submission

1. **Complete implementation**:
   - All components implemented
   - All tests passing
   - Model trains successfully

2. **Kaggle Submission**:
   - Generate predictions on test set
   - Submit to Kaggle competition

3. **Code Submission**:
   - Submit to Autolab
   - Include all required files

## Important Notes

### Academic Integrity
- **NO pre-trained models**: Cannot use models from Hugging Face
- **NO external data**: Only use provided dataset
- **Own implementation**: Must implement transformer from scratch
- **Own results**: Must submit your own code and results

### Submission Requirements
1. Set `ACKNOWLEDGED = True` in notebook
2. Provide Kaggle username for score verification
3. Code submission due 48 hours after Kaggle deadline
4. Late submissions must use Slack Kaggle competition

## Resources

- Assignment Writeup: `F25_HW4P2.pdf`
- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- SpecAugment Paper: https://arxiv.org/abs/1904.08779
- CTC Paper: https://www.cs.toronto.edu/~graves/icml_2006.pdf
- PyTorch Transformer Tutorial: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

## Example Configuration

From `config.yaml` and `README.txt`:
- Encoder-Decoder Transformer with Pre-LN
- 4 encoder layers, 4 decoder layers
- 8 attention heads, d_model=256
- CTC auxiliary loss (weight 0.2)
- SpecAugment data augmentation
- Beam search decoding (width 4)
- AdamW optimizer with cosine annealing

## WandB Tracking

From README.txt:
- Final run: https://wandb.ai/agcheria-carnegie-mellon-university/HW4P2/runs/4r4hchz6

Use WandB to track experiments and compare different hyperparameters.

