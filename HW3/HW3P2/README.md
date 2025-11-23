# HW3P2: Sequence-to-Sequence Automatic Speech Recognition

## Overview

This assignment focuses on building an end-to-end automatic speech recognition (ASR) system using RNNs and CTC loss. You will process MFCC features from speech audio and generate text transcriptions using a sequence-to-sequence architecture with encoder-decoder design.

## Learning Objectives

- Build complete ASR pipeline from raw features to text
- Implement encoder-decoder architecture for sequence-to-sequence tasks
- Apply CTC loss for training without frame-level alignments
- Implement beam search decoding for improved accuracy
- Optimize hyperparameters for speech recognition performance

## Task Description

Given MFCC features extracted from speech audio, predict the corresponding text transcription. The model processes variable-length audio sequences and outputs variable-length text sequences.

### Input
- MFCC features: 28-dimensional vectors per time frame
- Variable-length audio sequences

### Output
- Text transcriptions
- Variable-length sequences of characters/words

## Dataset

The dataset consists of:
- **train-clean-100**: Training data with MFCC features and text transcripts
- **dev-clean**: Validation data for hyperparameter tuning
- **test-clean**: Test data (no transcripts) for Kaggle submission

### Data Structure
```
train-clean-100/
├── mfcc/          # MFCC feature files (*.npy)
└── transcript/    # Text transcript files

dev-clean/
├── mfcc/
└── transcript/

test-clean/
└── mfcc/         # No transcripts (for prediction)
```

## Assignment Structure

```
HW3P2/
├── 11_485_11_685_11_785_HW3P2_F25_Student_Starter_Notebook.ipynb  # Main notebook
├── HW3P2_Checkpoint_Notebook(RUN_ON_COLAB).ipynb  # Checkpoint submission
├── 11785_IDL_Homework_3_Part_2_Student_Writeup_Fall_2025.pdf     # Writeup
├── config.yaml                    # Configuration file
├── acknowledgement.txt            # Submission acknowledgements
├── model_arch.txt                 # Model architecture description
├── model_metadata_2025-11-06_19-48.json  # Model metadata
└── HW3P2_final_submission.zip    # Final submission
```

## Model Architecture

### Encoder-Decoder Architecture

Based on the configuration file, the model uses:

**Encoder**:
- Conv1d embedding layer (kernel_size=15) for initial feature extraction
- Multi-layer LSTM/RNN for sequence encoding
- Hidden dimension: 256 (embed_size)
- Dropout: 0.2 (encoder), 0.2 (LSTM), 0.25 (decoder)

**Decoder**:
- Linear projection to vocabulary size
- CTC head for loss computation
- Beam search for decoding

### Architecture Details (from config.yaml)

```yaml
input_size: 28              # MFCC feature dimension
kernel_size: 15             # Conv1d kernel size
embed_size: 256             # LSTM hidden dimension
encoder_dropout: 0.2
lstm_dropout: 0.2
decoder_dropout: 0.25
```

## Implementation Requirements

### 1. Data Loading

- Load MFCC features from `.npy` files
- Load text transcripts
- Create vocabulary (character-level or subword)
- Handle variable-length sequences
- Apply padding and masking

### 2. Encoder

Implement encoder that:
- Takes MFCC sequences as input
- Processes through Conv1d embedding
- Encodes with LSTM/RNN layers
- Outputs encoded representations

### 3. Decoder/CTC Head

- Linear projection to vocabulary
- CTC loss computation
- Handle blank tokens

### 4. Training Loop

- Forward pass through encoder
- Compute CTC loss
- Backpropagate and update weights
- Track training metrics

### 5. Decoding

- Implement greedy decoding
- Implement beam search decoding
- Remove blank tokens
- Collapse repetitions

## Setup Instructions

### Option 1: Google Colab (Recommended)

1. **Upload notebook to Colab**:
   - Upload `11_485_11_685_11_785_HW3P2_F25_Student_Starter_Notebook.ipynb`
   - Set runtime to GPU

2. **Install dependencies**:
```python
!pip install torch torchvision torchaudio
!pip install wandb  # For experiment tracking
```

3. **Download dataset**:
   - Follow instructions in notebook
   - Extract data to appropriate directory

### Option 2: Local Setup

1. **Create conda environment**:
```bash
conda create -n hw3p2 python=3.9
conda activate hw3p2
```

2. **Install dependencies**:
```bash
pip install torch torchvision torchaudio
pip install wandb numpy pandas
```

## Training Strategy

### Hyperparameters (from config.yaml)

**Initial Training**:
- Learning Rate: 1e-4
- Epochs: 250
- Batch Size: 96
- Optimizer: Adam/AdamW
- Scheduler: Cosine annealing with warmup

**Data Augmentation**:
- SpecAugment: Frequency and time masking
- Frequency mask: width range 5, 2 masks
- Time mask: width range 30, 2 masks

**Beam Search**:
- Training beam width: 6
- Test beam width: 6

### Training Tips

1. **Start Simple**:
   - Begin with greedy decoding
   - Add beam search later
   - Use smaller model initially

2. **Progressive Training**:
   - Train with subset of data first
   - Gradually increase data size
   - Fine-tune hyperparameters

3. **Monitoring**:
   - Track CTC loss
   - Monitor character error rate (CER)
   - Use WandB for experiment tracking

4. **Checkpointing**:
   - Save best model based on validation CER
   - Resume from checkpoints
   - Experiment with different learning rates

## Data Augmentation

### SpecAugment

Apply frequency and time masking:
- **Frequency Masking**: Mask consecutive frequency bands
- **Time Masking**: Mask consecutive time steps
- Helps model generalize better
- Applied only during training

### Configuration (from config.yaml)

```yaml
freq_mask_param: 30
time_mask_param: 30
num_freq_mask: 2
num_time_mask: 2
```

## Evaluation

### Metrics

- **Character Error Rate (CER)**: Percentage of incorrect characters
- **Word Error Rate (WER)**: Percentage of incorrect words
- **Kaggle Score**: Private leaderboard score (primary evaluation)

### Decoding Strategies

1. **Greedy Decoding**:
   - Fast but may be suboptimal
   - Good for initial training

2. **Beam Search**:
   - Maintains top-k hypotheses
   - Better accuracy but slower
   - Beam width: 4-6 typically

### Submission Format

Generate text predictions for test set:
- One transcription per audio file
- Submit to Kaggle competition

## Implementation Notes

### CTC Loss

- No need for frame-level alignments
- Handles variable input/output lengths
- Blank token allows "skipping" outputs
- Use CTC loss from HW3P1 or PyTorch's `CTCLoss`

### Sequence Processing

- Handle variable-length sequences
- Use padding and masking
- Process in batches efficiently

### Vocabulary

- Character-level: Each character is a token
- Subword-level: Use BPE or similar
- Handle special tokens (blank, start, end)

## Checkpoint Submission

- **Due Date**: Check writeup for specific deadline
- **Requirements**:
  - Achieve minimum CER on Kaggle leaderboard
  - Complete checkpoint notebook quiz
  - Submit to Autolab

## Final Submission

- **Kaggle Competition**: Submit transcriptions to private leaderboard
- **Code Submission**: Submit final notebook to Autolab
- **Deadline**: See writeup for specific dates

## Important Notes

### Academic Integrity
- **NO pre-trained models**: Cannot use models from Hugging Face
- **NO external data**: Only use provided dataset
- **Own implementation**: Must implement encoder-decoder from scratch
- **Own results**: Must submit your own code and results

### Submission Requirements
1. Set `ACKNOWLEDGED = True` in notebook
2. Provide Kaggle username for score verification
3. Code submission due 48 hours after Kaggle deadline
4. Late submissions must use Slack Kaggle competition

## Resources

- Assignment Writeup: `11785_IDL_Homework_3_Part_2_Student_Writeup_Fall_2025.pdf`
- CTC Paper: https://www.cs.toronto.edu/~graves/icml_2006.pdf
- SpecAugment Paper: https://arxiv.org/abs/1904.08779
- PyTorch CTC Documentation: https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
- ASR Tutorials: Various online resources

## Troubleshooting

1. **Poor CER**:
   - Try different architectures (LSTM vs GRU)
   - Adjust learning rate
   - Experiment with data augmentation
   - Use beam search instead of greedy

2. **Training Instability**:
   - Use gradient clipping
   - Adjust learning rate
   - Check data preprocessing
   - Monitor gradient norms

3. **Memory Issues**:
   - Reduce batch size
   - Use gradient accumulation
   - Process shorter sequences
   - Use mixed precision training

4. **Slow Training**:
   - Use GPU acceleration
   - Optimize data loading (num_workers)
   - Reduce model size
   - Use efficient decoding

## Model Metadata

The `model_metadata_*.json` file contains:
- Model architecture details
- Training hyperparameters
- Performance metrics
- WandB run information (if used)

## Example Configuration

From `config.yaml`:
- EfficientNet-based embedding with curriculum learning
- Progressive augmentation strategy
- Cosine annealing with warmup
- Beam search decoding

Use the configuration file to manage hyperparameters and experiment settings.

