# HW1P2: Frame-Level Speech Recognition

## Overview

This assignment focuses on building a neural network model for frame-level phoneme recognition from speech data. You will work with MFCC (Mel-Frequency Cepstral Coefficients) features extracted from audio recordings and train a model to predict phonemes at each time frame.

## Learning Objectives

- Understand speech signal processing and MFCC features
- Build and train deep neural networks for sequence classification
- Implement data loading and preprocessing pipelines
- Optimize hyperparameters for speech recognition tasks
- Participate in a Kaggle competition for model evaluation

## Task Description

Given MFCC features (28-dimensional vectors) at each time frame, predict the phoneme class for that frame. The model processes sequences of MFCC features and outputs phoneme predictions for each frame.

## Dataset

The dataset consists of:
- **train-clean-100**: Training data with MFCC features and phoneme transcripts
- **dev-clean**: Validation data for hyperparameter tuning
- **test-clean**: Test data (no labels) for Kaggle submission

### Data Structure
```
train-clean-100/
├── mfcc/          # MFCC feature files (*.npy)
└── transcript/    # Phoneme transcript files (*.npy)

dev-clean/
├── mfcc/
└── transcript/

test-clean/
└── mfcc/          # No transcripts (for prediction)
```

### MFCC Features
- 28-dimensional feature vectors per time frame
- Extracted from audio using mel spectrogram analysis
- Each `.npy` file contains a sequence of MFCC frames

## Assignment Structure

```
HW1P2/
├── HW1P2_F25_Student_Final.ipynb          # Main notebook for implementation
├── HW1P2_F25_Checkpoint_Notebook_(RUN_ON_COLAB_ONLY).ipynb  # Checkpoint submission
├── HW1P2_Writeup_Fall_25.pdf              # Detailed assignment writeup
├── acknowledgement.txt                    # Submission acknowledgements
├── model_arch.txt                         # Model architecture description
├── model_metadata_2025-09-15_22-17.json   # Model metadata
├── submission.csv                         # Kaggle submission file
└── README.txt                             # Implementation notes
```

## Implementation Requirements

### Model Architecture
You need to design and implement a neural network that:
- Takes sequences of MFCC features as input
- Processes temporal information effectively
- Outputs phoneme class predictions for each frame
- Handles variable-length sequences

### Key Components

1. **Data Loading**:
   - Load MFCC features from `.npy` files
   - Load corresponding phoneme transcripts
   - Handle variable sequence lengths
   - Create train/validation splits

2. **Model Design**:
   - Choose appropriate architecture (MLP, CNN, RNN, or combinations)
   - Handle sequence processing
   - Implement proper padding/masking for variable lengths
   - Output layer for phoneme classification

3. **Training**:
   - Implement training loop
   - Use appropriate loss function (CrossEntropy for classification)
   - Implement validation loop
   - Track metrics (accuracy, loss)

4. **Inference**:
   - Generate predictions on test set
   - Format predictions for Kaggle submission

## Setup Instructions

### Option 1: Google Colab (Recommended)

1. **Upload notebook to Colab**:
   - Upload `HW1P2_F25_Student_Final.ipynb` to Google Colab
   - Set runtime to GPU (Runtime → Change runtime type → GPU)

2. **Install dependencies**:
```python
!pip install torch torchvision torchaudio
!pip install wandb  # Optional: for experiment tracking
```

3. **Download dataset**:
   - Follow instructions in notebook to download Kaggle dataset
   - Extract data to appropriate directory

### Option 2: Local Setup

1. **Create conda environment**:
```bash
conda create -n hw1p2 python=3.9
conda activate hw1p2
```

2. **Install dependencies**:
```bash
pip install torch torchvision torchaudio
pip install wandb numpy pandas
```

3. **Download dataset**:
   - Download from Kaggle competition
   - Extract to `data/` directory

## Model Architecture Example

Based on the implementation notes, a successful architecture might include:

- **Input**: MFCC features (batch_size, sequence_length, 28)
- **Feature Extraction**: 
  - Convolutional layers for local pattern extraction
  - Or MLP layers for frame-level processing
- **Sequence Processing**:
  - RNN/LSTM layers for temporal modeling
  - Or Transformer layers for attention-based modeling
- **Classification Head**:
  - Fully connected layers
  - Output: (batch_size, sequence_length, num_phonemes)

### Example Architecture (from README.txt)
- Pyramid architecture with ~19.95M parameters
- Multiple layers with decreasing dimensions
- ReLU activations
- Batch normalization

## Training Strategy

### Hyperparameters
- **Learning Rate**: Start with 0.001, adjust based on validation performance
- **Batch Size**: Typically 32-128 depending on GPU memory
- **Optimizer**: Adam or SGD with momentum
- **Epochs**: Train until convergence or early stopping

### Data Augmentation
- Consider augmenting MFCC features
- Time warping, frequency masking (if applicable)
- Use lower probability for augmentation to avoid over-augmentation

### Training Tips
1. **Start Simple**: Begin with a basic MLP or CNN before adding complexity
2. **Monitor Metrics**: Track both training and validation accuracy/loss
3. **Early Stopping**: Stop training when validation performance plateaus
4. **Checkpointing**: Save model checkpoints for best validation performance
5. **Learning Rate Scheduling**: Use learning rate decay or cosine annealing

## Evaluation

### Metrics
- **Frame Accuracy**: Percentage of correctly predicted frames
- **Kaggle Score**: Private leaderboard score (primary evaluation)

### Submission Format
Generate a CSV file with predictions:
```csv
id,predicted
0,phoneme_class_0
1,phoneme_class_1
...
```

## Checkpoint Submission

- **Due Date**: Check writeup for specific deadline
- **Requirements**: 
  - Achieve minimum score on Kaggle leaderboard
  - Complete checkpoint notebook quiz
  - Submit to Autolab

## Final Submission

- **Kaggle Competition**: Submit predictions to private leaderboard
- **Code Submission**: Submit final notebook to Autolab
- **Deadline**: See writeup for specific dates

## Important Notes

### Academic Integrity
- **NO pre-trained models**: Cannot use models from Hugging Face or similar
- **NO external data**: Only use provided dataset
- **Own implementation**: Must implement models using fundamental PyTorch operations
- **Own results**: Must submit your own code and results (even if collaborating on experiments)

### Submission Requirements
1. Set `ACKNOWLEDGED = True` in notebook after reading all requirements
2. Provide Kaggle username for score verification
3. Code submission due 48 hours after Kaggle deadline
4. Late submissions must use Slack Kaggle competition

## Resources

- Assignment Writeup: `HW1P2_Writeup_Fall_25.pdf`
- PyTorch Documentation: https://pytorch.org/docs/
- Speech Recognition Tutorials: Various online resources
- MFCC Explanation: https://en.wikipedia.org/wiki/Mel-frequency_cepstrum

## Troubleshooting

1. **Memory Issues**:
   - Reduce batch size
   - Use gradient accumulation
   - Process shorter sequences

2. **Poor Performance**:
   - Try different architectures
   - Adjust learning rate
   - Add regularization (dropout, weight decay)
   - Experiment with data augmentation

3. **Overfitting**:
   - Add dropout layers
   - Increase weight decay
   - Use data augmentation
   - Reduce model complexity

## Model Metadata

The `model_metadata_*.json` file contains:
- Model architecture details
- Training hyperparameters
- Performance metrics
- WandB run information (if used)

## Example WandB Run

From README.txt:
- Final run: https://wandb.ai/agcheria-carnegie-mellon-university/hw1p2/runs/w0zpsvte

Use WandB to track experiments and compare different architectures/hyperparameters.

