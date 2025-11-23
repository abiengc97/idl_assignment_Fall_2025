# HW2P2: Face Recognition with Metric Learning

## Overview

This assignment focuses on building a face recognition system using deep metric learning. You will implement a ResNet-based architecture with ArcFace loss to learn discriminative face embeddings that can be used for face verification and identification tasks.

## Learning Objectives

- Understand metric learning and face recognition
- Implement ResNet architectures from scratch
- Apply ArcFace loss for improved face recognition
- Design effective data augmentation strategies
- Optimize hyperparameters for face recognition performance

## Task Description

Given face images, learn a mapping to a high-dimensional embedding space where:
- Faces of the same person are close together
- Faces of different people are far apart

The model is evaluated using Equal Error Rate (EER) on a face verification task (determining if two face images belong to the same person).

## Dataset

The dataset consists of face images with:
- **Training set**: Face images with identity labels
- **Validation set**: Face pairs for verification
- **Test set**: Face pairs for final evaluation (submitted to Kaggle)

### Data Characteristics
- Face images (typically 112x112 pixels after preprocessing)
- Large number of identities (8631 classes in training)
- High intra-class variation (same person, different poses/lighting)
- High inter-class similarity (different people may look similar)

## Assignment Structure

```
HW2P2/
├── HW2P2_Starter_Notebook.ipynb          # Main notebook for implementation
├── HW2P2_Checkpoint_Notebook(RUN_ON_COLAB).ipynb  # Checkpoint submission
├── HW2P2_Writeup_F25.pdf                 # Detailed assignment writeup
├── acknowledgement.txt                   # Submission acknowledgements
├── config.yaml                           # Configuration file (if used)
├── model_arch.txt                        # Model architecture description
├── checkpoint_submission.json           # Checkpoint metadata
├── submission.csv                        # Kaggle submission file
└── README.txt                            # Implementation notes
```

## Model Architecture

### ResNet-34 Backbone

Based on the implementation notes, the model uses:

```
Stem:
  Conv7x7 (stride=2) → BatchNorm → ReLU → MaxPool(3x3, stride=2)

Residual Blocks:
  [3, 4, 6, 3] BasicBlocks
  - Each BasicBlock: Conv3x3 → BN → ReLU → Conv3x3 → BN → Skip connection

Output:
  AdaptiveAvgPool2d → Flatten → 512-D feature vector
```

### Embedding Head

```
Linear(512 → 512) → L2 Normalization
```

### Classification Head (ArcFace)

```
ArcMarginProduct:
  - Scale (s): 30.0
  - Margin (m): 0.35
  - easy_margin: False
```

### Loss Function

- **CrossEntropy Loss** over ArcFace logits
- ArcFace adds angular margin to improve discrimination

## Implementation Requirements

### 1. ResNet Implementation

Build ResNet-34 from scratch:
- Implement BasicBlock with skip connections
- Stack blocks in [3, 4, 6, 3] configuration
- Handle downsampling in first block of each stage
- Apply batch normalization and ReLU activations

### 2. ArcFace Loss

Implement ArcMarginProduct:
- Compute cosine similarity between embeddings and weight vectors
- Add angular margin to target class
- Apply scale factor
- Compute cross-entropy loss

### 3. Data Loading

- Load face images
- Apply data augmentation (training only)
- Create identity-based sampling (if using triplet/pair sampling)
- Handle variable batch sizes

### 4. Training Loop

- Forward pass through ResNet
- Extract 512-D embeddings
- L2 normalize embeddings
- Compute ArcFace logits
- Compute loss and backpropagate
- Track training metrics

## Setup Instructions

### Option 1: Google Colab (Recommended)

1. **Upload notebook to Colab**:
   - Upload `HW2P2_Starter_Notebook.ipynb`
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
conda create -n hw2p2 python=3.9
conda activate hw2p2
```

2. **Install dependencies**:
```bash
pip install torch torchvision torchaudio
pip install wandb numpy pandas pillow
```

## Data Augmentation

### Training Transforms (from README.txt)

```python
Resize → RandomResizedCrop(112, scale=(0.9, 1.0), ratio=(0.98, 1.02))
RandomHorizontalFlip(p=0.5)
ToTensor() → ToDtype(torch.float32, scale=True)
Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
```

### Validation/Test Transforms

```python
Resize(112, 112)
ToTensor() → ToDtype(torch.float32, scale=True)
Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
```

## Training Strategy

### Hyperparameters (from README.txt)

**Initial Training**:
- Learning Rate: 0.001
- Optimizer: SGD with momentum=0.9, weight_decay=1e-4, nesterov=True
- Epochs: 100
- Batch Size: 64
- Result: 0.0417 EER

**Fine-tuning**:
- Learning Rate: 0.0005 (from checkpoint)
- Optimizer: SGD (same settings)
- Epochs: 100 additional
- Result: 0.03881 EER

**Final Training**:
- Learning Rate: 0.0001
- Optimizer: Adam with Cosine Annealing
- Result: 0.0243 EER

### Training Tips

1. **Progressive Training**:
   - Start with higher learning rate
   - Fine-tune with lower learning rate
   - Use learning rate scheduling

2. **Checkpointing**:
   - Save best model based on validation EER
   - Resume training from checkpoints
   - Experiment with different learning rates

3. **Monitoring**:
   - Track training/validation loss
   - Monitor EER on validation set
   - Use WandB for experiment tracking

## Evaluation

### Equal Error Rate (EER)

EER is the point where False Acceptance Rate (FAR) equals False Rejection Rate (FRR):
- Lower EER = Better performance
- Typical good EER: < 0.05 (5%)

### Verification Protocol

1. Extract embeddings for all face images
2. Compute cosine similarity between pairs
3. Set threshold to minimize EER
4. Evaluate on test pairs

### Submission Format

Generate CSV with predictions:
```csv
id,predicted
0,1  # 1 = same person, 0 = different person
1,0
...
```

## Implementation Notes

### ArcFace Details

ArcFace adds angular margin in the embedding space:
- Computes angle between embedding and class weight vector
- Adds margin to target class angle
- Improves discrimination between classes

### Embedding Normalization

L2 normalization is crucial:
- Ensures embeddings lie on unit sphere
- Makes cosine similarity equivalent to dot product
- Improves training stability

### ResNet Design Choices

- **Pre-activation**: BN → ReLU → Conv (better than post-activation)
- **Skip connections**: Help with gradient flow
- **Stem**: Initial downsampling reduces computation

## Checkpoint Submission

- **Due Date**: Check writeup for specific deadline
- **Requirements**:
  - Achieve minimum EER on Kaggle leaderboard
  - Complete checkpoint notebook quiz
  - Submit to Autolab

## Final Submission

- **Kaggle Competition**: Submit predictions to private leaderboard
- **Code Submission**: Submit final notebook to Autolab
- **Deadline**: See writeup for specific dates

## Important Notes

### Academic Integrity
- **NO pre-trained models**: Cannot use models from Hugging Face
- **NO external data**: Only use provided dataset
- **Own implementation**: Must implement ResNet and ArcFace from scratch
- **Own results**: Must submit your own code and results

### Submission Requirements
1. Set `ACKNOWLEDGED = True` in notebook
2. Provide Kaggle username for score verification
3. Code submission due 48 hours after Kaggle deadline
4. Late submissions must use Slack Kaggle competition

## Resources

- Assignment Writeup: `HW2P2_Writeup_F25.pdf`
- ArcFace Paper: https://arxiv.org/abs/1801.07698
- ResNet Paper: https://arxiv.org/abs/1512.03385
- PyTorch Documentation: https://pytorch.org/docs/
- Face Recognition Tutorials: Various online resources

## Troubleshooting

1. **Poor EER**:
   - Try different learning rates
   - Adjust ArcFace margin and scale
   - Experiment with data augmentation
   - Try different ResNet depths

2. **Training Instability**:
   - Use gradient clipping
   - Adjust learning rate
   - Check embedding normalization
   - Monitor gradient norms

3. **Overfitting**:
   - Increase data augmentation
   - Add dropout (if not using BN)
   - Increase weight decay
   - Use more diverse training data

## Example WandB Runs

From README.txt:
- Run 1: https://wandb.ai/agcheria-carnegie-mellon-university/hw2p2-ablations/runs/be5rg51v
- Run 2: https://wandb.ai/agcheria-carnegie-mellon-university/hw2p2-ablations/runs/r6oge9oj
- Run 3: https://wandb.ai/agcheria-carnegie-mellon-university/hw2p2-ablations/runs/i5vvq8k4

Use WandB to track experiments and compare different hyperparameters.

