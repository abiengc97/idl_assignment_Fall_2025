- **Model**: Encoder-Decoder Transformer architecture for ASR with Pre-LN (Layer Normalization) design. The model consists of:
  - Speech embedding layer with convolutional time reduction (factor 2) to downsample input features from 80-dim log mel filterbank features
  - Encoder stack: 4 layers with 8 attention heads each, d_model=256, d_ff=1024
  - Decoder stack: 4 layers with 8 attention heads each, d_model=256, d_ff=1024
  - Cross-attention mechanism in decoder layers to attend to encoder outputs
  - CTC auxiliary head for joint training with attention-based decoder
  - Positional encoding applied to both encoder and decoder
  - Dropout rate: 0.05, no layer dropout or weight tying

- **Training Strategy**: 
  - Optimizer: AdamW with learning rate 0.0003, weight decay 0.0001, betas [0.9, 0.999]
  - Scheduler: Cosine annealing with linear warmup (3 epochs, start_factor 0.1, T_max=50, eta_min=0.00001)
  - Loss function: Cross-entropy loss with label smoothing (0.05) + CTC auxiliary loss (weight 0.2) for joint training
  - Mixed precision training enabled for faster training
  - Gradient accumulation: 1 step
  - Batch size: 16
  - Tokenization: 5k subword BPE tokenizer

- **Augmentations**: SpecAugment data augmentation applied during training:
  - Frequency masking: 2 masks per sample with width range of 5 frequency bands
  - Time masking: 2 masks per sample with width range of 30 time steps
  - Applied only during training, not during validation/testing
  https://wandb.ai/agcheria-carnegie-mellon-university/HW4P2/runs/4r4hchz6