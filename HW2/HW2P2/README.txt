# Configuration, Model, and Transforms Summary

- **Config**:
  - batch_size: 64  
  - lr: 0.0005  
  - momentum: 0.9  
  - weight_decay: 1e-4  
  - nesterov: True  
  - epochs: 100  
  - num_classes: 8631  
  - augment: True  

- **Model**:
  - Backbone: Custom **ResNet-34**
    - Stem: Conv7x7 (stride=2) → BN → ReLU → MaxPool(3x3, stride=2)
    - Residual layers: [3, 4, 6, 3] BasicBlocks
    - Output: 512-D feature vector after AdaptiveAvgPool2d + flatten  
  - Embedding head: Linear(512 → 512) + L2 normalization  
  - Classification head: **ArcMarginProduct** with
    - s = 30.0  
    - m = 0.35  
    - easy_margin = False  
  - Loss: CrossEntropy over ArcFace logits  

- **Transforms**:
  - **Train**:
    - Resize → RandomResizedCrop(112, scale=(0.9, 1.0), ratio=(0.98, 1.02))  
    - RandomHorizontalFlip(p=0.5)  
    - ToTensor() → ToDtype(torch.float32, scale=True)  
    - Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])  
  - **Validation/Test**:
    - Resize(112,112)  
    - ToTensor() → ToDtype(torch.float32, scale=True)  
    - Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    
    I first trained with lr 0.001 for 100 epochs and got 0.0417 and then lowered the lr to 0.0005 and trained for 100 more epochs and got 0.03881