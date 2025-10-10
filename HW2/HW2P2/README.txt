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
    
    I first trained with lr 0.001 for 100 epochs and got 0.0417 and then lowered the lr to 0.0005 and trained for 100 more epochs and got 0.03881 EER both both I used SGD with momentum 0.9 and weight decay 1e-4 and nesterov True then I retrained with lr of 0.0001 and used Adam with Cosine Annealing and got 0.0243 EER
    Run 1: https://wandb.ai/agcheria-carnegie-mellon-university/hw2p2-ablations/runs/be5rg51v?nw=nwuseragcheria trained for 100 epochs from scratch
    Run 2: https://wandb.ai/agcheria-carnegie-mellon-university/hw2p2-ablations/runs/r6oge9oj?nw=nwuseragcheria trained for 100 epochs from scratch with lr 0.0005 with run 1 checkpoint
    Run 3: https://wandb.ai/agcheria-carnegie-mellon-university/hw2p2-ablations/runs/i5vvq8k4?nw=nwuseragcheria stoped early at 17 epochs trained with lr 0.0001 with run 2 checkpoint
