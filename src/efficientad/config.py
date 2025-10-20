# src/efficientad/config.py

import torch
from torchvision import transforms

# =============================================================================
# Training Configuration
# =============================================================================
SEED = 42
# 注意：TRAIN_STEPS 的默认值现在移到了 run.py 的 argparse 中
# TRAIN_STEPS = 70000

# =============================================================================
# Device Configuration
# =============================================================================
ON_GPU = torch.cuda.is_available()
DEVICE = 'cuda' if ON_GPU else 'cpu'

# =============================================================================
# Model & Feature Constants
# =============================================================================
OUT_CHANNELS = 384
IMAGE_SIZE = 256

# =============================================================================
# Data Transformation Settings
# =============================================================================
DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

TRANSFORM_AE = transforms.RandomChoice([
    transforms.ColorJitter(brightness=0.2),
    transforms.ColorJitter(contrast=0.2),
    transforms.ColorJitter(saturation=0.2)
])

def get_train_transform(image):
    """对输入图像应用不同的变换，分别用于学生模型和自编码器。"""
    return DEFAULT_TRANSFORM(image), DEFAULT_TRANSFORM(TRANSFORM_AE(image))