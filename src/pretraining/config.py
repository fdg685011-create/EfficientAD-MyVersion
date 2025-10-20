# src/pretraining/config.py
import torch
from torchvision import transforms

MODEL_SIZE = 'small'
IMAGENET_TRAIN_PATH = './ILSVRC/Data/CLS-LOC/train' # 请根据实际情况修改
SEED = 42
BATCH_SIZE = 16
NUM_WORKERS = 7
TRAIN_STEPS = 60000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_CHANNELS = 384
INPUT_SHAPE_EXTRACTOR = (3, 512, 512)
LAYERS_TO_EXTRACT_FROM = ['layer2', 'layer3']

grayscale_transform = transforms.RandomGrayscale(0.1)  # apply same to both

#为将要送入 extractor 模型（即 wide_resnet101_2）的图片准备的
extractor_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
#提供给将要训练的模型
pdn_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_train_transform(image):
    image = grayscale_transform(image)
    return extractor_transform(image), pdn_transform(image)