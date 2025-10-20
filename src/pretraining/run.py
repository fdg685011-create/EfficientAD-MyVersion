# src/pretraining/run.py

# !/usr/bin/python
# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models import Wide_ResNet101_2_Weights
from tqdm import tqdm

# 注意这里的导入路径变化，我们从同一目录下的其他文件导入
from . import config as cfg
from .model import FeatureExtractor
from .utils import get_argparse, feature_normalization
# common.py 在上层目录(src/)，需要这样导入
from ..common import (get_pdn_small, get_pdn_medium,
                      ImageFolderWithoutTarget, InfiniteDataloader)


def main():
    """运行预训练的主函数。"""

    # 1. 初始化设置
    args = get_argparse()
    # 使用 exist_ok=True 避免在文件夹已存在时报错
    os.makedirs(args.output_folder, exist_ok=True)

    # 设置随机种子以保证结果可复现
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)

    # 2. 模型初始化
    print("正在初始化模型...")
    backbone = torchvision.models.wide_resnet101_2(
        weights=Wide_ResNet101_2_Weights.IMAGENET1K_V1)

    extractor = FeatureExtractor(
        backbone=backbone,
        layers_to_extract_from=cfg.LAYERS_TO_EXTRACT_FROM,
        device=cfg.DEVICE,
        input_shape=cfg.INPUT_SHAPE_EXTRACTOR
    )

    if cfg.MODEL_SIZE == 'small':
        pdn = get_pdn_small(cfg.OUT_CHANNELS, padding=True)
    elif cfg.MODEL_SIZE == 'medium':
        pdn = get_pdn_medium(cfg.OUT_CHANNELS, padding=True)
    else:
        raise ValueError(f"未知的模型尺寸: {cfg.MODEL_SIZE}")

    pdn.to(cfg.DEVICE)
    print("模型初始化完成。")

    # 3. 数据加载
    print("正在加载数据集...")
    train_set = ImageFolderWithoutTarget(
        cfg.IMAGENET_TRAIN_PATH,
        transform=cfg.get_train_transform
    )
    train_loader = DataLoader(
        train_set, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=True
    )
    # 使用无限数据加载器以匹配固定的训练步数
    train_loader_infinite = InfiniteDataloader(train_loader)
    print("数据集加载完成。")

    # 4. 预计算 (特征归一化)
    print("开始计算特征均值和标准差...")
    channel_mean, channel_std = feature_normalization(
        extractor=extractor,
        train_loader=train_loader,  # 使用有限的数据加载器进行计算
        device=cfg.DEVICE
    )
    print("特征归一化完成。")

    # 5. 核心训练循环
    pdn.train()
    optimizer = torch.optim.Adam(
        pdn.parameters(), lr=1e-4, weight_decay=1e-5
    )

    tqdm_obj = tqdm(range(cfg.TRAIN_STEPS), desc="开始训练")
    for iteration, (image_fe, image_pdn) in zip(tqdm_obj, train_loader_infinite):
        image_fe = image_fe.to(cfg.DEVICE)
        image_pdn = image_pdn.to(cfg.DEVICE)

        # 前向传播：计算目标和预测
        with torch.no_grad():
            target = extractor.embed(image_fe)
        target = (target - channel_mean) / channel_std
        prediction = pdn(image_pdn)
        loss = torch.mean((target - prediction) ** 2)

        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tqdm_obj.set_description(f'Loss: {loss.item():.4f}')

        # 模型检查点：每10000步保存一次临时模型
        if (iteration > 0 and iteration % 10000 == 0):
            tmp_path = os.path.join(args.output_folder, f'teacher_{cfg.MODEL_SIZE}_tmp_state.pth')
            torch.save(pdn.state_dict(), tmp_path)

    # 6. 保存最终模型
    # 我们通常只保存 state_dict，因为它更灵活
    final_path_state = os.path.join(args.output_folder, f'teacher_{cfg.MODEL_SIZE}_final_state.pth')
    torch.save(pdn.state_dict(), final_path_state)
    print(f"\n训练完成！最终模型权重已保存到: {final_path_state}")


if __name__ == '__main__':
    main()