# src/efficientad/run.py

import os
import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import itertools
from tqdm import tqdm
from torchvision import transforms

# 从本模块导入配置和工具函数
from . import config as cfg
from . import utils

# 从上层共享模块导入模型和数据加载器类
from ..common import (get_autoencoder, get_pdn_small, get_pdn_medium,
                      ImageFolderWithoutTarget, ImageFolderWithPath,
                      InfiniteDataloader)


def get_argparse():
    """
    直接在主程序中定义和解析命令行参数。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='mvtec_ad',
                        choices=['mvtec_ad', 'mvtec_loco'])
    parser.add_argument('-s', '--subdataset', default='bottle',
                        help='Mvtec AD或Mvtec LOCO的子数据集名称')
    parser.add_argument('-o', '--output_dir', default='results/efficientad_1',
                        help='输出结果的根目录')
    parser.add_argument('-m', '--model_size', default='small',
                        choices=['small', 'medium'])
    parser.add_argument('-w', '--weights', default='models/teacher_small.pth',
                        help='预训练的教师模型权重路径')
    parser.add_argument('-i', '--imagenet_train_path',
                        default='none',
                        help='ImageNet路径，用于启用预训练惩罚项')
    parser.add_argument('-a', '--mvtec_ad_path',
                        default='./datasets/MVTec',
                        help='Mvtec AD 数据集路径')
    parser.add_argument('-b', '--mvtec_loco_path',
                        default='./datasets/mvtec_loco_anomaly_detection',
                        help='Mvtec LOCO 数据集路径')
    parser.add_argument('-t', '--train_steps', type=int, default=70000,
                        help='总训练步数')
    return parser.parse_args()


def main():
    # 1. 初始化和配置加载
    # 直接调用本文件内的 get_argparse 函数来读取命令行参数
    args = get_argparse()

    print("=" * 40)
    print("命令行参数加载成功:")
    print(f"  - 模型尺寸: {args.model_size}")
    print(f"  - 数据集: {args.dataset} / {args.subdataset}")
    print(f"  - MVTec AD 路径: {args.mvtec_ad_path}")
    print(f"  - 输出目录: {args.output_dir}")
    print(f"  - 教师模型权重: {args.weights}")
    print("=" * 40)

    # 2. 设置随机种子
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)

    # 3. 确定并验证数据集路径
    if args.dataset == 'mvtec_ad':
        dataset_path = args.mvtec_ad_path
    elif args.dataset == 'mvtec_loco':
        dataset_path = args.mvtec_loco_path
    else:
        raise ValueError("未知的数据集名称")

    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"错误：数据集路径不存在 -> {dataset_path}")

    # 4. 创建输出目录
    train_output_dir = os.path.join(args.output_dir, 'trainings',
                                    args.dataset, args.subdataset)
    test_output_dir = os.path.join(args.output_dir, 'anomaly_maps',
                                   args.dataset, args.subdataset, 'test')
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    # 5. 加载数据集
    full_train_set = ImageFolderWithoutTarget(
        os.path.join(dataset_path, args.subdataset, 'train'),
        transform=transforms.Lambda(cfg.get_train_transform)
    )
    test_set = ImageFolderWithPath(
        os.path.join(dataset_path, args.subdataset, 'test')
    )

    if args.dataset == 'mvtec_ad':
        train_size = int(0.9 * len(full_train_set))
        validation_size = len(full_train_set) - train_size
        rng = torch.Generator().manual_seed(cfg.SEED)
        train_set, validation_set = torch.utils.data.random_split(
            full_train_set, [train_size, validation_size], rng
        )
    elif args.dataset == 'mvtec_loco':
        train_set = full_train_set
        validation_set = ImageFolderWithoutTarget(
            os.path.join(dataset_path, args.subdataset, 'validation'),
            transform=transforms.Lambda(cfg.get_train_transform)
        )
    else:
        raise ValueError("未知的数据集名称")

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                              num_workers=4, pin_memory=True)
    train_loader_infinite = InfiniteDataloader(train_loader)
    validation_loader = DataLoader(validation_set, batch_size=1)

    # 6. 加载ImageNet惩罚项数据（如果需要）
    if args.imagenet_train_path != 'none':
        penalty_transform = transforms.Compose([
            transforms.Resize((2 * cfg.IMAGE_SIZE, 2 * cfg.IMAGE_SIZE)),
            transforms.RandomGrayscale(0.3),
            transforms.CenterCrop(cfg.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        penalty_set = ImageFolderWithoutTarget(
            args.imagenet_train_path, transform=penalty_transform
        )
        penalty_loader = DataLoader(
            penalty_set, batch_size=1, shuffle=True, num_workers=4,
            pin_memory=True
        )
        penalty_loader_infinite = InfiniteDataloader(penalty_loader)
    else:
        penalty_loader_infinite = itertools.repeat(None)

    # 7. 创建模型
    if args.model_size == 'small':
        teacher = get_pdn_small(cfg.OUT_CHANNELS)
        student = get_pdn_small(2 * cfg.OUT_CHANNELS)
    elif args.model_size == 'medium':
        teacher = get_pdn_medium(cfg.OUT_CHANNELS)
        student = get_pdn_medium(2 * cfg.OUT_CHANNELS)
    else:
        raise ValueError("未知的模型尺寸")

    state_dict = torch.load(args.weights, map_location='cpu')
    teacher.load_state_dict(state_dict)
    autoencoder = get_autoencoder(cfg.OUT_CHANNELS)

    teacher.to(cfg.DEVICE)
    student.to(cfg.DEVICE)
    autoencoder.to(cfg.DEVICE)

    # 8. 预计算教师模型特征
    teacher.eval()
    teacher_mean, teacher_std = utils.teacher_normalization(
        teacher, train_loader, cfg.DEVICE
    )

    # 9. 训练设置
    student.train()
    autoencoder.train()
    optimizer = torch.optim.Adam(
        itertools.chain(student.parameters(), autoencoder.parameters()),
        lr=1e-4, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(0.95 * args.train_steps), gamma=0.1
    )

    # 10. 核心训练循环
    tqdm_obj = tqdm(range(args.train_steps))
    for iteration, (image_st, image_ae), image_penalty in zip(
            tqdm_obj, train_loader_infinite, penalty_loader_infinite):

        image_st = image_st.to(cfg.DEVICE)
        image_ae = image_ae.to(cfg.DEVICE)
        if image_penalty is not None:
            image_penalty = image_penalty.to(cfg.DEVICE)

        with torch.no_grad():
            teacher_output_st = teacher(image_st)
            teacher_output_st = (teacher_output_st - teacher_mean) / teacher_std
        student_output_st = student(image_st)[:, :cfg.OUT_CHANNELS]
        distance_st = (teacher_output_st - student_output_st) ** 2
        d_hard = torch.quantile(distance_st, q=0.999)
        loss_hard = torch.mean(distance_st[distance_st >= d_hard])

        if image_penalty is not None:
            student_output_penalty = student(image_penalty)[:, :cfg.OUT_CHANNELS]
            loss_penalty = torch.mean(student_output_penalty ** 2)
            loss_st = loss_hard + loss_penalty
        else:
            loss_st = loss_hard

        ae_output = autoencoder(image_ae)
        with torch.no_grad():
            teacher_output_ae = teacher(image_ae)
            teacher_output_ae = (teacher_output_ae - teacher_mean) / teacher_std
        student_output_ae = student(image_ae)[:, cfg.OUT_CHANNELS:]
        distance_ae = (teacher_output_ae - ae_output) ** 2
        distance_stae = (ae_output - student_output_ae) ** 2
        loss_ae = torch.mean(distance_ae)
        loss_stae = torch.mean(distance_stae)
        loss_total = loss_st + loss_ae + loss_stae

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        if iteration % 10 == 0:
            tqdm_obj.set_description(
                f"当前损失: {loss_total.item():.4f}")

        if iteration > 0 and iteration % 10000 == 0:
            print(f"\n在步骤 {iteration} 进行中期评估...")
            student.eval()
            autoencoder.eval()
            q_st_start, q_st_end, q_ae_start, q_ae_end = utils.map_normalization(
                validation_loader, teacher, student, autoencoder,
                teacher_mean, teacher_std, cfg.DEVICE, cfg.OUT_CHANNELS
            )
            auc = utils.test(
                test_set, teacher, student, autoencoder, teacher_mean, teacher_std,
                q_st_start, q_st_end, q_ae_start, q_ae_end, cfg.DEVICE,
                cfg.DEFAULT_TRANSFORM, cfg.OUT_CHANNELS
            )
            print(f'中期评估AUC: {auc:.2f}')
            student.train()
            autoencoder.train()

    # 11. 最终评估
    print("\n训练完成，进行最终评估...")
    student.eval()
    autoencoder.eval()
    q_st_start, q_st_end, q_ae_start, q_ae_end = utils.map_normalization(
        validation_loader, teacher, student, autoencoder, teacher_mean,
        teacher_std, cfg.DEVICE, cfg.OUT_CHANNELS
    )
    auc = utils.test(
        test_set, teacher, student, autoencoder, teacher_mean, teacher_std,
        q_st_start, q_st_end, q_ae_start, q_ae_end, cfg.DEVICE,
        cfg.DEFAULT_TRANSFORM, cfg.OUT_CHANNELS, test_output_dir
    )
    print(f'最终评估AUC: {auc:.2f}')

    # 12. 保存最终模型
    torch.save(student.state_dict(), os.path.join(train_output_dir, 'student_final.pth'))
    torch.save(autoencoder.state_dict(), os.path.join(train_output_dir, 'autoencoder_final.pth'))
    print("最终模型已保存。")


if __name__ == '__main__':
    main()