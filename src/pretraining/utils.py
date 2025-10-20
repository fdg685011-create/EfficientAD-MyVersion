# src/pretraining/utils.py
import argparse
import torch
from tqdm import tqdm

def get_argparse():
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')
    parser.add_argument('-o', '--output_folder',
                        default='output/pretraining/1/')
    return parser.parse_args()

@torch.no_grad()
def feature_normalization(extractor, train_loader, device, steps=10000):

    mean_outputs = []
    normalization_count = 0
    with tqdm(desc='Computing mean of features', total=steps) as pbar:
        for image_fe, _ in train_loader:
            image_fe = image_fe.to(device)

            output = extractor.embed(image_fe)
            mean_output = torch.mean(output, dim=[0, 2, 3])
            mean_outputs.append(mean_output)
            normalization_count += len(image_fe)
            if normalization_count >= steps:
                pbar.update(steps - pbar.n)
                break
            else:
                pbar.update(len(image_fe))
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    #这样做是为了让 channel_mean 的维度数量（4维）与特征图 output 的维度数量（[Batch, Channels, Height, Width]）相匹配
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    normalization_count = 0
    with tqdm(desc='Computing variance of features', total=steps) as pbar:
        for image_fe, _ in train_loader:
            image_fe = image_fe.to(device)

            output = extractor.embed(image_fe)
            distance = (output - channel_mean) ** 2
            mean_distance = torch.mean(distance, dim=[0, 2, 3])
            mean_distances.append(mean_distance)
            normalization_count += len(image_fe)
            if normalization_count >= steps:
                pbar.update(steps - pbar.n)
                break
            else:
                pbar.update(len(image_fe))
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)  #  对每个通道的方差 channel_var 进行开平方根运算，得到最终需要的标准差

    return channel_mean, channel_std
