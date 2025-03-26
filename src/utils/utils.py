import torch
import numpy as np

def mean_std(features):
    """输入 VGG16 计算的四个特征，输出每张特征图的均值和标准差，长度为特征拼接"""
    mean_std_features = []
    for x in features:
        batch, C, H, W = x.shape
        x_flat = x.view(batch, C, -1)
        mean = x_flat.mean(dim=-1)
        std = torch.sqrt(x_flat.var(dim=-1) + 1e-5)
        feature = torch.cat([mean, std], dim=1)
        mean_std_features.append(feature)
    return torch.cat(mean_std_features, dim=-1)

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
    return torch.clamp((tensor * std + mean), 0, 1)

def create_grid(styles, contents, transformed):
    grid = []
    for s, c, t in zip(styles, contents, transformed):
        row = np.concatenate([s, c, t], axis=1)
        grid.append(row)
    full_grid = np.concatenate(grid, axis=0)
    return (full_grid * 255).astype(np.uint8)
