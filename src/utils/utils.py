import os
import torch
import numpy as np

def mean_std(features) -> torch.Tensor:
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

def check_dir(path: str)-> str:
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def save_model(model: torch.nn.Module, save_path: str, model_name: str):
    check_dir(save_path)
    assert model_name.endswith('.pth'), 'model name should end with .pth'
    model_save_path = os.path.join(save_path, model_name)
    
    torch.save(obj=model.state_dict(), f=model_save_path)
    print(f"[INFO] Model saved to {model_save_path}")

def load_model(model: torch.nn.Module, path: str) -> torch.nn.Module:
    model.load_state_dict(torch.load(path, weights_only=True))
    return model
