import os
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from typing import Tuple, Dict, List

def gram_matrix(feature):
    b, c, h, w = feature.size()
    features = feature.view(b * c, h * w)
    gram = torch.mm(features, features.t())
    return gram.div(b * c * h * w)

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

def denormalize(tensor: torch.Tensor):
    """张量的反向归一化，输入已归一化为 0-1 范围"""
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
    return torch.clamp((tensor * std + mean), 0, 1)

def create_grid(styles: list, contents: list, transformed: list) -> np.ndarray:
    """创建包含样式、内容和转换后的图像的图像网格"""
    grid = []
    for s, c, t in zip(styles, contents, transformed):
        row = np.concatenate([s, c, t], axis=1)
        grid.append(row)
    full_grid = np.concatenate(grid, axis=0)
    return (full_grid * 255).astype(np.uint8)

def check_dir(path: str) -> str:
    """检查目录是否存在，如果不存在则创建目录"""
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def save_model(model: torch.nn.Module, save_path: str, model_name: str) -> None:
    """保存模型"""
    check_dir(save_path)
    assert model_name.endswith('.pth'), 'model name should end with .pth'
    model_save_path = save_path + model_name
    
    torch.save(obj=model.state_dict(), f=model_save_path)
    print(f"[INFO] Model saved to {model_save_path}")

def load_model(model: torch.nn.Module, path: str) -> torch.nn.Module:
    """加载模型"""
    model.load_state_dict(torch.load(path, weights_only=True))
    return model

def calculate_ssim(content: np.ndarray, transformed: np.ndarray) -> float:
    """计算 SSIM 值"""
    ssim_score = []
    for channel in range(3):
        score = ssim(content[:, :, channel],
                     transformed[:, :, channel],
                     multichannel=False,
                     data_range=content.max() - content.min()
        )
        ssim_score.append(score)

    return np.mean(ssim_score)

def calculate_psnr(content: np.ndarray, transformed: np.ndarray) -> float:
    """计算 PSNR 值，[0,255]"""
    mse = np.mean((content - transformed) ** 2)
    psnr = 20 * np.log10(255 / np.sqrt(mse))
    return psnr 


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """在目标目录中查找类文件夹名称，并创建类到索引的映射"""
    # 1. 通过扫描目标目录获取类名
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    # 2. 如果找不到类名，则引发错误
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")        
    # 3. 创建索引标签的字典
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx
