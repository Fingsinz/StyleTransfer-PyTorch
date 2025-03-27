import os
from tqdm import tqdm
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.networks import VGG_3_8_15_22
from models.MetaNet_model import TransformNet, MetaNet

from data.ImageDataset import ImageDataset
import utils.config as Config
from utils.utils import mean_std, denormalize, create_grid, save_model, check_dir

import swanlab

def train(model_vgg, model_transform, metanet):
    model_vgg = model_vgg.to(Config.device)
    model_transform = model_transform.to(Config.device)
    metanet = metanet.to(Config.device)
    
    content_dataset, style_dataset = Config.get_data()
    
    if len(content_dataset) == 0 or len(style_dataset) == 0:
        raise ValueError("数据集为空")
    
    content_data_loader = DataLoader(content_dataset,
                                     batch_size=Config.get_training_batch(),
                                     shuffle=True,
                                     num_workers=Config.get_num_workers())
    
    trainable_params = {}
    trainable_param_shapes = {}
    for model in [model_vgg, model_transform, metanet]:
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params[name] = param
                trainable_param_shapes[name] = param.shape

    loss_content = torch.nn.functional.mse_loss
    loss_style = torch.nn.functional.mse_loss
    optimizer = torch.optim.Adam(trainable_params.values(), Config.get_lr())
    
    metanet.train()
    model_transform.train()
    
    style_weight, content_weight, tv_weight = Config.get_training_weight()
    record_per_epochs = Config.get_record_per_epochs()
    test_batch = Config.get_test_batch()
    epochs = Config.get_epochs()
    is_save = False if Config.get_model_save() == '' else True
    
    for epoch in range(epochs):
        content_loss_sum = 0
        style_loss_sum = 0
        avg_max_value = 0
        batch = 0
        
        for content in tqdm(content_data_loader, desc=f"{epoch + 1} / {epochs}"):
            if batch % 20 == 0:
                random_idx = random.randint(0, len(style_dataset) - 1)
                style_image = style_dataset[random_idx].unsqueeze(0).to(Config.device)
                style_features = model_vgg(style_image)
                style_mean_std = mean_std(style_features)
            
            # 检测是否为纯色，纯色不做处理
            x = content.cpu().numpy()
            if (x.min(-1).min(-1) == x.max(-1).max(-1)).any():
                continue
                    
            optimizer.zero_grad()
            weights = metanet(style_mean_std)
            model_transform.set_weights(weights, 0)
            content = content.to(Config.device)
            output = model_transform(content)
                    
            content_features = model_vgg(content)
            transformed_features = model_vgg(output)
            transformed_mean_std = mean_std(transformed_features)
                    
            content_loss = content_weight * loss_content(transformed_features[2], content_features[2])
            style_loss = style_weight * loss_style(transformed_mean_std,
                                                   style_mean_std.expand_as(transformed_mean_std))
            y = output
            tv_loss = tv_weight * (torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) +
                                                    torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))
                    
            total_loss = content_loss + style_loss + tv_loss
            total_loss.backward()
            optimizer.step()
                    
            max_value = max([x.max().item() for x in weights.values()])
                    
            content_loss_sum += content_loss.item()
            style_loss_sum += style_loss.item()
            avg_max_value += max_value
            
            batch += 1

        if Config.is_swanlab:        
            swanlab.log({"content_loss": content_loss_sum / len(content_data_loader),
                         "style_loss": style_loss_sum / len(content_data_loader),
                         "max_value": avg_max_value / len(content_data_loader)})

        print(f"{epoch + 1} / {epochs} | \
            content_loss: {content_loss_sum / len(content_data_loader)} | \
                style_loss: {style_loss_sum / len(content_data_loader)} | \
                    max_value: {avg_max_value / len(content_data_loader)} ")
        
        if (epoch + 1) % record_per_epochs == 0:
            if Config.is_save:
                result_dir = check_dir(Config.get_model_save())
                save_model(model_transform, result_dir, f"transform_{epoch + 1}.pth")
                save_model(metanet, result_dir, f"metanet_{epoch + 1}.pth")
                
            val_in_training(content_dataset, style_dataset,
                            model_vgg, model_transform, metanet, test_batch)

def val_in_training(content_dataset, style_dataset, model_vgg, model_transform, metanet, test_batch):
    random_idx = random.randint(0, len(style_dataset) - 1)
    style_tensor = style_dataset[random_idx].unsqueeze(0).to(Config.device)
    
    with torch.inference_mode():
        features = model_vgg(style_tensor)
        mean_std_features = mean_std(features)
        weights = metanet.forward(mean_std_features)
        model_transform.set_weights(weights)
    
        content_images = torch.stack([random.choice(content_dataset)
                                      for _ in range(test_batch)]).to(Config.device)
        transformed_images = model_transform(content_images)
    
        style_denorm = denormalize(style_tensor).squeeze(0)
        style_vis = style_denorm.cpu().permute(1, 2, 0).numpy()
        style_images = np.repeat(style_vis[np.newaxis], 4, axis=0)

        content_vis = denormalize(content_images).cpu().permute(0, 2, 3, 1).numpy()
        transformed_vis = denormalize(transformed_images).cpu().detach().permute(0, 2, 3, 1).numpy()

        merged_image = create_grid(style_images, content_vis, transformed_vis)
    
        if Config.is_swanlab:
            swanlab.log({'transformed_grid': swanlab.Image(merged_image)})

if __name__ == '__main__':
    if Config.is_swanlab:
        swanlab.init(
            project="StyleTransfer",
            experiment_name="MetaNet_demo_30",
            description="MetaNet demo",
        )
    
    vgg16 = VGG_3_8_15_22().eval()
    transform_net = TransformNet(Config.get_base()).to(Config.device)
    metanet = MetaNet(transform_net.get_param_dict()).to(Config.device)
    
    train(vgg16, transform_net, metanet)
    