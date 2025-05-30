import os
import time
from tqdm import tqdm
import random

from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.networks import VGG16_3_8_15_22, VGG19_3_8_17_26
from models.metanet_model import TransformNet, MetaNet

from data.image_dataset import ImageDataset
import utils.config as Config
from utils.utils import mean_std, denormalize, create_grid, save_model, check_dir, gram_matrix
from utils.recorder import Recorder

import swanlab

device = Config.device
style_weight, content_weight, tv_weight = Config.get_training_weight()
style_interval = Config.get_style_interval()
epochs = Config.get_epochs()

def train(model_vgg, model_transform, metanet, record_path):
    model_vgg = model_vgg.to(device)
    model_transform = model_transform.to(device)
    metanet = metanet.to(device)
    
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
    
    # 使用余弦退火学习率调度器
    schedulaer = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    # 增加梯度裁剪防止梯度爆炸
    torch.nn.utils.clip_grad_norm_(trainable_params.values(), max_norm=1.0)
    
    metanet.train()
    model_transform.train()
    
    record_per_epochs = Config.get_record_per_epochs()
    test_batch = Config.get_test_batch()
    
    is_save = False if Config.get_model_save() == '' else True
    
    logger = Recorder(record_path, "log.txt")
    Config.print_training_config(logger)
    recorder = Recorder(record_path, "loss.csv")
    recorder.set_statistic(["epoch", "content_loss", "style_loss", "lr"])

    for epoch in range(epochs):
        content_loss_sum = 0
        style_loss_sum = 0
        batch = 0
        
        for content in tqdm(content_data_loader, desc=f"{epoch + 1} / {epochs}", ncols=100):
            if batch % style_interval == 0:
                random_idx = random.randint(0, len(style_dataset) - 1)
                style_image = style_dataset[random_idx].unsqueeze(0).to(device)
                style_features = model_vgg(style_image)
                style_mean_std = mean_std(style_features)
            
            # 检测是否为纯色，纯色不做处理
            x = content.cpu().numpy()
            if (x.min(-1).min(-1) == x.max(-1).max(-1)).any():
                continue
                    
            optimizer.zero_grad()
            weights = metanet(style_mean_std)
            model_transform.set_weights(weights, 0)
            content = content.to(device)
            output = model_transform(content)
                    
            content_features = model_vgg(content)
            transformed_features = model_vgg(output)
            
            content_loss = content_weight * loss_content(transformed_features[2], content_features[2])
            
            # 均值和标准差损失
            transformed_mean_std = mean_std(transformed_features)
            style_loss = style_weight * loss_style(transformed_mean_std,
                                                   style_mean_std.expand_as(transformed_mean_std))   
            
            # Gram 矩阵损失
            # style_grams = []
            # for sf in style_features:
            #     sf_expanded = sf.repeat(transformed_features[0].shape[0], 1, 1, 1)
            #     style_grams.append(gram_matrix(sf_expanded))
            # transformed_grams = [gram_matrix(f) for f in transformed_features]
            
            # style_loss = 0
            # for sf, tf, w in zip(style_grams, transformed_grams, [1.0, 0.8, 0.5, 0.3]):
            #     style_loss += w * loss_style(tf, sf)
            # style_loss = style_weight * style_loss
            
            y = output
            tv_loss = tv_weight * (torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) +
                                                    torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))
                    
            total_loss = content_loss + style_loss + tv_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params.values(), max_norm=1.0) # 梯度裁剪
            
            optimizer.step()
            schedulaer.step()
                    
            content_loss_sum += content_loss.item()
            style_loss_sum += style_loss.item()
            
            batch += 1

        content_loss_value = content_loss_sum / len(content_data_loader)
        style_loss_value = style_loss_sum / len(content_data_loader)
        last_lr = schedulaer.get_last_lr()[0]
        recorder.record([f"{epoch + 1}", f"{content_loss_value}",
                         f"{style_loss_value}", f"{last_lr}"])

        if Config.is_swanlab:        
            swanlab.log({"content_loss": content_loss_value,
                         "style_loss": style_loss_value,
                         "lr": last_lr})

        logger.log(f"{epoch + 1} / {epochs} | content_loss: {content_loss_value} | "
                   f"style_loss: {style_loss_value} | lr: {last_lr}", )
        
        if (epoch + 1) % record_per_epochs == 0:
            if is_save:
                result_dir = check_dir(record_path + "pth/")
                save_model(model_transform, result_dir, f"transform_{epoch + 1}.pth")
                logger.log(f"[INFO] Model saved to {result_dir}transform_{epoch + 1}.pth")
                save_model(metanet, result_dir, f"metanet_{epoch + 1}.pth")
                logger.log(f"[INFO] Model saved to {result_dir}metanet_{epoch + 1}.pth")
                
            val_in_training(content_dataset, style_dataset,
                            model_vgg, model_transform, metanet,
                            test_batch, epoch + 1, logger, record_path)

def val_in_training(content_dataset, style_dataset, model_vgg, model_transform, metanet,
                    test_batch, epoch, logger, val_path=""):
    random_idx = random.randint(0, len(style_dataset) - 1)
    style_tensor = style_dataset[random_idx].unsqueeze(0).to(device)
    
    with torch.inference_mode():
        features = model_vgg(style_tensor)
        mean_std_features = mean_std(features)
        weights = metanet.forward(mean_std_features)
        model_transform.set_weights(weights)
    
        content_images = torch.stack([random.choice(content_dataset)
                                      for _ in range(test_batch)]).to(device)
        transformed_images = model_transform(content_images)
    
        style_denorm = denormalize(style_tensor).squeeze(0)
        style_vis = style_denorm.cpu().permute(1, 2, 0).numpy()
        style_images = np.repeat(style_vis[np.newaxis], test_batch, axis=0)

        content_vis = denormalize(content_images).cpu().permute(0, 2, 3, 1).numpy()
        transformed_vis = denormalize(transformed_images).cpu().detach().permute(0, 2, 3, 1).numpy()

        merged_image = create_grid(style_images, content_vis, transformed_vis)
        
        if val_path != "":
            val_path = check_dir(val_path + "png/")
            file_name = f"{val_path}transformed_grid_{epoch}.png"
            Image.fromarray(merged_image).save(file_name)
            logger.log(f"[INFO] Image saved to {file_name}")
    
        if Config.is_swanlab:
            swanlab.log({'transformed_grid': swanlab.Image(merged_image)})

if __name__ == '__main__':
    if Config.is_swanlab:
        swanlab.init(
            project="StyleTransfer",
            experiment_name="MetaNet_demo",
            description="MetaNet demo",
            config={
                "device": device,
                "epochs": epochs,
                "content_weight": content_weight,
                "style_weight": style_weight,
                "tv_weight": tv_weight,
                "style_interval": style_interval,
            }
        )
    
    vgg = None
    if Config.vgg_version == 16:
        vgg = VGG16_3_8_15_22().to(Config.device).eval()
    elif Config.vgg_version == 19:
        vgg = VGG19_3_8_17_26().to(Config.device).eval()

    vgg = vgg.to(device)
    transform_net = TransformNet(Config.get_base()).to(device)

    attention = Config.get_attention()
    metanet = MetaNet(transform_net.get_param_dict(), attention).to(device)

    now_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    record_path = check_dir(f"../results/{now_time}/")
    
    train(vgg, transform_net, metanet, record_path)
    