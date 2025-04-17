import os
import time
import sys
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

import utils.config as Config
from models.networks import VGG16_3_8_15_22, VGG19_3_8_17_26
from models.metanet_model import TransformNet, MetaNet
from utils.utils import mean_std, denormalize, check_dir, load_model

def one_image_transfer(content_img_path, style_img_path, model_vgg, model_transform, metanet):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    content_image = Image.open(content_img_path).convert('RGB')
    content_tensor = preprocess(content_image).unsqueeze(0).to(Config.device)
    content_img_width, content_img_height = content_image.size

    style_image = Image.open(style_img_path).convert('RGB')
    style_tensor = preprocess(style_image).unsqueeze(0).to(Config.device)

    with torch.inference_mode():
        style_features = model_vgg(style_tensor)
        style_mean_std = mean_std(style_features)
        weights = metanet(style_mean_std)
        model_transform.set_weights(weights, 0)

    model_transform.eval()
    with torch.inference_mode():
        transformed_tensor = model_transform(content_tensor)
    model_transform.train()

    transformed_vis = denormalize(transformed_tensor).squeeze(0).cpu().permute(1, 2, 0).numpy()
    transformed_pil = Image.fromarray((transformed_vis * 255).astype(np.uint8))
    transformed_pil = transformed_pil.resize((content_img_width, content_img_height), Image.LANCZOS)

    out_dir = check_dir('../output')
    filename = os.path.basename(style_img_path).split('.')[0] + '_' + \
        os.path.basename(content_img_path).split('.')[0] + '.png'

    output_path = f"./{out_dir}/{filename}"
    transformed_pil.save(output_path)
    print(f'[INFO] "{content_img_path}" to "{style_img_path}". Result saved to {output_path}')
    
if __name__ == "__main__":
    
    content_path = sys.argv[1]
    style_path = sys.argv[2]
    
    if not os.path.exists(content_path) or not os.path.exists(style_path):
        print(f"[ERROR] {content_path} or {style_path} not exist")
        exit()
    
    vgg = None
    if Config.vgg_version == 16:
        vgg = VGG16_3_8_15_22().to(Config.device).eval()
    elif Config.vgg_version == 19:
        vgg = VGG19_3_8_17_26().to(Config.device).eval()
    
    metanet_path, transformnet_path = Config.get_inference_model_path()
    
    load_transform_net = TransformNet(Config.get_base()).to(Config.device)
    load_model(load_transform_net, transformnet_path)
    load_transform_net.to(Config.device)
    
    attention = Config.get_attention()
    load_metanet = MetaNet(load_transform_net.get_param_dict(), attention).to(Config.device)
    load_model(load_metanet, metanet_path)
    load_metanet.to(Config.device)
    
    Config.print_inference_config()
    
    if os.path.isfile(content_path) and os.path.isfile(style_path):
        start = time.perf_counter()
        one_image_transfer(content_img_path=content_path, style_img_path=style_path,
                           model_vgg=vgg, model_transform=load_transform_net, metanet=load_metanet)
        end = time.perf_counter()
        print(f"[INFO] 生成图片用时：{end - start} 秒")
        exit()
    elif os.path.isdir(content_path) and os.path.isdir(style_path):
        total_seconds = 0
        total_imgs = 0
        for content_img_path in os.listdir(content_path):
            for style_img_path in os.listdir(style_path):
                total_imgs += 1
                start = time.perf_counter()
                one_image_transfer(content_img_path=f"{content_path}/{content_img_path}",
                                   style_img_path=f"{style_path}/{style_img_path}",
                                   model_vgg=vgg, model_transform=load_transform_net, metanet=load_metanet)
                end = time.perf_counter()
                total_seconds += end - start
        print(f"[INFO] 生成 {total_imgs} 张图片平均用时：{total_seconds / total_imgs} 秒")
        exit()
    else:
        print(f"[ERROR] {content_path} or {style_path} not exist")
        exit()

    