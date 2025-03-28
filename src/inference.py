import os
import time
import sys
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

import utils.config as Config
from models.networks import VGG16_3_8_15_22
from models.MetaNet_model import TransformNet, MetaNet
from utils.utils import mean_std, denormalize, check_dir, load_model

def one_image_transfer(content_path, style_path, model_vgg, model_transform, metanet):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    content_image = Image.open(content_path).convert('RGB')
    content_tensor = preprocess(content_image).unsqueeze(0).to(Config.device)
    content_img_width, content_img_height = content_image.size

    style_image = Image.open(style_path).convert('RGB')
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
    filename = os.path.basename(content_path).split('.')[0] + \
        os.path.basename(style_path).split('.')[0] + '.png'

    output_path = f"./{out_dir}/{filename}"
    transformed_pil.save(output_path)
    print(f"[INFO] Result saved to {output_path}")
    
if __name__ == "__main__":
    
    content_img_path = sys.argv[1]
    style_img_path = sys.argv[2]
    
    if not os.path.exists(content_img_path) or not os.path.exists(style_img_path):
        print(f"[ERROR] {content_img_path} or {style_img_path} not exist")
        exit()
    
    vgg16 = VGG16_3_8_15_22().to(Config.device).eval()
    
    load_transform_net = TransformNet(Config.get_base()).to(Config.device)
    load_model(load_transform_net, './transform_net_100.pth')
    load_transform_net.to(Config.device)
    
    load_metanet = MetaNet(load_transform_net.get_param_dict()).to(Config.device)
    load_model(load_metanet, './metanet_100.pth')
    load_metanet.to(Config.device)
    
    start = time.perf_counter()
    one_image_transfer(content_path=content_img_path, style_path=style_img_path,
                       model_vgg=vgg16, model_transform=load_transform_net, metanet=load_metanet)
    end = time.perf_counter()
    print(f"[INFO] 生成图片用时：{end - start} 秒")

    