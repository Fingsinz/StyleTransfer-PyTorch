import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

import utils.config as Config
from models.networks import VGG_3_8_15_22
from models.MetaNet_model import TransformNet, MetaNet
from utils.utils import mean_std, denormalize

def one_image_transfer(content_path, style_path, model_vgg, model_transform, metanet):
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),          # 强制缩放至256x256
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    style_image = Image.open(style_path).convert('RGB')
    style_tensor = preprocess(style_image).unsqueeze(0).to(Config.device)

    content_image = Image.open(content_path).convert('RGB')
    content_tensor = preprocess(content_image).unsqueeze(0).to(Config.device)

    with torch.inference_mode():
        style_features = model_vgg(style_tensor)
        style_mean_std = mean_std(style_features)
        weights = metanet(style_mean_std)
        model_transform.set_weights(weights, 0)

    model_transform.eval()
    with torch.inference_mode():
        transformed_tensor = model_transform(content_tensor)
    model_transform.train()

    content_vis = denormalize(content_tensor).squeeze(0).cpu().permute(1, 2, 0).numpy()
    transformed_vis = denormalize(transformed_tensor).squeeze(0).cpu().permute(1, 2, 0).numpy()
    style_vis = denormalize(style_tensor).squeeze(0).cpu().permute(1, 2, 0).numpy() 

    comparison = np.concatenate([content_vis, style_vis, transformed_vis], axis=1)
    plt.imshow(comparison)
    plt.axis('off')
    plt.savefig(f"./{os.path.basename(content_path).split('.')[0]}+{os.path.basename(style_path).split('.')[0]}.png")
    plt.close()
    
if __name__ == "__main__":
    
    vgg16 = VGG_3_8_15_22().to(Config.device).eval()
    
    load_transform_net = TransformNet(Config.get_base()).to(Config.device)
    load_transform_net.load_state_dict(torch.load('./transform_net_100.pth',
                                                  map_location=Config.device, weights_only=True))
    
    load_metanet = MetaNet(load_transform_net.get_param_dict()).to(Config.device)
    load_metanet.load_state_dict(torch.load('./MetaNet_100.pth',
                                            map_location=Config.device, weights_only=True))
    
    one_image_transfer(content_path='./content1.JPEG', style_path='./style1.jpg',
                       model_vgg=vgg16, model_transform=load_transform_net, metanet=load_metanet)
    