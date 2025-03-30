from pathlib import Path
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import swanlab
from tqdm import tqdm

from models.networks import VGG19_3_8_17_26
from models.unet_model import UNet
from utils.utils import gram_matrix, check_dir, save_model, load_model

def train(is_swanlab=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    content_image = Image.open(config["content_path"]).convert('RGB')
    content_img = transform(content_image).unsqueeze(0).to(config["device"])
    style_image = Image.open(config["style_path"]).convert('RGB')
    style_img = transform(style_image).unsqueeze(0).to(config["device"])

    # 初始化网络
    generator = UNet().to(config["device"])
    extractor = VGG19_3_8_17_26().to(config["device"])
    optimizer = optim.Adam(generator.parameters(), lr=0.002)

    # 获取参考特征
    content_features = extractor(content_img)
    style_features = extractor(style_img)
    style_grams = [gram_matrix(f) for f in style_features]

    # 训练循环
    loss = 0
    bar = tqdm(range(config["epochs"]), ncols=80)
    for step in bar:
        generated = generator(content_img)
        gen_features = extractor(generated)
        
        # 内容损失（使用conv3_4层）
        content_loss = torch.mean((gen_features[2] - content_features[2])**2)
        
        # 风格损失（使用conv1_2, conv2_2, conv3_4层）
        style_loss = 0
        for i in [0, 1, 2]:
            gen_gram = gram_matrix(gen_features[i])
            style_loss += torch.mean((gen_gram - style_grams[i])**2)
        
        # 总损失
        total_loss = config["content_weight"] * content_loss + config["style_weight"] * style_loss
        
        # 优化步骤
        loss = total_loss.item()
        bar.set_description(f"loss: {loss:.4f}")
        bar.update()
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if (step + 1) % 50 == 0 or (step + 1) == config["epochs"]:
            if is_swanlab:
                swanlab.log({"unet_content_loss": content_loss,
                            "unet_style_loss": style_loss,
                            "unet_total_loss": total_loss.item()})
                output = generated.clone().detach().cpu().squeeze()
                output = output * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
                output += torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)  
                image = swanlab.Image(output.clamp(0,1), caption="example")
                swanlab.log({"Output": image})
            
        if (step + 1) == config["epochs"]:
            # 保存生成图像
            save_path = check_dir("../output/")
            output = generated.clone().detach().cpu().squeeze()
            output = output * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
            output += torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)  
            save_image(output.clamp(0,1), f"{save_path}unet_{config['style']}.png")

    print("Training completed!")

    model_path = check_dir("../output/unet/")
    save_model(generator, model_path, f"unet_{config['style']}_model.pth")

def test(generator, content_img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    generator = load_model(generator, f"../output/unet/unet_{config['style']}_model.pth")
    
    content = transform(content_img).unsqueeze(0).to(config["device"])
    generated = generator(content)
    
    output = generated.clone().detach().cpu().squeeze()
    output = output * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    output += torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)  
    save_path = check_dir("../output/")
    save_image(output.clamp(0,1), f"{save_path}unet_{config['style']}_test.png")

config= {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "content_path": "",
    "style_path": "",
    "style": "monet",
    "epochs": 500,
    "content_weight": 1,       # 内容损失权重
    "style_weight": 1e6       # 风格损失权重
}

if __name__ == "__main__":
    config["content_path"] = sys.argv[1]
    config["style_path"] = sys.argv[2] if len(sys.argv) > 2 else ""
    
    is_swanlab = sys.argv[3] if len(sys.argv) > 3 else False

    if is_swanlab:
        run = swanlab.init(
            project="StyleTransfer",
            experiment_name="vgg_unet",
            description="vgg + U-Net(优化后) 进行特定风格迁移",
            config=config
        )
    
    if config["style_path"] != "":
        train(is_swanlab)
    else:
        content_img = Image.open(config["content_path"]).convert('RGB')
        generator = UNet().to(config["device"])
        test(generator, content_img)

