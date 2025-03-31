"""
基于卷积神经网络的风格迁移

Reference: 
    - Gatys et al. "A Neural Algorithm of Artistic Style"
    - https://zh-v2.d2l.ai/chapter_computer-vision/neural-style.html
"""

import os
import sys

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from models.networks import VGG19_Pretrained
from utils.utils import gram_matrix, check_dir

class Feature_Extractor(torch.nn.Module):
    def __init__(self):
        super(Feature_Extractor, self).__init__()
        self.model = VGG19_Pretrained([0, 5, 10, 19, 25, 28])
        self.content_layers = [25]
        self.style_layers = [0, 5, 10, 19, 28]
        
    def extract_features(self, x):
        contents = []
        styles = []
        
        for i in range(len(self.model)):
            x = self.model[i](x)
            if i in self.content_layers:
                contents.append(x)
            if i in self.style_layers:
                styles.append(x)                
        return contents, styles
    
class SynthesizedImage(torch.nn.Module):
    def __init__(self, img_shape):
        super(SynthesizedImage, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(img_shape).requires_grad_(True))
    
    def forward(self):
        return self.weight
    
def train(model, content_path, style_path):
    content_img = Image.open(content_path).convert('RGB')
    style_img = Image.open(style_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    content_X = transform(content_img).unsqueeze(0).to(config["device"])
    contents_Y, _ = model.extract_features(content_X)
    style_X = transform(style_img).unsqueeze(0).to(config["device"])
    _, styles_Y = model.extract_features(style_X)
    style_Y_gram = [gram_matrix(Y) for Y in styles_Y]
    
    generated = SynthesizedImage(content_X.shape).to(config["device"])
    
    optimizer = torch.optim.Adam(generated.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, 0.8)
    
    def content_loss(Y_hat, Y):
        return torch.square(Y_hat - Y).mean()
    
    def style_loss(Y_hat, gram_Y):
        return torch.square(gram_matrix(Y_hat) - gram_Y.detach()).mean()
    
    def tv_loss(Y_hat):
        return 0.5 * (torch.square(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() + \
            torch.square(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())
    
    with tqdm(range(config["epochs"]), ncols=80) as bar:
        for step in bar:
            optimizer.zero_grad()
            contents_Y_hat, styles_Y_hat = model.extract_features(generated())
            loss_content = [content_loss(Y_hat, Y) * config["content_weight"]
                            for Y_hat, Y in zip(contents_Y_hat, contents_Y)]
            loss_style = [style_loss(Y_hat, Y) * config["style_weight"]
                        for Y_hat, Y in zip(styles_Y_hat, style_Y_gram)]
            loss_tv = tv_loss(generated()) * config["tv_weight"]
            loss = sum(10 * loss_style + loss_content + [loss_tv])
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            bar.set_description(f"loss: {loss.item():.4f}")
            bar.update()            
    return generated
     
config = {
    "content_weight": 10,
    "style_weight": 1e4,
    "tv_weight": 10,
    "lr": 0.3,
    "epochs": 500,
    "style": "monet",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

if __name__ == "__main__":
    content_path = sys.argv[1]
    style_path = sys.argv[2]
    
    if not os.path.exists(content_path) or not os.path.exists(style_path):
        print(f"[ERROR] {content_path} or {style_path} 不存在")
        exit()
    
    model = Feature_Extractor().to(config["device"])
    generator = train(model, content_path, style_path)
    
    save_path = check_dir("../output/")
    output = generator().clone().detach().cpu().squeeze()
    output = output * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    output += torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    torchvision.transforms.ToPILImage()(output).save(f"{save_path}cnn_{config['style']}.png")    
