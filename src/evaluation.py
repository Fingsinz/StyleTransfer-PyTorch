import sys
import time

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns

from utils.utils import calculate_ssim, calculate_psnr
from models.networks import ResNet18_Pretrained
from data.image_dataset import StyleImageDataset
import utils.config as Config
from utils.utils import check_dir

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class Scorer:
    def __init__(self, num_classes, target_class, train_path):
        self.num_classes = num_classes
        self.target_class = target_class
        self.model = ResNet18_Pretrained(num_classes=num_classes).to(Config.device)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.train_dataset = StyleImageDataset(train_path, transform=self.transform)
        self.class_names = self.train_dataset.classes
    
    def initialize_by_model(self, model_path):
        print(f"[INFO] Loading model from {model_path}")
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()
        print(f"[INFO] Initialization completed")
        
    def initialize_by_data(self):
        print(f"[INFO] Training model")
        train_loader = DataLoader(dataset=self.train_dataset,
                                  batch_size=32, shuffle=True)
        self.model.fine_tuning_model(epochs=10, train_loader=train_loader)
        self.model.eval()
        print(f"[INFO] Training completed")
    
    def get_all_prob(self, transformed_path):
        image = Image.open(transformed_path).convert('RGB')
        image = self.transform(image).unsqueeze(0)
        image = image.to(Config.device)
        
        with torch.inference_mode():
            output = self.model(image)
            probs = torch.softmax(output, dim=1)

        labels_prob = {}
        for i in range(self.num_classes):
            labels_prob[self.class_names[i]] = probs[0][i].item()

        print("[INFO] 风格概率: ")
        for label, prob in labels_prob.items():
            print(f"    {label}: {prob:.6f}")
        return labels_prob
    
    def get_style_score(self, transformed_path):
        return self.get_all_prob(transformed_path)[self.target_class]
    
    def get_content_score(self, content_path, transformed_path):
        content = np.array(Image.open(content_path).convert('RGB'))
        transformed = np.array(Image.open(transformed_path).convert('RGB'))
        ssim = calculate_ssim(content, transformed)
        return ssim

    def get_score(self, content_path, style_path, transformed_path):
        style_score = self.get_style_score(transformed_path)
        content_score = self.get_content_score(content_path, transformed_path)
        return style_score, content_score
    
    def generate_html_report(self, content_path, style_path, transformed_path, 
                            style_score, content_score):
        now_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        output_path=f"./report_{now_time}.html"
        style_display = self.target_class.replace('-', ' ').title()
        html_template = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Style Transfer Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 2rem; }}
                    .container {{ max-width: 1000px; margin: 0 auto; }}
                    .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }}
                    .card {{ border: 1px solid #ddd; padding: 1rem; border-radius: 8px; }}
                    img {{ max-width: 100%; height: auto; border-radius: 4px; }}
                    .metrics {{ background: #f8f9fa; padding: 1.5rem; margin-top: 2rem; }}
                    .score {{ color: #2c3e50; font-size: 1.2rem; margin: 0.5rem 0; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Style Transfer Evaluation Report</h1>
                    
                    <div class="grid">
                        <div class="card">
                            <h3>原始内容图</h3>
                            <img src="{content_path}" alt="Content Image">
                        </div>
                        <div class="card">
                            <h3>目标风格图</h3>
                            <img src="{style_path}" alt="Style Image">
                        </div>
                        <div class="card">
                            <h3>迁移结果图</h3>
                            <img src="{transformed_path}" alt="Transformed Image">
                        </div>
                    </div>
                    
                    <div class="style-header">
                        <h2>目标艺术风格：{style_display}</h2>
                    </div>
                    
                    <div class="metrics">
                        <h2>评估指标</h2>
                        <div class="score">风格匹配度: {style_score:.4f}</div>
                        <div class="score">内容保持度: {content_score:.4f}</div>
                        <div class="score" style="color: #e74c3c; font-weight: bold;">
                            综合评分: {(0.5*style_score + 0.5*content_score):.4f}
                        </div>
                    </div>
                </div>
            </body>
            </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_template)
        print(f"[INFO] 已生成HTML报告：{output_path}")

if __name__ == '__main__':
    content_path = sys.argv[1]
    style_path = sys.argv[2]
    transforms_path = sys.argv[3]
    style = sys.argv[4] # [bartolome-esteban-murillo, chinese-art, claude-monet, vincent-van-gogh]
    
    scorer = Scorer(4, target_class=style, train_path="../datasets/style_classification")
    scorer.initialize_by_model("../output/ResNet18_Pretrained.pth")
    # scorer.initialize_by_data()
    style_score, content_score = scorer.get_score(content_path, style_path, transforms_path)

    print(f"原始图像 {content_path} | 风格图像 {style_path} | 迁移后图像 {transforms_path}")
    print(f"迁移后图像风格为{style}")
    print("风格评分: {:.4f}, 内容评分: {:.4f}, 平均评分: {:.4f}"
          .format(style_score, content_score, (0.5 * style_score + 0.5 * content_score)))
    
    scorer.generate_html_report(content_path, style_path, transforms_path, style_score, content_score)
    