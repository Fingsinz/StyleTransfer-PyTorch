import os
import sys
import time
import yaml

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.utils.data as Data
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from tqdm import tqdm

from utils.utils import calculate_ssim, calculate_psnr
from models.networks import ResNet18_Pretrained
from data.image_dataset import StyleImageDataset
import utils.config as Config
from utils.utils import check_dir, save_model

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def train_style_classification(model, train_loader, test_loader, epochs=10):
    """微调模型"""
    model.to(Config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"{epoch + 1} / {epochs}"):
            images = images.to(Config.device)
            labels = labels.to(Config.device)
            output = model(images)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        with torch.inference_mode():
            correct = 0
            total = 0
            model.eval()
            for images, labels in test_loader:
                images = images.to(Config.device)
                labels = labels.to(Config.device)
                output = model(images)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = correct / total
            print(f"Epoch [{epoch + 1}/{epochs}], Accuracy: {accuracy:.4f}")
    save_model(model, '../output/', 'ResNet18_Pretrained.pth')

class Scorer:
    def __init__(self, num_classes, target_class, class_names, test_model):
        self.num_classes = num_classes
        self.target_class = target_class
        self.model = ResNet18_Pretrained(num_classes=num_classes).to(Config.device)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.class_names = class_names
        self.test_model = test_model 
    
    def initialize_by_model(self, model_path):
        print(f"[INFO] Loading model from {model_path}")
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()
        print(f"[INFO] Initialization completed")
        
    def initialize_by_data(self, data_path):
        print(f"[INFO] Training model")
        dataset = StyleImageDataset(data_path, transform=self.transform)
        train_dataset, test_dataset = \
            Data.random_split(dataset,
                              [int(len(dataset) * 0.8),
                               len(dataset) - int(len(dataset) * 0.8)])
        train_loader = Data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = Data.DataLoader(test_dataset, batch_size=16, shuffle=True)
        train_style_classification(self.model, train_loader, test_loader)
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
        test_model = self.test_model
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
                        <h2>测试迁移模型：{test_model}</h2>
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
    transformed_path = sys.argv[3]
    use_model = sys.argv[4]
    
    if not os.path.exists("./config_evaluation.yaml"):
        print("[ERROR] evaluation.yaml 不存在")
        exit()

    with open("./config_evaluation.yaml", 'r', encoding='utf-8') as f:
        eval_config = yaml.safe_load(f)
    
    class_names = eval_config["model"].get("classes")
    style = eval_config["style"]
    
    if content_path is None or style_path is None or transformed_path is None or \
        not os.path.exists(content_path) or not os.path.exists(style_path) or \
            not os.path.exists(transformed_path) or class_names is None or style is None:
        print("[ERROR] 参数错误")
        exit()
    
    scorer = Scorer(4, target_class=style, class_names=class_names, test_model=use_model)
    
    is_train = eval_config["model"].get("model_train")
    if is_train == True:
        data_path = eval_config["model"].get("dataset")
        scorer.initialize_by_data(data_path=data_path)
    else:
        model_path = eval_config["model"].get("model_path")
        scorer.initialize_by_model(model_path=model_path)
        
    style_score, content_score = scorer.get_score(content_path, style_path, transformed_path)

    print(f"[INFO] 原始图像 {content_path} | 风格图像 {style_path} | 迁移后图像 {transformed_path}")
    print(f"[INFO] 迁移后图像风格为{style}")
    print("[INFO] 风格评分: {:.4f}, 内容评分: {:.4f}, 平均评分: {:.4f}"
          .format(style_score, content_score, (0.5 * style_score + 0.5 * content_score)))
    
    scorer.generate_html_report(content_path, style_path, transformed_path, style_score, content_score)
    