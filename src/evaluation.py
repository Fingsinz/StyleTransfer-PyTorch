import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from utils.utils import calculate_ssim, calculate_psnr
from models.networks import ResNet18_Pretrained
from data.image_dataset import StyleImageDataset
import utils.config as Config

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

        print(f"Style Probably: {labels_prob}")
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

if __name__ == '__main__':
    content_path = "./content1.JPEG"
    style_path = "./style1.jpg"
    transforms_path = "../output/content1style1.png"
    
    scorer = Scorer(4, target_class="chinese-art", train_path="../datasets/style_classification")
    scorer.initialize_by_model("../output/ResNet18_Pretrained.pth")
    # scorer.initialize_by_data()
    style_score, content_score = scorer.get_score(content_path, style_path, transforms_path)
    print(f"Style score: {style_score}, Content score: {content_score}, "
        f"Final score: {0.5 * style_score + 0.5 * content_score}")
    