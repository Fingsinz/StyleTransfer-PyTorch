import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet18

from utils.utils import calculate_ssim, calculate_psnr

class Scorer:
    def __init__(self, model_path, target_class_idx):
        self.model = resnet18(weights='IMAGENET1K_V1')  
        num_classes = 2
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.target_class_idx = target_class_idx
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def get_style_score(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0)
        with torch.inference_mode():
            output = self.model(image)
            prob = torch.softmax(output, dim=1)[0][self.target_class_idx].item()
        return prob
    
    def get_content_score(self, content_path, transform_image):
        content = np.array(Image.open('./content1.JPEG').convert('RGB'))
        transformed = np.array(Image.open('./content1style1.png').convert('RGB'))
        ssim = calculate_ssim(content, transformed)
        return ssim

    def get_score(self, content_path, style_path, transform_image):
        style_score = self.get_style_score(style_path)
        content_score = self.get_content_score(content_path, transform_image)
        return style_score, content_score        

if __name__ == '__main__':
    content = np.array(Image.open('./content1.JPEG').convert('RGB'))
    transformed = np.array(Image.open('./content1style1.png').convert('RGB'))
    ssim = calculate_ssim(content, transformed)
    psnr = calculate_psnr(content, transformed)
    print(ssim, psnr)
    