import pathlib

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

data_transform = transforms.Compose([
    transforms.RandomResizedCrop(256, scale=(256/480, 1), ratio=(1, 1)), 
    transforms.ToTensor(), 
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

class ImageDataset(Dataset):
    def __init__(self, targ_dir: str, transform=None):
        self.paths = list(pathlib.Path(targ_dir).glob("*.jpg"))
        self.transform= transform

    def load_image(self, index: int) -> Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path).convert('RGB')

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        img = self.load_image(index)
        if self.transform:
            return self.transform(img)
        else:
            return img
