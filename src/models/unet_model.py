from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()
        
        self.encoder1 = self.down_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self.down_block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = self.down_block(128, 256)
        
        self.decoder2 = self.up_block(256, 128)
        self.decoder1 = self.up_block(256, 64)
        
        self.final = nn.Sequential(
            nn.Conv2d(128, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)
        
        b = self.bottleneck(p2)
        
        d2 = self.decoder2(b)
        d2 = self._pad_and_cat(d2, e2)
        d1 = self.decoder1(d2)
        d1 = self._pad_and_cat(d1, e1)
        
        return self.final(d1)
        
    def _pad_and_cat(self, up, skip):
        diff_h = skip.size()[2] - up.size()[2]
        diff_w = skip.size()[3] - up.size()[3]
        
        # 对称填充
        up = nn.functional.pad(up, [diff_w // 2, diff_w - diff_w // 2,
                                    diff_h // 2, diff_h - diff_h // 2])
        return torch.cat([up, skip], dim=1) 
        
    def down_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
