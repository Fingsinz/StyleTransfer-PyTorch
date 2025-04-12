"""
注意力模块
"""

import torch

class ChannelAttention(torch.nn.Module):
    def __init__(self, num_groups, ratio=16):
        super(ChannelAttention, self).__init__()
        self.num_groups = num_groups
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(128, 128 // ratio),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128 // ratio, 128),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: [batch_size, num_groups * 128]
        batch = x.size(0)
        x_grouped = x.view(batch, self.num_groups, 128) # [batch_size, num_groups, 128]
        gap = x_grouped.mean(dim=1)                     # [batch_size, 128]
        weights = self.fc(gap).unsqueeze(1)             # [batch_size, 1, 128]
        x_weighted = x_grouped * weights                # [batch_size, num_groups, 128]
        return x_weighted.view(batch, -1)               # [batch_size, num_groups * 128]
        
class SpatialAttention(torch.nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        return self.conv(combined)
        