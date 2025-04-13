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
        """x: [B, G * 128]"""
        batch = x.size(0)
        x_grouped = x.view(batch, self.num_groups, 128) # [B, G, 128]
        gap = x_grouped.mean(dim=1)                     # [B, 128]
        weights = self.fc(gap).unsqueeze(1)             # [B, 1, 128]
        x_weighted = x_grouped * weights                # [B, G, 128]
        return x_weighted.view(batch, -1)               # [B, G * 128]

class EnhancedChannelAttention(torch.nn.Module):
    def __init__(self, num_groups, ratio=16):
        super(EnhancedChannelAttention, self).__init__()
        self.num_groups = num_groups
        
        self.group_fc = torch.nn.Sequential(
            torch.nn.Linear(128, 128 // ratio),
            torch.nn.LayerNorm(128 // ratio),
            torch.nn.GELU(),
            torch.nn.Linear(128 // ratio, 128),
            torch.nn.Sigmoid()
        )
        
        self.global_fc = torch.nn.Sequential(
            torch.nn.Linear(128 * num_groups, 128),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        """x: [B, G * 128]"""
        batch = x.size(0)
        
        x_grouped = x.view(batch, self.num_groups, 128)             # [B, G, 128]
        
        gap_group = x_grouped.mean(dim=1)                           # [B, 128]
        group_w = self.group_fc(gap_group).unsqueeze(1)             # [B, 1, 128]
        
        gap_global = x_grouped.view(batch, -1)                      # [B, G * 128]
        global_w = self.global_fc(gap_global).unsqueeze(1)          # [B, 1, 128]
                
        combined_w = group_w * 0.7 + global_w * 0.3                 # [B, 1, 128]
        return (x_grouped * combined_w).view(batch, -1)             # [B, G * 128]
     
class SpatialAttention(torch.nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(2, 1, kernel_size=7, padding=3),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        """x: [B, C, H, W]"""
        avg_out = torch.mean(x, dim=1, keepdim=True)    # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        combined = torch.cat([avg_out, max_out], dim=1) # [B, 2, H, W]
        return self.conv(combined)                      # [B, 1, H, W]
