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
        group_w = self.group_fc(gap_group)                          # [B, 128]
        
        gap_global = x                                              # [B, G * 128]
        global_w = self.global_fc(gap_global)                       # [B, 128]
                
        combined_w = (group_w * 0.7 + global_w * 0.3).unsqueeze(1)  # [B, 1, 128]
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

class SelfAttention(torch.nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.quary = torch.nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key = torch.nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value = torch.nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = torch.nn.Parameter(torch.zeros(1))
        self.softmax = torch.nn.Softmax(dim=-1)
        
    def forward(self, x):
        """x: [B, C, H, W]"""
        B, C, H, W = x.size()
        
        proj_query = self.quary(x).view(B, -1, H * W).permute(0, 2, 1)  # [B, H * W, C // 8]
        proj_key = self.key(x).view(B, -1, H * W)                       # [B, C // 8, H * W]
        
        energy = torch.bmm(proj_query, proj_key)                        # [B, H * W, H * W]
        attention = self.softmax(energy)                                # [B, H * W, H * W]
        
        proj_value = self.value(x).view(B, -1, H * W)                   # [B, C, H * W]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))         # [B, C, H * W]
        out = out.view(B, C, H, W)                                      # [B, C, H, W]
        
        return self.gamma * out + x
        
class TransformerBlock(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.attn = torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = torch.nn.LayerNorm(embed_dim)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim * 4),
            torch.nn.GELU(),
            torch.nn.Linear(embed_dim * 4, embed_dim),
            torch.nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """x: [L, B, C] (序列长度，批次大小，特征维度)"""
        attn_output , _ = self.attn(x, x, x)
        x = x + attn_output
        x = self.norm1(x)
        
        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.norm2(x)
        
        return x
    
class VisionTransformer(torch.nn.Module):
    def __init__(self, in_channels, embed_dim, num_layers, num_heads):
        super(VisionTransformer, self).__init__()
        self.patch_embed = torch.nn.Conv2d(in_channels, embed_dim, kernel_size=1, stride=1)
        self.transformer = torch.nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        """x: [B, C, H, W]"""
        x = self.patch_embed(x)             # [B, E, H, W]
        B, E, H, W = x.size()
        
        x = x.flatten(2).permute(2, 0, 1)   # [H * W, B, E]
        x = self.transformer(x)             # [H * W, B, E]
        x = x.permute(1, 2, 0)              # [B, E, H * W]
        x = x.reshape(B, E, H, W)           # [B, E, H, W]
        
        return x
        