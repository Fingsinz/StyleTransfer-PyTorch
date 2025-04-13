import numpy as np
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F

from models.networks import ConvLayer, ResidualBlock_2Conv_NoTrain, Conv2D_NoTrain
from utils.utils import mean_std
from models.attention import ChannelAttention, EnhancedChannelAttention, SpatialAttention

class TransformNet(nn.Module):
    """图像转换网络"""
    def __init__(self, base=8):
        super(TransformNet, self).__init__()
        self.base = base
        self.weights = []
        
        self.downsampling = nn.Sequential(
            *ConvLayer(3, base, kernel_size=9, trainable=True),
            *ConvLayer(base, base * 2, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 显式下采样
            *ConvLayer(base * 2, base * 4, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.residuals = nn.Sequential(*[ResidualBlock_2Conv_NoTrain(base*4) for _ in range(5)])
        
        self.upsampling = nn.Sequential(
            *ConvLayer(base * 4, base * 2, kernel_size=3, upsample=2),
            *ConvLayer(base * 2, base, kernel_size=3, upsample=2),
            *ConvLayer(base, 3, kernel_size=9, instance_norm=False, relu=False, trainable=True)
        )
        
        self.get_param_dict()

    def forward(self, x):
        orig_h, orig_w = x.shape[2], x.shape[3]
        y = self.downsampling(x)
        y = self.residuals(y)
        y = self.upsampling(y)
        y = F.interpolate(y, size=(orig_h, orig_w), mode='bilinear',align_corners=False)
        return y
    
    def get_param_dict(self):
        """找出该网络所有 MyConv2D 层，计算它们需要的权值数量"""
        param_dict = defaultdict(int)
        def dfs(module, name):
            for _name, layer in module.named_children():
                dfs(layer, '%s.%s' % (name, _name) if name != '' else _name)
            if isinstance(module, Conv2D_NoTrain):
                param_dict[name] += int(np.prod(module.weight.shape))
                param_dict[name] += int(np.prod(module.bias.shape))
        dfs(self, '')
        return param_dict
    
    def set_my_attr(self, name, value):
        """遍历字符串（如 residuals.0.conv.1）找到对应的权值"""
        target = self
        for x in name.split('.'):
            if x.isnumeric():
                target = target.__getitem__(int(x))
            else:
                target = getattr(target, x)
        n_weight = np.prod(target.weight.shape)
        target.weight = value[:n_weight].view(target.weight.shape)
        target.bias = value[n_weight:].view(target.bias.shape)

    def set_weights(self, weights, i=0):
        """输入权值字典，对应网络所有的 MyConv2D 层进行设置"""
        for name, param in weights.items():
            self.set_my_attr(name, weights[name][i])
            
class MetaNet(nn.Module):
    def __init__(self, param_dict, attention: str):
        super(MetaNet, self).__init__()
        self.param_num = len(param_dict)
        self.hidden = nn.Linear(1920, 128 * self.param_num)
        
        self.fc_dict = {}
        for i, (name, params) in enumerate(param_dict.items()):
            self.fc_dict[name] = i
            setattr(self, 'fc{}'.format(i + 1), nn.Linear(128, params))
        
        self.att_type = attention
        if self.att_type == 'channel':      # 通道注意力
            self.attention = ChannelAttention(num_groups=self.param_num)
        elif self.att_type == 'enhanced_channel': # 增强通道注意力
            self.attention = EnhancedChannelAttention(num_groups=self.param_num)

            
    def forward(self, mean_std_features):
        hidden = F.relu(self.hidden(mean_std_features))
        
        if self.att_type == 'channel': # 注意力处理
            hidden = self.attention(hidden)
        elif self.att_type == 'enhanced_channel':
            hidden = self.attention(hidden)
        
        filters = {}
        for name, i in self.fc_dict.items():
            fc = getattr(self, 'fc{}'.format(i + 1))
            filters[name] = fc(hidden[:, i * 128 : (i + 1) * 128])
            
        return filters
