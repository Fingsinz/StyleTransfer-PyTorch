import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Conv2D_NoTrain(nn.Module):
    """自定义卷积层，不可训练"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(Conv2D_NoTrain, self).__init__()
        self.weight = nn.init.kaiming_normal_(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = torch.zeros(out_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        return s.format(**self.__dict__)

class VGG_Pretrained(nn.Module):
    """用于提取任意特征的预训练 VGG 模型"""
    def __init__(self, layer_ids):
        super(VGG_Pretrained, self).__init__()
        self.features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.layer_ids = set(layer_ids)
                 
        for p in self.parameters():
            p.requires_grad = False
            
    def forward(self, x):
        outputs = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self.layer_ids:
                outputs.append(x)
        return outputs

class VGG_3_8_15_22(nn.Module):
    """3, 8, 15, 22 层的预训练 VGG 模型"""
    def __init__(self):
        super(VGG_3_8_15_22, self).__init__()
        self.features = VGG_Pretrained([3, 8, 15, 22])
       
    def forward(self, x):
        return self.features(x)

class ResidualBlock_2Conv_NoTrain(nn.Module):
    """不可训练的残差块，2 层卷积 + x"""
    def __init__(self, channels):
        super(ResidualBlock_2Conv_NoTrain, self).__init__()
        self.conv = nn.Sequential(
            *ConvLayer(channels, channels, kernel_size=3, stride=1),
            *ConvLayer(channels, channels, kernel_size=3, stride=1, relu=False)
        )
        
    def forward(self, x):
        return self.conv(x) + x
    
def ConvLayer(in_channels, out_channels, kernel_size=3, stride=1, upsample=None,
              instance_norm=True, relu=True, trainable=False):
    """
    构造一个卷积层。

    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int, optional): 卷积核大小，默认为 3
        stride (int, optional): 卷积的步幅，默认为 1
        upsample (float or None, optional): 上采样的比例因子，None 表示不上采样。
        instance_norm (bool, optional): 是否在卷积后应用实例归一化，默认为 True。
        relu (bool, optional): 是否在规范化后应用 ReLU 激活，默认为 True。
        trainable (bool, optional): 参数是否可训练，默认为 False。

    Returns:
        list: 卷积层的列表。
    """
    layers = []
    if upsample:
        layers.append(nn.Upsample(mode='nearest', scale_factor=upsample))

    layers.append(nn.ReflectionPad2d(kernel_size // 2))  # 填充以保持空间维度

    if trainable:
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride))
    else:
        layers.append(Conv2D_NoTrain(in_channels, out_channels, kernel_size, stride))

    if instance_norm:
        layers.append(nn.InstanceNorm2d(out_channels))

    if relu:
        layers.append(nn.ReLU(inplace=True))

    return layers

