from models.metanet_model import TransformNet, MetaNet

from torchvision import models, transforms
from torchsummary import summary

vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to('cuda')
vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).to('cuda')

transformnet = TransformNet(32).to('cpu')
metanet = MetaNet(transformnet.get_param_dict(), 'transformer').to('cpu')

# summary(vgg19, input_size=(3, 224, 224))

# summary(transformnet, input_size=(3, 256, 256), device='cpu')

summary(metanet, input_size=(1920,), batch_size=8, device='cpu')