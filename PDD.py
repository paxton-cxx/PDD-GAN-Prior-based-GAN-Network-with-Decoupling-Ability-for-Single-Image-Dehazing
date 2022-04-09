import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as tfs
import torch.nn as nn
from collections import OrderedDict
import math
from cbam import CBAM
      
class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(TransitionBlock, self).__init__()
        self.relu = nn.LeakyReLU(0.2,inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                                        padding=0, bias=False)

    def forward(self, x):
        out = self.conv1(self.relu(x))
        return F.interpolate(out, scale_factor=2)


class conv3x3(nn.Module):
    """3x3 convolution with padding"""
    def __init__(self, inplanes, planes,kernel_size=3,stride=1,padding=1):
        super().__init__()
        self.conv=nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, bias=False)
    def forward(self,x):
        return self.conv(x)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.LeakyReLU(0.2,inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class up_sample(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(up_sample, self).__init__()
        self.relu = nn.LeakyReLU(0.2,inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                                        padding=0, bias=False)

    def forward(self, x):
        out = self.conv1(self.relu(x))
        return F.interpolate(out, scale_factor=2)




class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bottleneck_width, drop_rate):
        super(_DenseLayer, self).__init__()


        growth_rate = int(growth_rate / 2)
        inter_channel = int(growth_rate  * bottleneck_width / 4) * 4 

        if inter_channel > num_input_features / 2:
            inter_channel = int(num_input_features / 8) * 4
            # print('adjust inter_channel to ',inter_channel)

        self.branch1a = BasicConv2d(num_input_features, inter_channel, kernel_size=1)
        self.branch1b = BasicConv2d(inter_channel, growth_rate, kernel_size=3, padding=1)
        
        self.branch2a = BasicConv2d(num_input_features, inter_channel, kernel_size=1)
        self.branch2b = BasicConv2d(inter_channel, growth_rate, kernel_size=3, padding=1)
        self.branch2c = BasicConv2d(growth_rate, growth_rate, kernel_size=3, padding=1)
        self.attention=CBAM(growth_rate,growth_rate)
    def forward(self, x):
        branch1 = self.branch1a(x)
        branch1 = self.branch1b(branch1)

        branch2 = self.branch2a(x)
        branch2 = self.branch2b(branch2)
        branch2 = self.branch2c(branch2)
        branch2=self.attention(branch2)
        return torch.cat([x, branch1, branch2], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)



class _StemBlock(nn.Module):
    def __init__(self, num_input_channels, num_init_features,stride=1):
        super(_StemBlock, self).__init__()

        num_stem_features = int(num_init_features/2)

        self.stem1 = BasicConv2d(num_input_channels, num_init_features, kernel_size=3, stride=stride, padding=1)  
        self.stem2a = BasicConv2d(num_init_features, num_stem_features, kernel_size=1, stride=1, padding=0)
        self.stem2b = BasicConv2d(num_stem_features, num_init_features, kernel_size=3, stride=2, padding=1)
        self.stem3 = BasicConv2d(2*num_init_features, num_init_features, kernel_size=1, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.stem1(x)

        branch2 = self.stem2a(out)
        branch2 = self.stem2b(branch2)
        branch1 = self.pool(out)

        out = torch.cat([branch1, branch2], 1)
        out = self.stem3(out)

        return out

class BasicResBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(BasicResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.relu = nn.LeakyReLU(0.2,inplace=True)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        out += residual
        out = self.relu(out)
        out=torch.clip(out,0,1)
        return out

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, activation=True, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels) 
        self.drop=nn.Dropout2d(0.05)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x=self.drop(x)
        if self.activation:
            return F.leaky_relu(x,0.2, inplace=True)
        else:
            return x

class Generator(nn.Module):
    r"""PeleeNet model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf> and
     "Pelee: A Real-Time Object Detection System on Mobile Devices" <https://arxiv.org/pdf/1608.06993.pdf>` 
    Args:
        growth_rate (int or list of 4 ints) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bottleneck_width (int or list of 4 ints) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, growth_rate=16, block_config=[4,6,8],      #[4,2,2]  [2,2,2]
                 num_init_features=16, bottleneck_width=[5,7,9], drop_rate=0.05):

        super().__init__()


        self.features = nn.Sequential(OrderedDict([
            ('stemblock', _StemBlock(4, num_init_features)), 
        ]))

        if type(growth_rate) is list:
            growth_rates = growth_rate
           
        else:
            growth_rates = [growth_rate] * 4

        if type(bottleneck_width) is list:
            bottleneck_widths = bottleneck_width
        else:
            bottleneck_widths = [bottleneck_width] * 4
        self.block_n=[]
        self.trans_n=[]
        self.pool=nn.MaxPool2d(2,2)
        self.drop_rate = drop_rate
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bottleneck_widths[i], growth_rate=growth_rates[i], drop_rate=drop_rate)
            # self.densblock1
            if i==0:self.bl0=block
            elif i==1:self.bl1=block
            elif i==2:self.bl2=block

            self.block_n.append(block)
            num_features = num_features + num_layers * growth_rates[i]
            trans=BasicConv2d(num_features, num_features, kernel_size=1, stride=1, padding=0)
            self.trans_n.append(trans)
            if i==0:self.tr0=trans
            elif i==1:self.tr1=trans
            elif i==2:self.tr2=trans
            num_features = num_features
    
        # self.attention1=GAM_Attention(448,448)
        # self.deform1=DeformConv2d(448,448)

        self._initialize_weights()

        self.up1=up_sample(304,224)
        self.bl3=_DenseBlock(num_layers=5, num_input_features=352,
                                bn_size=2, growth_rate=32, drop_rate=drop_rate)

      

        self.up2=up_sample(512,368)
        self.bl4=_DenseBlock(num_layers=5, num_input_features=528,
                                bn_size=3, growth_rate=32, drop_rate=drop_rate)

     

       
        self.up3=up_sample(688,368)
        self.bl5=_DenseBlock(num_layers=5, num_input_features=400,
                                bn_size=3, growth_rate=32, drop_rate=drop_rate)


       
        self.up4=up_sample(560,368)
        self.bl6=_DenseBlock(num_layers=5, num_input_features=372,
                                bn_size=3, growth_rate=32, drop_rate=drop_rate)

        
        self.conv1=nn.Conv2d(532, 368, kernel_size=1, stride=1,padding=0)
        self.batch_norm=nn.BatchNorm2d(24)
        self.conv2=nn.Conv2d(368, 128, kernel_size=3, stride=1,padding=1)
        self.relu = nn.LeakyReLU(0.2)
        self.conv3=nn.Conv2d(128, 4, kernel_size=3, stride=1,padding=1)
        self.relu = nn.LeakyReLU(0.2)
        # self.res5=BasicResBlock(48,32)
        

        self.x0_c=BasicConv2d(16,32,kernel_size=3, stride=1, padding=1)
        self.x1_c=BasicConv2d(80,160,kernel_size=3, stride=1, padding=1)
        self.x2_c=BasicConv2d(176,128,kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        x0=self.features(x)
        
        x0_c=self.x0_c(x0)
        
        # x1=self.block_n[0].to('cuda')(x0)
        # x1=self.trans_n[0].to('cuda')(x1)
        x1=self.bl0(x0)
        x1=self.tr0(x1)
        x1=self.pool(x1)
        # print(x1.shape)
        
        x1_c=self.x1_c(x1)

        # x2=self.block_n[1].to('cuda')(x1)
        # x2=self.trans_n[1].to('cuda')(x2)
        x2=self.bl1(x1)
        x2=self.tr1(x2)
        x2=self.pool(x2)
        
        x2_c=self.x2_c(x2)

        # x3=self.block_n[2].to('cuda')(x2)
        # x3=self.trans_n[2].to('cuda')(x3)
        x3=self.bl2(x2)
        x3=self.tr2(x3)
        x3=self.pool(x3)
        # x3=self.deform1(x3)
        print(x3.shape)
       
        x4=self.up1(x3)
        x4=torch.cat([x4,x2_c],1)
        x4=self.bl3(x4)
        # print(x4.shape)
       
        x5=self.up2(x4)
        x5=torch.cat([x5,x1_c],1)
        x5=self.bl4(x5)
        
       
        x6=self.up3(x5)
        x6=torch.cat([x6,x0_c],1)
        x6=self.bl5(x6)
        
       
        
        x7=self.up4(x6)
        x7=torch.cat([x7,x],1)
        x7=self.bl6(x7)
       
        # print('*************')
        # print(x7.shape)
        x8=self.relu(self.conv1(x7))
        x8=self.conv2(x8)
        x8=self.conv3(x8)
        out=x8
        out=torch.clip(out,0,1)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



class Discriminator(nn.Module):
    def __init__(self, nc=4, nf=36):
        super(Discriminator, self).__init__()
        
        self.features = nn.Sequential(OrderedDict([
            ('stemblock', _StemBlock(4, 32,stride=2)), 
        ]))                                         #4*224*224->32*56*56
        self.pool=nn.MaxPool2d(2,2)                 #32*56*56->32*28*28
        self.bl1=_DenseBlock(num_layers=5, num_input_features=32,
                                bn_size=3, growth_rate=24,drop_rate=0.05)     #32*28*28->104*28*28

        self.last = nn.Sequential(
            nn.Conv2d(152, 71, kernel_size=3, stride=1,padding=1),  # 104*28*28->52*28*28
            nn.LeakyReLU(0.2),
            nn.Conv2d(71, 32, kernel_size=3, stride=1,padding=1),  # 52*28*28->1*28*28
            nn.Conv2d(32, 1, kernel_size=3, stride=1,padding=1),  # 52*28*28->1*28*28
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.features(x)
        x2=self.bl1(x1)
        x2=self.pool(x2)
        x3=self.last(x2)
   
        out=x3
        return out

    def requires_grad(self, req):
        for param in self.parameters():
            param.requires_grad = req




if __name__ == '__main__':
    # print(output.shape)
    x=torch.rand(5,4,224,224)
    print(Generator()(x).shape)
