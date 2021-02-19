import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    # expand-pointwise + depthwise + linear-pointwise
    def __init__(self, in_channel, out_channel, expansion, stride, pad):
        super(Bottleneck, self).__init__()
        channel = expansion * in_channel
        self.block = nn.Sequential(
            # expand-pointwise convolution
            nn.Conv2d(in_channel,channel,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            # depthwise convolution
            nn.Conv2d(channel,channel,kernel_size=3,stride=stride,padding=pad,groups=channel,bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            # linear-pointwise convolution
            nn.Conv2d(channel,out_channel,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(out_channel)
        )
        # residual: skip connection 
        # Assumption base on previous ResNet papers: If the in_channel is not equal to the out_channel,
        # there should be an extra pointwise convolution layer. In fact that in MobileNetV2, the extra
        # layer does not exsist.
        self.use_residual = (stride == 1) and (in_channel == out_channel)

    def forward(self,x):
        if self.use_residual:
            return self.block(x) + x
        else:
            return self.block(x)

class MobileNetV2_FaceNet(nn.Module):
    #(expansion, output_channels, num_blocks, stride, padding)
    cfg = [(2,  64, 1, 2, 1),
           (2,  64, 4, 1, 1),
           (4, 128, 1, 2, 1),
           (2, 128, 6, 1, 1),
           (4, 128, 1, 2, 0),
           (2, 128, 3, 1, 1)]

    # start convolution layer
    def conv2d_bn_relu(self,in_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=2,padding=1,bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
    # bottleneck layers
    def make_bottleneck_layers(self,in_channel):
        layers= []
        for expansion, output_channel, num_block, stride, padding in self.cfg:
            for i in range(0,num_block):
                layers.append(Bottleneck(in_channel,output_channel,expansion,stride,padding))
                in_channel = output_channel
        return nn.Sequential(*layers)
    
    def __init__(self, num_classes = 10575):
        super(MobileNetV2_FaceNet, self).__init__()

        self.conv1 = self.conv2d_bn_relu(3,64)
        self.bottleneck_layers = self.make_bottleneck_layers(64)
        self.expand_conv = nn.Conv2d(128,512,kernel_size=1,stride=1,padding=0,bias=False)
        self.depthconv = nn.Conv2d(512,512,kernel_size=6,stride=1,padding=0,bias=False,groups=512)
        self.linear_conv = nn.Conv2d(512,128,kernel_size=1,stride=1,padding=0,bias=False)
        self.innerproduct = nn.Linear(128,num_classes)

    def forward(self,x):
        out = self.conv1(x)
        out = self.bottleneck_layers(out)
        out = self.expand_conv(out)
        out = self.depthconv(out)
        out = self.linear_conv(out)
        out = out.view(1,128)
        out = self.innerproduct(out)
        return out

def test():
    net = MobileNetV2_FaceNet()
    x = torch.randn(1,3,112,112)
    y = net(x)
    print(y.size())

def main():
    test()

if __name__ == "__main__":
    main()