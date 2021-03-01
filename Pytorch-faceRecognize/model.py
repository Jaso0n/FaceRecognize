import torch
import torch.nn as nn
import torch.nn.functional as F
from NormLinear import NormLinear
from Scale import Scale
import math

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
           (4, 128, 1, 2, 1),
           (2, 128, 3, 1, 1)]

    # start convolution layer
    def conv2d_bn_relu(self,in_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=2,padding=1,bias=False),
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
    
    def __init__(self, embedding_size):
        super(MobileNetV2_FaceNet, self).__init__()
        self.conv1 = self.conv2d_bn_relu(3,64)
        self.bottleneck_layers = self.make_bottleneck_layers(64)
        self.expand_conv = nn.Conv2d(128,512,kernel_size=1,stride=1,padding=0,bias=False)
        self.depthconv = nn.Conv2d(512,512,(6,6),stride=1,padding=0,bias=False,groups=512)
        self.linear_conv = nn.Linear(512,embedding_size,bias=False)     #input = 512, output = embedding_size
        self.bn1d = nn.BatchNorm1d(embedding_size)

    def forward(self,x):
        out = self.conv1(x)
        out = self.bottleneck_layers(out)
        out = self.expand_conv(out)
        out = self.depthconv(out)
        out = out.view(out.shape[0],-1)# flatten
        out = self.linear_conv(out)
        out = self.bn1d(out)
        return out

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            with torch.no_grad():
                m.bias.zero_()

class MyBlock(nn.Module):
    def __init__(self, planes, stride=1, downsample=None, nonlinear='relu'):
        super(MyBlock, self).__init__()
        self.downsample = downsample
        
        self.conv1 = conv3x3(planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        if nonlinear == 'relu':
            self.relu1 = nn.ReLU(inplace=False)
        else:
            self.relu1 = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        if nonlinear == 'relu':
            self.relu2 = nn.ReLU(inplace=False)
        else:
            self.relu2 = nn.PReLU(planes)
        self.stride = stride

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu2(out)

        return out

class MyNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, gray=False, nonlinear='relu', pruning=1):
        super(MyNet, self).__init__()
        input_dim = 1 if gray else 3
        self.nonlinear = nonlinear
        self.layer1 = self._make_layer(block, input_dim, 64//pruning, layers[0])
        self.layer2 = self._make_layer(block, 64//pruning, 128//pruning, layers[1])
        self.layer3 = self._make_layer(block, 128//pruning, 256//pruning, layers[2])
        self.layer4 = self._make_layer(block, 256//pruning, 512//pruning, layers[3])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        if self.nonlinear == 'relu':
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes,
                          kernel_size=3, stride=2, padding=1, bias=True),
                nn.ReLU(inplace=False),
            )
        else:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes,
                          kernel_size=3, stride=2, padding=1, bias=True),
                nn.PReLU(planes),
            )

        layers = []
        layers.append(block(planes, stride, downsample, nonlinear=self.nonlinear))
        for i in range(1, blocks):
            layers.append(block(planes, nonlinear=self.nonlinear))

        return nn.Sequential(*layers)

class Resnet(nn.Module):
    def __init__(self, layers=[1,2,4,1], pretrained=False, wn=True, fn=True, ring=None, sphere=False, gray=False, pruning=1, fea_dim=512, **kwargs):
        super(Resnet, self).__init__()
        self.fea_dim = fea_dim
        self.model = MyNet(MyBlock, layers, gray=gray, pruning=pruning, **kwargs)
        self.model.dropout = nn.Dropout2d(p=0.4)
        self.model.fc1 = nn.Linear(512//pruning*7*6, self.fea_dim, bias=False)

        #self.model.fc1_bn = nn.BatchNorm1d(self.fea_dim, affine=True)
        self.wn = wn
        self.fn = fn
        self.sphere = sphere
        #self.model.classifier = NormLinear(self.fea_dim, num_classes, bias=False, wn=self.wn, fn=self.fn)

        # self.model.apply(weight_init)

    def forward(self, x):
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc1(x)
        return x

def test():
    myresnet20 = Resnet()
    #print(myresnet20)
    #net = MobileNetV2_FaceNet(128)
    #print(net)
    x = torch.randn(10,3,112,96)
    #x = torch.randn(10,3,96,96)
    #y = net(x)
    y = myresnet20(x)
    print(y.size(0),y.size(1))

if __name__ == "__main__":
    test()