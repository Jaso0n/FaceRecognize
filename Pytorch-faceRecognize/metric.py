import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class DenseClassifier(nn.Module):
    def __init__(self, embedding_size, class_num):
        super().__init__()
        self.linear = nn.Linear(embedding_size,class_num,bias=False)
    
    def forward(self,input):
        output = self.linear(input)
        return output

class ArcFace(nn.Module):
    def __init__(self, embedding_size, class_num, s = 32.0, m = 0.5): # suggestion scale = 32, 64
        """ ArcFace formular in programming
            origin formular : cos(m + theta) = cos(m)cos(theta) - sin(m)sin(theta)
            the angular between two vectors is met:
                0 <= m + theta <= pi 
            Note that:
                -m <= theta <= pi - m
            we have 
                cos(m) <= cos(theta) < cos(pi - m)
            So we can use cos(pi-m) as threshold to check whether (m+theta) go out of [0,pi]

            Args:
            embedding_size: usually 128, 256, 512 ...
            class_num: num of people when training
            s: scale, see normface https://arxiv.org/abs/1704.06369
            m: margin, see SphereFace, CosFace, and ArcFace paper
        """
        super().__init__()
        self.input_features = embedding_size
        self.out_features = class_num
        self.scale = s
        self.margin = m
        self.weight = nn.Parameter(torch.FloatTensor(class_num, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m) # threshold
        self.mm = math.sin(math.pi - m) * m # 

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))#cos(theta), target logits
        sine = ((1.0 - cosine.pow(2)).clamp(0,1)).sqrt()               #sin(theta)
        phi = cosine * self.cos_m - sine * self.sin_m                  #cos(theta+m), target logits with additional margin(penalize)
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)     #drop to CosFace
            
        output = cosine * 1.0 # target logits, tricky way to make backward work
        batch_size = len(output)
        output[range(batch_size), label] = phi[range(batch_size), label]
        return output * self.scale

class CosFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        """
        Args:
            embedding_size: usually 128, 256, 512 ...
            class_num: num of people when training
            s: scale, see normface https://arxiv.org/abs/1704.06369
            m: margin, see SphereFace, CosFace, and ArcFace paper
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        output = cosine * 1.0  # make backward works
        batch_size = len(output)
        output[range(batch_size), label] = phi[range(batch_size), label]
        return output * self.s

class NormLinear(nn.Module):
    '''
        See normface https://arxiv.org/abs/1704.06369,
        Please turn bias off in linear layer
    '''
    # weight norm=wn,feature norm = fn
    def __init__(self, in_features, out_features, bias=False, wn=True, fn=True, scale = 32): 
        super(NormLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wn = wn
        self.fn = fn
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.scale = scale
        nn.init.xavier_uniform_(self.weight) # reset weight 
    
    def forward(self, input):
        weight_norm = F.normalize(self.weight)
        input_norm = F.normalize(input)
        output = F.linear(input_norm, weight_norm)
        return self.scale * output

if __name__ == '__main__':
    model = NormLinear(4, 1, wn=False, fn=False)
    x = Variable(torch.Tensor([[1, 2, 3, 4]]), requires_grad=True)
    y = model(x)
    print('x: {} y: {}'.format(x, y))
    y.backward()
    print('x.grad: {}'.format(x.grad))
    print('weight.grad: {}'.format(model.weight.grad))
