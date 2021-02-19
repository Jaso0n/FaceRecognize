import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFace(nn.Module):
    def __init__(self, embedding_size, class_num, s = 30.0, m = 0.5):
        """ ArcFace formular in programming
            origin formular : cos(m + theta) = cos(m)cos(theta) - sin(m)sin(theta)
            the angular between two vectors is met:
                0 <= m + theta <= pi 
            Note that:
            if (m+theta) > pi, then theta >= pi - m. 
            In [0,pi] we have 
                cos(theta) < cos(pi - m)
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
        self.mm = math.sin(math.pi - m) * m # CosFace

        def forward(self, input, label):
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
            sine = ((1.0 - cosine.pow(2)).clamp(0,1)).sqrt()
            theta = cosine * self.cos_m - sine * self.sin_m
            theta = torch.where(cosine > self.th, theta, cosine - self.mm) # drop to CosFace
            
            output = cosine * 1.0
            batch_size = len(output)
            output[range(batch_size), label] = theta[range(batch_size),label]
            return output * self.s

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