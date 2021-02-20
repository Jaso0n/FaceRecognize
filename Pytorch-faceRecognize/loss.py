import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        """
        when gamma = 0, the focalLoss drops to softmax,
        focal loss usually used in object detection
        """
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss() # 交叉熵损失函数
    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1-p) ** self.gamma * logp
        return loss.mean()
    