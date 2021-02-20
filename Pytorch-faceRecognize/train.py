import os
import os.path as osp


import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import MobileNetV2_FaceNet as MobileFaceNet
from metric import ArcFace
from loss import FocalLoss
from dataset import load_data
from config import Config as conf

dataloader,class_num = load_data(conf, training=True)
embedding_size = conf.embedding_size
device = conf.device
print("The total of class numbers is %d" %(class_num))
print("The training device is set as %s" %(conf.device))


# Network setup
if conf.backbone == 'myfmobile':
    net = MobileFaceNet(embedding_size).to(device)
    print("Network backbone is myfmobile")
else:
    net = MobileFaceNet(embedding_size).to(device)

if conf.metric == 'arcface':
    metric = ArcFace(embedding_size, class_num).to(device)
    print("Metric fucntion is Arcface")
else:
    metric = ArcFace(embedding_size, class_num).to(device)

net = nn.DataParallel(net)
metric = nn.DataParallel(metric)

# Training setup
if conf.loss == 'focal_loss':
    criterion = FocalLoss(gamma = 2)
    print("Loss function is FocalLoss")
else:
    criterion = nn.CrossEntropyLoss()

if conf.optimizer == 'sgd':
    optimizer = optim.SGD([{'params': net.parameters()}, {'params': metric.parameters()}], 
                            lr=conf.lr, weight_decay=conf.weight_decay)
    print("Optimaizer is SGD")
else:
    optimizer = optim.Adam([{'params': net.parameters()}, {'params': metric.parameters()}],
                            lr=conf.lr, weight_decay=conf.weight_decay)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=conf.lr_step, gamma=0.1)
os.makedirs(conf.checkpoints, exist_ok = True)


net.train()
for e in range(conf.epoch):
    for data, labels in tqdm(dataloader, desc=f"Epoch {e}/{conf.epoch}", ascii=True, total=len(dataloader)):
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        embeddings = net(data)              # embedding features
        thetas = metric(embeddings, labels) # arcface
        loss = criterion(thetas, labels)    # loss function ce
        loss.backward()
        optimizer.step()
    print(f"Epoch {e}/{conf.epoch}, Loss: {loss}")
    backbone_path = osp.join(checkpoints, f"{e}.pth")
    torch.save(net.state_dict(), backbone_path)
    scheduler.step()