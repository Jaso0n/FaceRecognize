import os
import os.path as osp


import torch
import torch.nn as nn
import torch.optim as optim

from model import MobileNetV2_FaceNet as MobileFaceNet
from metric import ArcFace,DenseClassifier
from loss import FocalLoss
from dataset import load_data
from config import Config as conf
import test

print("Import OK")

def train(conf):
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
        metric = DenseClassifier(embedding_size, class_num).to(device)
        print("Metric fucntion is DenseClassifier")

    net = nn.DataParallel(net)
    metric = nn.DataParallel(metric)

    # Training setup
    if conf.loss == 'focal_loss':
        criterion = FocalLoss(gamma = 2)
        print("Loss function is FocalLoss")
    else:
        criterion = nn.CrossEntropyLoss()

    if conf.optimizer == 'sgd':
        optimizer = optim.SGD([{'params': net.parameters()}, {'params': metric.parameters()}], lr=conf.lr, weight_decay=conf.weight_decay)
        print("Optimaizer is SGD")
    else:
        optimizer = optim.Adam([{'params': net.parameters()}, {'params': metric.parameters()}],lr=conf.lr, weight_decay=conf.weight_decay)
        print("Optimaizer is Adam")

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=conf.lr_step, gamma=0.1)
    os.makedirs(conf.checkpoints, exist_ok = True)
    best_acc = 0
    best_th = 0
    net.train()
    for e in range(conf.epoch):
        for batch_idx,data in enumerate(dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            embeddings = net(inputs)              # embedding features
            thetas = metric(embeddings)           # arcface
            loss = criterion(thetas, labels)      # loss function ce
            loss.backward()
            optimizer.step()
            if (batch_idx % conf.step_show) == 0:
                print("Train Epoch:[%d/%d]:interation %d \t \t Loss: %.5f \t \t learning rate: %f" %(e,conf.epoch,batch_idx, loss,scheduler.get_lr()[0]))
            if (batch_idx % 500) == 0 and batch_idx != 0:#conf.test_step
                backbone_path = osp.join(conf.checkpoints, f"{batch_idx}.pth")
                torch.save(net.state_dict(), backbone_path)
                accuracy,threshold = test.test(conf,f"./checkpoints/{batch_idx}.pth")
                print(f"\nLFW Test:Epoch[{e:d}]: Iteration {batch_idx:d} \t \t accuracy: {accuracy:.3f} \t \t threshold: {threshold:.3f}\n")
        scheduler.step()


train(conf)