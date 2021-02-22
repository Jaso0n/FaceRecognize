import os
import os.path as osp
import logging
import time
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

def add_weight_decay(net,weight_decay, skip_list=()):
    decay = []
    no_decay = []
    for name ,param in net.named_parameters():
        if not param.requires_grad: continue  # skip frozen weights
        if len(param.shape) == 1 or name in skip_list:
            #print(name)
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params' : no_decay, 'weight_decay' : 0.0},
            {'params' : decay, 'weight_decay' : weight_decay}]


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

class Train():
    def __init__(self, conf):
        os.makedirs(conf.checkpoints, exist_ok = True)
        self.nowtime = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
        self.logger = get_logger("./"+ self.nowtime +"-train.log")
        self.logger.info('Successfully create train log file at \'./{}-train.log\''.format(self.nowtime))
        # dataloader
        self.dataloader, self.class_num = load_data(conf, training=True)
        self.embedding_size = conf.embedding_size
        self.device = conf.device
        self.logger.info('DataLoader is created, the total of class numbers is {}, embedding size is {}'.format(self.class_num,self.embedding_size))
        # generate network
        if conf.backbone == 'myfmobile':
            self.net = MobileFaceNet(self.embedding_size).to(self.device)
            print("Network backbone is myfmobile")
        else:
            self.net = MobileFaceNet(self.embedding_size).to(self.device)

        if conf.metric == 'arcface':
            self.metric = ArcFace(self.embedding_size, self.class_num).to(self.device)
            print("Metric fucntion is Arcface")
        else:
            self.metric = DenseClassifier(self.embedding_size, self.class_num).to(self.device)
            print("Metric fucntion is DenseClassifier")

        self.net = nn.DataParallel(self.net)
        self.metric = nn.DataParallel(self.metric)
        # remove weight_decay in batchnorm and convolution bias, refer to https://arxiv.org/abs/1706.05350
        net_params = add_weight_decay(self.net,conf.weight_decay)
        metric_params = add_weight_decay(self.metric,conf.class_wd)
        parameters = net_params + metric_params
        if conf.loss == 'focal_loss':
            self.criterion = FocalLoss(gamma = 2)
            print("Loss function is FocalLoss")
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        if conf.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, lr = conf.lr, momentum=conf.momentum, weight_decay = conf.weight_decay)
            print("Optimaizer is SGD")
        else:
            self.optimizer = optim.Adam([{'params': net.parameters()}, {'params': metric.parameters()}],lr = conf.lr, weight_decay=conf.weight_decay)
            print("Optimaizer is Adam")
        

    def train(self,conf):
        self.net.train()
        current_lr = conf.lr
        iterations = 0
        for e in range(conf.epoch):
            if (e % conf.lr_step) == 0 and e !=0:
                current_lr = self._schedule_lr()
            
            for batch_idx,data in enumerate(self.dataloader):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                embeddings = self.net(inputs)              # embedding features
                thetas = self.metric(embeddings)           # first trainning is softmax then change to arcface
                loss = self.criterion(thetas, labels)      # loss function ce
                loss.backward()
                self.optimizer.step()
                if (batch_idx % conf.step_show) == 0:
                    self.logger.info('Epoch:[{}/{}] Iteration:{}\t\t loss = {:.5f}, lr = {:f}'.format(e ,conf.epoch, batch_idx,loss,current_lr))
                if (iterations % conf.test_step) == 0 and iterations !=0:
                    #check-point structure, ready to save 
                    checkpoint = {"net": self.net.state_dict(),
                              "metric": self.metric.state_dict(),
                              "optimizer": self.optimizer.state_dict(),
                              #"lr_scheduler": scheduler.state_dict(),
                              "epoch": e,
                              "batch_index":batch_idx,
                              "iteration":iterations
                        }
                    backbone_path = osp.join(conf.checkpoints, f"{iterations}.pth")
                    torch.save(checkpoint, backbone_path)
                    accuracy,threshold = test.test(conf,f"./checkpoints/{iterations}.pth")
                    self.logger.info('LFW Test Epoch:[{}] Iteration:{}\t\t accuracy = {:.4f}, threshold = {:.4f}\n'.format(e ,batch_idx, accuracy, threshold))
                iterations = iterations + 1

    def _schedule_lr(self):
        # there is a but in optim.lr_scheduler.StepLR when current equals lr_step, the lr is not current_lr * gamma **(current_epoch // lr-step)
        for params in self.optimizer.param_groups:
            params['lr']/=10
        return params['lr']
        
if __name__ == "__main__":
    train = Train(conf)
    train.train(conf)