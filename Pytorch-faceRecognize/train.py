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
        """ Train on pytorch:
            Step1: Create a dataloader to prepare train data.
            Step2: Create a logger to write train imfomartion to your .log.
            Step3: Choose your network backbone, metric function, loss function.
            Step4: In pytorch the weight_decay will be applied to bias, batchnorm layer, change weight_decay to 0 at these params.
            Step5: Define the optimizer eg. SGD, Adam
            Step6: Now, let's start your train
        """
        self.config = conf  # Config table for train including lr_step, checkpoints
        os.makedirs(self.config.checkpoints_path, exist_ok = True)
        nowtime = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
        self.logger = get_logger("./"+ nowtime +"-train.log")
        self.logger.info('Successfully create train log file at \'./{}-train.log\''.format(nowtime))
        # dataloader
        self.dataloader, self.class_num = load_data(self.config, training=True)
        self.embedding_size = self.config.embedding_size
        self.device = self.config.device
        self.logger.info('Successfully create DataLoader. In data the total of class numbers is {} and embedding size is {}'.format(self.class_num,self.embedding_size))
        # generate network
        if self.config.backbone == 'myfmobile':
            self.net = MobileFaceNet(self.embedding_size).to(self.device)
            self.logger.info("Network backbone is {}".format(self.config.backbone))
            self.logger.info("{}".format(self.net))
        else:
            self.net = MobileFaceNet(self.embedding_size).to(self.device)

        if self.config.metric == 'arcface':
            self.metric = ArcFace(self.embedding_size, self.class_num).to(self.device)
            self.logger.info("Metric fucntion is {}".format(self.config.metric))
            self.logger.info("{}".format(self.metric))
        else:
            self.metric = DenseClassifier(self.embedding_size, self.class_num).to(self.device)
            self.logger.info("Metric fucntion is {}".format(self.config.metric))
            self.logger.info("{}".format(self.metric))
        
        self._weight_init()
        # Send data to multiple gpu
        self.net = nn.DataParallel(self.net)
        self.metric = nn.DataParallel(self.metric)
        # Remove weight_decay in batchnorm and convolution bias, refer to https://arxiv.org/abs/1706.05350
        net_params = add_weight_decay(self.net,conf.weight_decay)
        metric_params = add_weight_decay(self.metric,conf.class_wd)
        self.parameters = net_params + metric_params

        if conf.loss == 'focal_loss':
            self.criterion = FocalLoss(gamma = 2)
            self.logger.info("Loss function is FocalLoss")
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.logger.info("Loss function is CrossEntropyLoss")
        
        if conf.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.parameters, lr = conf.lr, momentum=conf.momentum, weight_decay = conf.weight_decay)
            self.logger.info("Optimaizer is SGD")
            self.logger.info("{}".format(self.optimizer))
        else:
            self.optimizer = optim.Adam(self.parameters,lr = conf.lr, weight_decay=conf.weight_decay)
            self.logger.info("Optimaizer is Adam")
            self.logger.info("{}".format(self.optimizer))
    
    def _schedule_lr(self,optimizer):
        # there is a but in optim.lr_scheduler.StepLR when current equals lr_step, the lr is not current_lr * gamma **(current_epoch // lr-step)
        for params in optimizer.param_groups:
            params['lr'] = params['lr'] / self.config.lr_gamma
        return optimizer.param_groups[0]['lr']

    def _weight_init(self):
        for op in self.net.modules():
            if isinstance(op, nn.Conv2d):
                nn.init.kaiming_uniform_(op.weight.data,nonlinearity="relu")
                #nn.init.kaiming_uniform_(op.weight.bias,val=0)
            elif isinstance(op, nn.Linear):
                nn.init.normal_(op.weight.data) # default mean=0, std =1
                #nn.init.constant_(op.weight.bias,val=0)

    def _learner(self, optimizer, net, metric):
        iterations = 0
        current_lr = self.optimizer.param_groups[0]['lr']
        self.logger.info("Start to train, basic learning rate is {}. Every {} epochs lr divided by {}".format(current_lr, self.config.lr_step, self.config.lr_gamma))
        for e in range(0, self.config.MAX_EPOCH):
            if (e % self.config.lr_step) == 0 and e !=0:
                current_lr = self._schedule_lr(optimizer)
            for batch_idx, data in enumerate(self.dataloader):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                embedding_feature = net(inputs)              # embedding features
                thetas = metric(embedding_feature, labels) if self.config.metric == 'arcface' else metric(embedding_feature)
                loss = self.criterion(thetas, labels)             # loss function ce
                loss.backward()
                optimizer.step()

                if (batch_idx % self.config.step_show) == 0:
                    self.logger.info('Epoch:[{}/{}] Iteration:{}\t loss = {:.4f}, lr = {:f}'.format(e ,self.config.MAX_EPOCH, batch_idx,loss,current_lr))
                if (iterations % self.config.test_step) == 0 and iterations !=0: #self.config.test_step
                    #check-point structure, ready to save 
                    checkpoint = {"net": self.net.state_dict(),
                                  "metric": self.metric.state_dict(),
                                  "optimizer": self.optimizer.state_dict(),
                                  "epoch": e,
                                  "batch_index":batch_idx,
                                  "iteration":iterations}
                    
                    backbone_path = osp.join(self.config.checkpoints_path,f"epoch{e}_{batch_idx}.pth")
                    torch.save(checkpoint, backbone_path) #save modle
                    accuracy,threshold = test.test(conf,self.net)     #f"./{self.config.checkpoints_path}/epoch{e}_{batch_idx}.pth"
                    self.logger.info('Saveing model to \'epoch{}_{}.pth\',\t test accuracy = {:.4f}, threshold = {:.4f}\n'.format(e ,batch_idx, accuracy, threshold))
                iterations = iterations + 1

    def train(self):
        self.net.train()
        self._learner(self.optimizer, self.net, self.metric)
    
    def resume_train(self, model_name, metric_state, lr):
        checkpoint = torch.load(f"./{self.config.checkpoint_path}/{model_name}")
        self.net.load_state_dict(checkpoint['net'])
        iterations = checkpoint['iteration']
        optimizer = optim.SGD(self.parameters, lr = lr, momentum=self.config.momentum, weight_decay = self.config.weight_decay)
        current_lr = optimizer.param_groups[0]['lr']
        self.logger.info('Resume train {}, metric state is {}, learning rate is {:f}'.format(checkpoint_path, metric_state,current_lr))
        for e in range(0,self.config.MAX_EPOCH):
            if (e % 10) == 0 and e != 0:
                current_lr = self._schedule_lr(optimizer)
            for batch_idx,data in enumerate(self.dataloader):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                embeddings = self.net(inputs)              # embedding features
                thetas = self.metric(embeddings,labels) if self.config.metric == 'arcface' else self.metric(embeddings)
                loss = self.criterion(thetas, labels)      # loss function ce
                loss.backward()
                optimizer.step()
                if (batch_idx % self.config.step_show) == 0:
                    self.logger.info('Epoch:[{}/{}] Iteration:{}\t\t loss = {:.5f}, lr = {:f}'.format(e ,conf.epoch, batch_idx,loss,current_lr))
                if (iterations % 100) == 0 and iterations !=0:
                    #check-point structure, ready to save 
                    checkpoint = {"net": self.net.state_dict(),
                              "metric": self.metric.state_dict(),
                              "optimizer": self.optimizer.state_dict(),
                              "epoch": e,
                              "batch_index":batch_idx,
                              "iteration":iterations
                        }
                    backbone_path = osp.join(conf.checkpoints, f"{iterations}.pth")
                    torch.save(checkpoint, backbone_path)
                    accuracy,threshold = test.test(conf,self.net)#f"./checkpoints/{iterations}.pth"
                    self.logger.info('LFW Test Epoch:[{}] Iteration:{}\t\t accuracy = {:.4f}, threshold = {:.4f}\n'.format(e ,batch_idx, accuracy, threshold))
                iterations = iterations + 1

if __name__ == "__main__":
    train = Train(conf)
    #train.resume_train(conf,'./checkpoints/210000.pth','arcface',0.001)
    train.train()