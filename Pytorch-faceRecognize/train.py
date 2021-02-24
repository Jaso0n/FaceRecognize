import os
import os.path as osp
import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim

from model import MobileNetV2_FaceNet as MobileFaceNet
from metric import ArcFace,DenseClassifier,NormLinear
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
    def __init__(self, ConfigTable):
        """ Train on pytorch:
            Step1: Create a dataloader to prepare train data.
            Step2: Create a logger to write train imfomartion to your .log.
            Step3: Choose your network backbone, metric function, loss function.
            Step4: In pytorch the weight_decay will be applied to bias, batchnorm layer, change weight_decay to 0 at these params.
            Step5: Define the optimizer eg. SGD, Adam
            Step6: Now, let's start your train
        """
        self.config = ConfigTable  # Config table for train including lr_step, checkpoints
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
            self.net = MobileFaceNet(self.embedding_size).to(self.device)   # Create net and copy net tensor to the GPU, do it before loading data
            self.logger.info("Network backbone is {}".format(self.config.backbone))
            self.logger.info("{}".format(self.net))
        else:
            self.net = MobileFaceNet(self.embedding_size).to(self.device)

        if self.config.metric == 'arcface':
            self.metric = ArcFace(self.embedding_size, self.class_num).to(self.device)
            self.logger.info("Metric fucntion is {}".format(self.config.metric))
            self.logger.info("{}".format(self.metric))

        elif self.config.metric == 'softmax':
            self.metric = DenseClassifier(self.embedding_size, self.class_num).to(self.device)
            self.logger.info("Metric fucntion is {}".format(self.config.metric))
            self.logger.info("{}".format(self.metric))

        elif self.config.metric == 'normlinear':
            self.metric = NormLinear(self.embedding_size, self.class_num).to(self.device)
            self.logger.info("Metric fucntion is {}".format(self.config.metric))
            self.logger.info("{}".format(self.metric))
        else:
            self.logger.info("Please specify a metric")
            exit(0)

        self._weight_init()
        # Send data to multiple gpu
        self.net = nn.DataParallel(self.net)
        self.metric = nn.DataParallel(self.metric)
        # Remove weight_decay in batchnorm and convolution bias, refer to https://arxiv.org/abs/1706.05350
        net_params = add_weight_decay(self.net,self.config.weight_decay)
        metric_params = add_weight_decay(self.metric,self.config.class_wd)
        self.parameters = net_params + metric_params

        if self.config.loss == 'focal_loss':
            self.criterion = FocalLoss(gamma = 2)
            self.logger.info("Loss function is FocalLoss")
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.logger.info("Loss function is CrossEntropyLoss")
        
        if self.config.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.parameters, lr = self.config.lr, momentum=self.config.momentum, weight_decay = self.config.weight_decay)
            self.logger.info("Optimaizer is SGD")
            self.logger.info("{}".format(self.optimizer))
        else:
            self.optimizer = optim.Adam(self.parameters,lr = self.config.lr, weight_decay=self.config.weight_decay)
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

    def _learner(self, optimizer, model, metric, epoch, dataloader):
        model.train()
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            iteration = epoch * len(dataloader) + batch_idx
            # net start to forward
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            x = model(inputs)                 # embedding features
            thetas = metric(x, labels) if self.config.metric == 'arcface' else metric(x)
            loss = self.criterion(thetas, labels)             # loss function ce
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (batch_idx % self.config.step_show) == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    #prec1, prec5 = _get_accuracy(output.data, target, topk=(1,5))
                    self.logger.info('Epoch:[{}/{}] Iteration:{}\t loss = {:.4f}\t lr = {:f}'.format(
                        epoch ,self.config.MAX_EPOCH, iteration, loss, current_lr))
    
    def train(self,start_epoch):
        optimizer = self.optimizer
        model = self.net
        metric = self.metric
        dataloader = self.dataloader
        test_best_acc = 0
        best_epoch = 0
        for epoch in range(start_epoch, self.config.MAX_EPOCH):
            if (epoch % self.config.lr_step) == 0 and epoch !=0:
                self._schedule_lr(optimizer)
            
            self._learner(optimizer, model, metric, epoch, dataloader)

            model_name = 'checkpoint_{}.pth'.format(epoch)
            if epoch >= self.config.test_step:
                accuracy, threshold = test.test(self.config, model)
                self.logger.info('Start to test, accuracy = {}, threshold = {}'.format(accuracy, threshold))
                if accuracy > test_best_step : 
                    test_best_step = accuracy
                    best_epoch = epoch
                    checkpoint = {
                        "epoch" : best_epoch,
                        "lr" : optimizer.param_groups[0]['lr'],
                        "net" : model.state.dict(),
                        "metric" : metric.state_dict(),
                        "acc" : test_best_step
                    }
                    torch.save(checkpoint, model_name)
                    self.logger.info('Save model to \'{}\''.format(model_name))
    
    def _resume_train(self, model_path, lr): # only for arcface, change 'metric' in config to arcface to make _learner() work
        checkpoint = torch.load(model_path)
        model = MobileFaceNet(self.config.embedding_size).to(self.device)
        metric = ArcFace(self.embedding_size, self.class_num).to(self.device)
        model = nn.DataParallel(model)
        metric = nn.DataParallel(metric)

        model.load_state_dict(checkpoint['net'])
        
        net_params = add_weight_decay(model,self.config.weight_decay)
        metric_params = add_weight_decay(metric,self.config.class_wd)
        parameters = net_params + metric_params
        optimizer = optim.SGD(parameters, lr = lr, momentum = self.config.momentum, weight_decay = self.config.weight_decay)
        current_lr = optimizer.param_groups[0]['lr']
        self.logger.info('Resume train {}, metric state is arcface, basic learning rate is {:f}'.format(model_path, current_lr))
        accuracy,threshold = test.test(self.config, model)     #f"./{self.config.checkpoints_path}/epoch{e}_{batch_idx}.pth"
        self.logger.info('Loading model from \'{}\',\t test accuracy = {:.4f}, threshold = {:.4f}\n'.format(model_path,accuracy, threshold))
        self._learner(optimizer, model, metric)

if __name__ == "__main__":
    resume_train_model = './softmax_loss_checkpoints/210000.pth'
    train = Train(conf)
    #train.resume_train(conf,'./checkpoints/210000.pth','arcface',0.001)
    train.train(0)