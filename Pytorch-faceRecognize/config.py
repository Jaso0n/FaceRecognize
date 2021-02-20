import torch
import torchvision.transforms as T

class Config:
    # network settings
    backbone = 'myfmobile'
    metric = 'arcface'
    embedding_size = 128
    # data preprocess
    input_shape = [3,96,96] #CHW
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.Resize((128,128)),          #resize 250x250 to 128x128
        T.RandomCrop(input_shape[1:]), # crop image to 96x96 randomly
        T.ToTensor(),
        T.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
    ])
    test_transform = T.Compose([
        T.Resize((128,128)),          #resize 250x250 to 128x128
        T.CenterCrop(input_shape[1:]), 
        T.ToTensor(),
        T.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
    ])
    # CASIA cleaned dataset
    train_root = '/home/ubuntu/DataSets/HumanFace/casia_cleaned/CASIA-WebFace/'
    # lfw validation dataset
    lfw_test_root = '/home/ubuntu/Desktop/zhk/caffe_model/LFW/lfw/'
    lfw_test_list = '/home/ubuntu/Desktop/zhk/caffe_model/pairs.txt'

    # training settings
    checkpoints = "checkpoints"
    restore = False
    restore_model = ""
    test_model = "checkpoints/fmobile.pth"

    train_batch_size = 64
    test_batch_size = 60

    epoch = 24
    optimizer = 'sgd'
    lr = 1e-1
    lr_step = 8
    lr_decay = 0.95
    weight_decay = 5e-4
    loss = 'focal_loss'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pin_memory = True
    num_workers = 8 # dataloader

config = Config()