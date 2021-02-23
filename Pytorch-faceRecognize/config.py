import torch
import torchvision.transforms as T

class Config:
    # network settings
    backbone = 'myfmobile'
    metric = 'softmax'
    embedding_size = 128
    # data preprocess
    input_shape = [3,96,96] #CHW
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.Resize((128,128)),           #resize 250x250 to 128x128
        T.RandomCrop(input_shape[1:]), # crop image to 96x96 randomly
        T.ToTensor(),
        T.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
    ])
    test_transform = T.Compose([
        T.Resize((128,128)),           #resize 250x250 to 128x128
        T.CenterCrop(input_shape[1:]), 
        T.ToTensor(),
        T.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
    ])
    # CASIA cleaned dataset
    train_root = '/home/ubuntu/DataSets/HumanFace/casia_cleaned/CASIA-WebFace/'
    # lfw validation dataset
    lfw_test_root = '/home/ubuntu/Desktop/zhk/caffe_model/LFW/lfw/'
    lfw_test_list = '/home/ubuntu/FaceRecognize/Pytorch-faceRecognize/pairs.txt'

    # training settings
    checkpoints_path = "checkpoints"
    restore = False
    restore_model = ""

    train_batch_size = 64
    test_batch_size = 60

    test_step = 1000     # testing on lfw step
    step_show = 50       # display step

    #save_step = 5000
    MAX_EPOCH = 30       # max epoch
    optimizer = 'sgd'    # solver
    lr_gamma = 10
    lr = 1e-1            # base learning rate
    lr_step = 8          # learning rate changing step, every K epoch: lr = lr * gamma^(current_epoch // lr_step)
    momentum = 0.9       # momentum in solver
    weight_decay = 4e-5  # network general weight_decay, L2 norm in weight update to prevent overfit
    class_wd = 4e-4      # weight decay for classifier layer, eg arcface/full connect.
    loss = 'focal_loss'  # focal loss is based on softmax Loss
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pin_memory = True
    num_workers = 8      # dataloader working thread

config = Config()