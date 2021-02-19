import torch
import torchvision.transforms as T

class Config:
    # network settings
    backbone = 'myfmobile'
    metric = 'arcface'
    embedding_size = 128
    # data preprocess
    input_shape = [3,112,112]
    # CASIA cleaned dataset
    train_root = ''
    test_root = ''
    test_list = ''