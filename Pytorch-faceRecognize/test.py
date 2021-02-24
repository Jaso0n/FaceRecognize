import os
import os.path as osp

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from config import Config as conf
from model import MobileNetV2_FaceNet as MobileFaceNet
"""
    Only for LFW test with specific pair.txt
"""
def _getImageSet(pair_list) ->set :
    # pair_list: name1 name2 label
    with open (pair_list, 'r') as f:
        pairs = f.readlines()
    unique = set()
    for pair in pairs:
        id1, id2, _ = pair.split()
        unique.add(id1)
        unique.add(id2)
    return unique

def _groupImage(images: set, batch) -> list:
    images = list(images)
    size = len(images)
    ret = []
    k=0
    for i in range(0,size,batch):
        end = min(batch + batch * k, size)
        ret.append(images[i:end])
        k = k+1
    return ret

def _preprocess(images: list, transform) -> torch.Tensor:
    ret = []
    for img in images:
        im = Image.open(img)
        im = transform(im)
        ret.append(im)
    data = torch.stack(ret, dim=0) # torch.cat is different from torch.stack
    return data

def _getFeature(images:list, transform, net, device) -> dict:
    data = _preprocess(images, transform)
    data = data.to(device)
    net = net.to(device)
    with torch.no_grad():
        features = net(data)
    res = {img: feature for (img, feature) in zip(images,features)}
    return res

def CalcuSimilarity(x1,x2):
    return np.dot(x1,x2) / (np.linalg.norm(x1) * np.linalg.norm(x2) + 1e-10)

def searchThreshold(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th
    return best_acc, best_th

def compute_accuracy(feature_dict, pair_list, test_root):
    with open(pair_list, 'r') as f:
        pairs = f.readlines()

    similarities = []
    labels = []
    for pair in pairs:
        img1, img2, label = pair.split()
        img1 = osp.join(test_root, img1)
        img2 = osp.join(test_root, img2)
        feature1 = feature_dict[img1].cpu().numpy()
        feature2 = feature_dict[img2].cpu().numpy()
        label = int(label)

        similarity = CalcuSimilarity(feature1, feature2)
        similarities.append(similarity)
        labels.append(label)

    accuracy, threshold = searchThreshold(similarities, labels)
    return accuracy, threshold

def test(conf,net,model_path=0):
    #model = MobileFaceNet(conf.embedding_size)
    #model = nn.DataParallel(model)
    #checkpoint = torch.load(model_path)
    #model.load_state_dict(checkpoint['net'])
    net.eval()

    images = _getImageSet(conf.lfw_test_list)
    images = [osp.join(conf.lfw_test_root, img) for img in images]
    groups = _groupImage(images, conf.test_batch_size)

    feature_dict = dict()
    for group in groups:
        d = _getFeature(group, conf.test_transform, net, conf.device)
        feature_dict.update(d) 
    accuracy, threshold = compute_accuracy(feature_dict, conf.lfw_test_list, conf.lfw_test_root) 
    return accuracy, threshold
    