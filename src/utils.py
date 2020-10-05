import csv
import os
import os.path as osp
from os import listdir, rmdir
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt

def default_loader(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return torch.Tensor(img)

def get_label(x):
    if x == 100:
        return 11
    if x == 1000:
        return 12
    if x == 10000:
        return 13
    if x == 100000000:
        return 14
    return x

class IntImage(Dataset):
    def __init__(self, image_path, loader=default_loader):
        with open(image_path, newline='') as f:
            reader = csv.reader(f)
            self.image_path = list(reader)
        self.loader = loader

    def __getitem__(self, index):
        fn = self.image_path[index]
        img = self.loader(fn[1])
        label = get_label(int(fn[2]))
        return img,label

    def __len__(self):
        return len(self.image_path)

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

def make_checkpoint(root, name, epoch, model, optimizer, loss):
    if not osp.exists(root):
        import os
        os.mkdir(root)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, osp.join(root, name + '_' + str(epoch) + ".pickle"))

def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

if __name__=="__main__":
    image_path = 'data/processed.csv'
    intimage = IntImage(image_path)
    imageloader = DataLoader(intimage, batch_size=4,shuffle=True)
    model = IntRec()
    batch = next(iter(imageloader))
    model(torch.unsqueeze(batch[0], 1))

