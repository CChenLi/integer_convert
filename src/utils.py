import csv
import os
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

if __name__=="__main__":
    image_path = 'data/processed.csv'
    intimage = IntImage(image_path)
    imageloader = DataLoader(intimage, batch_size=4,shuffle=True)
    model = IntRec()
    batch = next(iter(imageloader))
    model(torch.unsqueeze(batch[0], 1))

