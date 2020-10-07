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
    kernal = np.ones((2,2),np.uint8)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.dilate(img, kernal, iterations=1)
    _, img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
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


class HelperFunc():

    @staticmethod
    def get_range(indexs):
        start_index = 0
        end_index = 0
        for i in indexs:
            if i:
                break
            start_index += 1
            end_index += 1
        for i in indexs[start_index :]:
            if not i:
                return start_index, end_index
            end_index += 1
        return start_index, -1

    @staticmethod
    def get_slices(indexs):
        pairs = []
        span = []
        start_index = 0
        end_index = 0
        while True:
            start_index, temp_index = HelperFunc.get_range(indexs[end_index:])
            if temp_index == -1:
                break
            start_index += end_index
            end_index += temp_index
            pairs.append([start_index, end_index])
            span.append(end_index - start_index)
        threshold = 0.2 * sum(span) / len(span)
        for i, diff in enumerate(span):
            if diff < threshold:
                pairs.remove(pairs[i])
                span.remove(span[i])
        return pairs

    @staticmethod
    def load_checkpoint(path, model, optimizer):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return epoch, loss

    @staticmethod
    def get_characters(slice):
        vertical_mask = torch.ones(1, slice.size(0))
        vertical_index = torch.mm(vertical_mask, slice)
        cuts = HelperFunc.get_slices((vertical_index != 0).view(-1))
        span = []
        for i in cuts:
            span.append(i[1] - i[0])
        threshold = 0.8 * sum(span) / len(span)
        idx_rm = []
        for i, diff in enumerate(span[:len(span)-1]):
            if diff < threshold:
                if i + 1 < len(span):
                    if span[i+1] < threshold * 1.2:
                        cuts[i][1] = cuts[i+1][1]
                        idx_rm.append(i+1)
                        threshold = 0.8 * sum(span) / len(span)
        offset = 0
        for idx in idx_rm:
            del cuts[idx - offset]
            offset += 1
        return cuts




if __name__=="__main__":
    x = [0,1,1,0,0,1,1,1,0]
    HelperFunc.get_slices(x)
