import csv
import cv2
import time
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from src.utils import IntImage, get_num_correct, HelperFunc
from PIL import Image, ImageOps
import numpy as np
import pickle
from argparse import ArgumentParser
import torch.nn as nn

def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

class IntRec(nn.Module):
    def __init__(self):
        super(IntRec, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12*13*13, out_features=240)
        self.fc2 = nn.Linear(in_features=240, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=15)

    def forward(self, t):
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = t.reshape(-1, 12 * 13 * 13)
        t = self.fc1(t)
        t = F.relu(t)

        t = self.fc2(t)
        t = F.relu(t)

        t = self.fc3(t)
        t = F.relu(t)

        t = self.out(t)
        # t = F.softmax(t, dim=1)

        return t

class IntConversion(nn.Module):
    def __init__(self, model, pad):
        super(IntConversion, self).__init__()
        self.model = model
        self.pad = pad
        self.kernal = np.ones((2,2),np.uint8)

    def pic_preprocess(self, path):
        pic = cv2.imread(path)
        gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        pic_tensor = torch.Tensor(bw)
        horizontal_mask = torch.ones(pic_tensor.size(1), 1)
        horizontal_index = torch.mm(pic_tensor, horizontal_mask)
        slices = HelperFunc.get_slices((horizontal_index != 0).view(-1))
        return slices, pic_tensor

    def trans(self, cuts, fig, example_slice):
        for i, cut in enumerate(cuts):
            fig.add_subplot(1, len(cuts), i+1)
            start_index, end_index = cut
            buff = int(0.2 * (end_index - start_index))
            short_pic_tensor = example_slice[: ,start_index-buff : end_index + buff]
            # Trim the data
            img = transforms.ToPILImage(mode='L')(short_pic_tensor)
            p = transforms.Compose([transforms.Scale((self.pad, self.pad))])
            new_img = ImageOps.expand(p(img),border=((64-self.pad) // 2),fill='black')
            new_img = np.array(new_img)
            new_img = cv2.dilate(new_img,self.kernal,iterations = 1)
            # Prediction
            trans = transforms.ToTensor()
            new_tensor = torch.unsqueeze(trans(new_img), 0)
            new_tensor[new_tensor > 0] = 255
            pred = self.model(new_tensor).argmax()
            plt.imshow(new_tensor.reshape(64, 64), cmap="gray")
            plt.title(f"pred: {pred}")

    def forward(self, path):
        slices, pic_tensor = self.pic_preprocess(path)
        for i, slice in enumerate(slices):
            example_slice = pic_tensor[slice[0] : slice[1], : ]
            cuts = HelperFunc.get_characters(example_slice)
            fig = plt.figure(figsize=[2 * len(cuts), 2])
            self.trans(cuts, fig, example_slice)
            plt.savefig(f"processed/pic_line_{i}.png")

def main():
    parser = ArgumentParser()
    parser.add_argument('path', type=str, help='path to the picture')
    path = args = parser.parse_args().path
    print(path)
    model = IntRec()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    load_checkpoint("saved_model/model1.pickle", model, optimizer)
    converter = IntConversion(model, 28)
    converter(path)

if __name__=="__main__":
    main()
