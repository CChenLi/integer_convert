# import model
from utils import IntImage, get_num_correct, make_checkpoint
from torch.utils.data import DataLoader
import csv
import cv2
import time
import matplotlib.pyplot as plt
import torch
from model import IntRec
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm, trange
import time

def run_epoch(data_loader,
              model,
              opt,
              device,
              is_train=True,
              desc = None):
    total_loss = 0
    total_correct = 0
    start = time.time()
    for i, batch in tqdm(enumerate(data_loader),
                         total=len(data_loader),
                         desc=desc):
        images = torch.unsqueeze(batch[0], 1).to(device)
        labels = batch[1].to(device)
        with torch.set_grad_enabled(is_train):
            preds = model(images)
            loss = F.cross_entropy(preds, labels)
            if is_train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += loss
            total_correct += get_num_correct(preds, labels)
    elapsed = time.time() - start
    ms = 'average train loss ' if is_train else 'average valid loss '
    print(ms + ': {}; average correct: {}'.format(total_loss / len(data_loader.dataset),
                                               total_correct / len(data_loader.dataset)))
    print(preds.argmax(dim=1))
    return total_loss, total_correct / len(data_loader.dataset)
    
if __name__=="__main__":
    bs = 16
    root = "saved_model"
    epoch_save = 5
    name = time.asctime()

    device = device = 'cuda' if torch.cuda.is_available() else 'cpu'
    int_image = IntImage('data/processed.csv')
    
    train_size, valid_size = int(len(int_image) * 0.8), int(len(int_image) * 0.1)
    test_size = len(int_image) - train_size - valid_size
    train_set, valid_set, test_set = torch.utils.data.random_split(int_image, [train_size, valid_size, test_size])
    train_loader = DataLoader(train_set, batch_size=bs,shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=bs,shuffle=True)
    test_loader = DataLoader(test_set, batch_size=bs,shuffle=True)
    model = IntRec()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        model.train(True)
        total_loss, acc = run_epoch(train_loader, model, optimizer, device, is_train=True, desc="Train Epoch {}".format(epoch))
        if epoch % epoch_save == 0:
            make_checkpoint(root, name, acc, model, optimizer, total_loss)
    

    



    
    
    
    


