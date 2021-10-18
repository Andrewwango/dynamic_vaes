import numpy as np
from tqdm import tqdm
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import os
import numpy as np

def export_vid(model, dataloader, state_dict=None, epoch=0, j=0, both=True, ret=False):
    t = torch.Tensor(dataloader.dataset[j:j+64])
    x,x_hat = model.eval(t)

    print(x.shape, x_hat.shape)
    fig = plt.figure()
    x_mat = np.vstack([np.hstack([x[0,:,:], x_hat[0,:,:]]), np.hstack([x[1,:,:], x_hat[1,:,:]])])
    plt.imshow(x_mat, cmap='gray')
    plt.savefig(f"results/test_x_{epoch}.jpg", bbox_inches='tight')
    plt.close()

def plot_loss(train, val, title, dir):
    plt.clf()
    fig = plt.figure(figsize=(8,6))
    plt.rcParams['font.size'] = 12
    plt.plot(train, label='training loss')
    plt.plot(val, label='validation loss')
    plt.legend(fontsize=16, title="loss_"+title, title_fontsize=20)
    plt.xlabel('epochs', fontdict={'size':16})
    plt.ylabel('loss', fontdict={'size':16})
    plt.savefig(os.path.join(dir, f'loss_{title}.png'))

def load_bouncing_ball(datafolder, dataname, eps=1e-6, static=False, singular=False):
    def load_images(path):
        npzfile = np.load(path)
        pics = (npzfile['images'].astype(np.float32) > 0).astype('float32') + eps
        if static:
            pics = np.repeat(pics[:,10,:,:][:,None,:,:], pics.shape[1], axis=1)
        if singular:
            pics = pics[:, 0, :, :][:, :, :] #batch x dim x dim
        return pics
    train_images = load_images(f"{datafolder}/{dataname}.npz")
    test_images  = load_images(f"{datafolder}/{dataname}_test.npz")

    return train_images, test_images

class CVAE:
    def __init__(self, model, lr, epochs, batch_size):
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, train_dataloader, val_dataloader):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        train_losses = []
        val_losses = []

        for epoch in range(self.epochs):
            print("Epoch", epoch)
            self.model.train()
            running_loss = 0.0
            for i, data in tqdm(enumerate(train_dataloader)):
                optimizer.zero_grad()
                _ = self.model(data, compute_loss=True)
                loss = self.model.loss
                loss.backward()
                running_loss += loss.item()
                optimizer.step()
            train_loss = running_loss / i
            train_losses += [train_loss]

            self.model.eval()
            running_loss = 0.0
            with torch.no_grad():
                for i, data in tqdm(enumerate(val_dataloader)):
                    _ = self.model(data, compute_loss=True)
                    loss = self.model.loss
                    running_loss += loss.item()
            val_loss = running_loss / i
            val_losses += [val_loss]

            print("Train loss", train_loss, "Val loss", val_loss)
            export_vid(self, val_dataloader, epoch=epoch, both=True)
        
        plot_loss(train_losses, val_losses, "", "results")
    
    def eval(self, x):
        self.model.eval()
        with torch.no_grad():
            x_hat = self.model(x)[0].to('cpu').detach().numpy()
        return x,x_hat