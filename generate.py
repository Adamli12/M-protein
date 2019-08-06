import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.utils import data
import os
import pandas as pd
import numpy as np

class Mdataset(data.Dataset):
    def __init__(self, path=".", train=True):
        self.path = path
        self.train = train
        if self.train:
            self.train_features=torch.from_numpy(np.loadtxt("ttrain.csv",delimiter="\t"))
            self.train_labels =torch.from_numpy(np.loadtxt("tlabel.csv",delimiter="\t"))
        else:
            raise RuntimeError("Not implemented")
 
    def __getitem__(self, index):
        if self.train:
            feature,label = self.train_features[index], self.train_labels[index]
        else:
            raise RuntimeError("Not implemented")
        return feature,label
 
    def __len__(self):
        if self.train:
            return len(self.train_labels)
        else:
            raise RuntimeError("Not implemented")

# Discriminator
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(9, 20),
            nn.LeakyReLU(0.2),
            nn.Linear(20, 1),
            nn.Sigmoid())
 
    def forward(self, x):
        x = self.dis(x)
        return x
 
 
# Generator
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dimension, 20),
            nn.ReLU(True),
            nn.Linear(20, 9), 
            nn.Tanh())
 
    def forward(self, x):
        x = self.gen(x)
        return x

class GANmodel():
    def __init__(self):
        self.batch_size = 1
        self.num_epoch = 100
        self.z_dimension = 9#for g
        self.G=generator()
        self.D=discriminator()
        if torch.cuda.is_available():
            D = D.cuda()
            G = G.cuda()
        self.criterion=nn.BCELoss()
        self.d_optimizer=torch.optim.Adam(D.parameters(), lr=0.0003)
        self.g_optimizer=torch.optim.Adam(G.parameters(), lr=0.0003)
        self.dataset=Mdataset()
        self.dataloader=data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True)
    
    def init_train():
        for epoch in range(self.num_epoch):
            for i, (feature, _) in enumerate(dataloader):
                num_img = img.size(0)#把img像上一行全部改成feature！不用view！！！
                # =================train discriminator
                img = img.view(num_img, -1)
                real_img = Variable(img).cuda()
                real_label = Variable(torch.ones(num_img)).cuda()
                fake_label = Variable(torch.zeros(num_img)).cuda()
        
                # compute loss of real_img
                real_out = D(real_img)
                d_loss_real = criterion(real_out, real_label)
                real_scores = real_out  # closer to 1 means better
        
                # compute loss of fake_img
                z = Variable(torch.randn(num_img, z_dimension)).cuda()
                fake_img = G(z)
                fake_out = D(fake_img)
                d_loss_fake = criterion(fake_out, fake_label)
                fake_scores = fake_out  # closer to 0 means better
        
                # bp and optimize
                d_loss = d_loss_real + d_loss_fake
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()
        
                # ===============train generator
                # compute loss of fake_img
                z = Variable(torch.randn(num_img, z_dimension)).cuda()
                fake_img = G(z)
                output = D(fake_img)
                g_loss = criterion(output, real_label)
        
                # bp and optimize
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()
        
                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '
                        'D real: {:.6f}, D fake: {:.6f}'.format(
                            epoch, self.num_epoch, d_loss.data[0], g_loss.data[0],
                            real_scores.data.mean(), fake_scores.data.mean()))
 
torch.save(G.state_dict(), './generator.pth')
torch.save(D.state_dict(), './discriminator.pth')