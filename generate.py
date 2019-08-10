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
from active import save_in_train_all

class Mdataset(data.Dataset):
    def __init__(self, dat, train=True):
        self.train = train
        if self.train:
            self.train_features=torch.from_numpy(dat)
        else:
            raise RuntimeError("Not implemented")
 
    def __getitem__(self, index):
        if self.train:
            feature= self.train_features[index]
        else:
            raise RuntimeError("Not implemented")
        return feature
 
    def __len__(self):
        if self.train:
            return self.train_features.shape[0]
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
        self.z_dimension=9
        self.gen = nn.Sequential(
            nn.Linear(self.z_dimension, 20),
            nn.ReLU(True),
            nn.Linear(20, 9), 
            nn.Tanh())
 
    def forward(self, x):
        x = self.gen(x)
        return x

class GANmodel():
    def __init__(self,dat):
        self.batch_size = 1
        self.init_num_epoch = 100
        self.iter_num_epoch = 10
        self.k=1
        self.G=generator()
        self.D=discriminator()
        if torch.cuda.is_available():
            self.D = self.D.cuda()
            self.G = self.G.cuda()
        self.criterion=nn.BCELoss()
        self.d_optimizer=torch.optim.Adam(self.D.parameters(), lr=0.0003)
        self.g_optimizer=torch.optim.Adam(self.G.parameters(), lr=0.0003)
        self.dataset=Mdataset(dat)
        self.dataloader=data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True)
    
    def init_train(self):
        for param in self.D.parameters():
            param.requires_grad=True
        for epoch in range(self.init_num_epoch):
            for i, feature in enumerate(self.dataloader):
                num_f = feature.size(0)
                # =================train discriminator
                real_f = Variable(feature).cuda()
                real_label = Variable(torch.ones(num_f)).cuda()
                fake_label = Variable(torch.zeros(num_f)).cuda()
        
                # compute loss of real feature
                real_out = self.D(real_f)
                d_loss_real = self.criterion(real_out, real_label)
                real_scores = real_out  # closer to 1 means better
        
                # compute loss of fake feature
                z = Variable(torch.randn(num_f, self.G.z_dimension)).cuda()
                fake_f = self.G(z)
                fake_out = self.D(fake_f)
                d_loss_fake = self.criterion(fake_out, fake_label)
                fake_scores = fake_out  # closer to 0 means better
        
                # bp and optimize
                d_loss = d_loss_real + d_loss_fake
                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()
        
                # ===============train generator
                # compute loss of fake feature
                z = Variable(torch.randn(num_f, self.G.z_dimension)).cuda()
                fake_f = self.G(z)
                output = self.D(fake_f)
                g_loss = self.criterion(output, real_label)
        
                # bp and optimize
                self.g_optimizer.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()
        
                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f}, D real: {:.6f}, D fake: {:.6f}'.format(epoch, self.init_num_epoch, d_loss.data[0], g_loss.data[0], real_scores.data.mean(), fake_scores.data.mean()))
        return 0

    def G_train(self,svm):
        for param in self.D.parameters():
            param.requires_grad=False

        for epoch in range(self.iter_num_epoch):
            real_label=torch.ones(self.batch_size).cuda()
            z=torch.randn(self.batch_size,self.G.z_dimension).cuda()
            fake_f=self.G(z)
            output=self.D(fake_f)#fake score, close to 1 means better
            svm_edge_loss=0
            for feature in fake_f:
                svm_edge_loss+=abs(svm.decision_function(feature))#可以这么写吗？？？应该会传不过去吧
            g_loss=self.criterion(output,real_label)+self.k*svm_edge_loss

            self.g_optimizer.zero_grad()
            g_loss.backward()
            self.g_optimizer.step()
            
            print("Epoch[{}/{}], g_loss:{:.6f}, D fake:{:.6f}".format(epoch,self.iter_num_epoch,g_loss.data[0],output.data.mean()))
        return 0

    def generate(self,g_num):
        g_path="data/expert/generated.txt"
        z=torch.randn(g_num,self.G.z_dimension).cuda()
        fake_f=self.G(z)
        np.savetxt(g_path,fake_f,delimiter="\t")
        return 0

    def generate_balance(self,svm,g_num,mindis):
        z=torch.randn(g_num,self.G.z_dimension).cuda()
        fake_f=self.G(z)
        deletelist=[]
        for i in range(len(fake_f)):
            if abs(svm.decision_function(fake_f[i]))<mindis:
                deletelist.append(i)
        fake_f=np.delete(fake_f,deletelist,axis=0)
        label=svm.predict(fake_f)
        fake_f=np.hstack(fake_f,label)
        save_in_train_all(fake_f,0)
        return 0

    def save(self):
        torch.save(self.G.state_dict(), './generator.pth')
        torch.save(self.D.state_dict(), './discriminator.pth')
        return 0