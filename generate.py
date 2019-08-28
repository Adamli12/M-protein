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
import numpy as np
from tensorboardX import SummaryWriter
import pickle
import main

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
            nn.LeakyReLU(0.2,inplace=True),
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
            nn.BatchNorm1d(20),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(20, 20),
            nn.BatchNorm1d(20),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(20, 9),
            nn.Tanh())
 
    def forward(self, x):
        x = self.gen(x)
        return x

class GANmodel():
    def __init__(self,dat):
        self.batch_size = 10
        self.init_num_epoch = 500
        self.iter_num_epoch = 250
        self.k=10
        self.G=generator()
        self.D=discriminator()
        if torch.cuda.is_available():
            self.D = self.D.cuda()
            self.G = self.G.cuda()
        self.criterion=nn.BCELoss()
        self.d_optimizer=torch.optim.Adam(self.D.parameters(), lr=0.001)
        self.g_optimizer=torch.optim.Adam(self.G.parameters(), lr=0.001)
        self.dataset=Mdataset(dat)
        self.dataloader=data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True)
    
    def init_train(self):
        writer=SummaryWriter(comment="init_train")
        for param in self.D.parameters():
            param.requires_grad=True
        for epoch in range(self.init_num_epoch):
            for i, feature in enumerate(self.dataloader):
                num_f = feature.size(0)
                real_f=torch.tensor(feature, dtype=torch.float32)
                real_label = torch.randn(num_f)/10+1
                fake_label = torch.clamp(torch.randn(num_f)/10,min=0)
                # =================train discriminator
                if torch.cuda.is_available():
                    real_f = real_f.cuda()
                    real_label = real_label.cuda()
                    fake_label = fake_label.cuda()
        
                # compute loss of real feature
                real_out = self.D(real_f)
                d_loss_real = self.criterion(real_out, real_label)
                real_score = real_out.mean(dim=0).item()  # closer to 1 means better
        
                # compute loss of fake feature
                z = Variable(torch.randn(num_f, self.G.z_dimension))
                if torch.cuda.is_available():
                    z=z.cuda()
                fake_f = self.G(z)
                fake_out = self.D(fake_f)
                d_loss_fake = self.criterion(fake_out, fake_label)
                fake_score = fake_out.mean(dim=0).item()  # closer to 0 means better
        
                # bp and optimize
                d_loss = (d_loss_real + d_loss_fake)/2
                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()
        
                # ===============train generator
                # compute loss of fake feature
                z = Variable(torch.randn(num_f, self.G.z_dimension))
                if torch.cuda.is_available():
                    z=z.cuda()
                fake_f = self.G(z)
                output = self.D(fake_f)
                g_loss = self.criterion(output, real_label)
        
                # bp and optimize
                self.g_optimizer.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()
            if (epoch+1)%50==0:
                print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f}, D real: {:.6f}, D fake: {:.6f}'.format(epoch, self.init_num_epoch, d_loss.item(), g_loss.item(), real_score, fake_score))
            writer.add_scalars("init_train_loss",{"d_loss":d_loss.item(),"g_loss":g_loss.item()},epoch)
            writer.add_scalars("init_train_score",{"d_real_score":real_score,"d_fake_score":fake_score},epoch)
        writer.close()
        return 0

    def decision_distance(self,svm,x):
        x=x.double()
        w=torch.tensor(svm.coef_)
        b=torch.tensor(svm.intercept_)
        dis=torch.abs(torch.sum(torch.mul(w,x))+b)
        dis=dis/torch.norm(w)
        return dis

    def G_train(self,svm):
        writer=SummaryWriter(comment="G_train")
        for param in self.D.parameters():
            param.requires_grad=False

        for epoch in range(self.iter_num_epoch):
            real_label = torch.randn(self.batch_size)/10+1
            z=torch.randn(self.batch_size,self.G.z_dimension)
            if torch.cuda.is_available():
                real_label=real_label.cuda()
                z=z.cuda()
            fake_f=self.G(z)
            output=self.D(fake_f)#fake score, close to 0 means better
            svm_edge_loss=0
            for feature in fake_f:
                svm_edge_loss+=self.decision_distance(svm,feature)
            svm_edge_loss=(svm_edge_loss/len(fake_f)).float()
            generator_loss=self.criterion(output,real_label)
            g_loss=generator_loss+self.k*svm_edge_loss
            writer.add_scalars("g_loss",{"svm_loss":svm_edge_loss,"generator_loss":generator_loss},epoch)

            self.g_optimizer.zero_grad()
            g_loss.backward()
            self.g_optimizer.step()
            
            if (epoch+1)%50==0:
                print("Epoch[{}/{}], g_loss:{:.6f}, D fake:{:.6f}".format(epoch,self.iter_num_epoch,g_loss.item(),output.data.mean()))
        writer.close()
        return 0

    def generate(self,g_num,scaler):#filter the generated data with cov less than 0
        g_path="data/expert/generated.txt"
        z=torch.randn(1000,self.G.z_dimension)
        if torch.cuda.is_available():
            z=z.cuda()
        fake_f=self.G(z)
        fake_f_np=fake_f.detach().numpy()
        bfsc_fake_f=scaler.inverse_transform(fake_f_np)
        j=0
        deletelist=[]
        for feature in bfsc_fake_f:
            for i in range(len(feature)):
                if i>5 and feature[i]<0:#cov less than 0
                    deletelist.append(j)
                    break
            j+=1
        fake_f_np=np.delete(fake_f_np,deletelist,axis=0)
        li=range(len(fake_f_np))
        chli=np.random.choice(li,int(g_num),replace=False)
        chosenf=fake_f_np[chli]
        np.savetxt(g_path,chosenf,delimiter="\t")
        return 0

    def save(self):
        torch.save(self.G.state_dict(), './generator.pth')
        torch.save(self.D.state_dict(), './discriminator.pth')
        return 0