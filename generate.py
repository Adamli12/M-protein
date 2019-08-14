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

def save_in_train_all(dat,mode):
    if mode==1:
        np.savetxt("data/train/generated.txt",dat)
        gen_str=pickle.dumps(dat)
        num=len(os.listdir("data/all/generated"))
        f=open("data/all/generated/"+str(num)+".txt","wb")
        f.write(gen_str)
        f.close()
    if mode==0:
        np.savetxt("data/train/balancing.txt",dat)
        gen_str=pickle.dumps(dat)
        num=len(os.listdir("data/all/balancing"))
        f=open("data/all/balancing/"+str(num)+".txt","wb")
        f.write(gen_str)
        f.close()
    return 0

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
        self.iter_num_epoch = 100
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
        self.meanflag=0
        self.dismean=0
    
    def init_train(self):
        writer=SummaryWriter(comment="init_train")
        for param in self.D.parameters():
            param.requires_grad=True
        for epoch in range(self.init_num_epoch):
            for i, feature in enumerate(self.dataloader):
                num_f = feature.size(0)
                real_f=torch.tensor(feature, dtype=torch.float32)
                real_label = torch.ones(num_f)
                fake_label = torch.zeros(num_f)
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
                d_loss = d_loss_real + d_loss_fake
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
            real_label=torch.ones(self.batch_size)
            z=torch.randn(self.batch_size,self.G.z_dimension)
            if torch.cuda.is_available():
                real_label=real_label.cuda()
                z=z.cuda()
            fake_f=self.G(z)
            output=self.D(fake_f)#fake score, close to 1 means better
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
            
            print("Epoch[{}/{}], g_loss:{:.6f}, D fake:{:.6f}".format(epoch,self.iter_num_epoch,g_loss.item(),output.data.mean()))
        writer.close()
        return 0

    def generate(self,g_num):
        g_path="data/expert/generated.txt"
        z=torch.randn(g_num,self.G.z_dimension)
        if torch.cuda.is_available():
            z=z.cuda()
        fake_f=self.G(z)
        fake_f_np=fake_f.detach().numpy()
        np.savetxt(g_path,fake_f_np,delimiter="\t")
        return 0

    def generate_balance(self,svm,g_num,dis_filter):
        dissum=0
        if self.meanflag==0:
            for feature in self.dataset.train_features:
                dissum+=abs((self.decision_distance(svm,feature)).detach().numpy())
            self.dismean=dissum/len(self.dataset.train_features)
        self.meanflag=1
        mindis=dis_filter*self.dismean
        z=torch.randn(int(50*g_num),self.G.z_dimension)
        if torch.cuda.is_available():
            z=z.cuda()
        fake_f=self.G(z).detach().numpy()
        deletelist=[]
        for i in range(len(fake_f)):
            if abs((self.decision_distance(svm,torch.tensor(fake_f[i]))).detach().numpy())<mindis:
                deletelist.append(i)
        fake_f=np.delete(fake_f,deletelist,axis=0)
        li=range(len(fake_f))
        chli=np.random.choice(li,int(g_num),replace=False)
        fake_f=fake_f[chli]
        label=svm.predict(fake_f).reshape(-1,1)
        fake_f=np.hstack((fake_f,label))
        save_in_train_all(fake_f,0)
        return 0

    def save(self):
        torch.save(self.G.state_dict(), './generator.pth')
        torch.save(self.D.state_dict(), './discriminator.pth')
        return 0