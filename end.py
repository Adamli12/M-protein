
from __future__ import print_function
import argparse
import torch
from torch.utils import data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import cv2
import utils

parser = argparse.ArgumentParser(description='end to end M-protain')
parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 4)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 300)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--latent-variable-num', type=int, default=9, metavar='N',
                    help='how many latent variables')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

class Menddataset(data.Dataset):
    def __init__(self, dat, label, train=True):
        self.train = train
        if self.train:
            self.train_features=torch.from_numpy(dat)
            self.train_labels=torch.from_numpy(label)
        else:
            raise RuntimeError("Not implemented")
 
    def __getitem__(self, index):
        if self.train:
            feature= self.train_features[index]
            label= self.train_labels[index]
        else:
            raise RuntimeError("Not implemented")
        return feature, label
 
    def __len__(self):
        if self.train:
            return self.train_features.shape[0]
        else:
            raise RuntimeError("Not implemented")


class END(nn.Module):
    def __init__(self,ln):
        super(END, self).__init__()
        self.fc1 = nn.Linear(300, 150)
        self.fc21 = nn.Linear(150, ln)
        self.fc22 = nn.Linear(150, ln)
        self.fc3 = nn.Linear(ln, 150)
        self.fc4 = nn.Linear(150, 300)
        self.fc5 = nn.Linear(ln,20)
        self.fc6 = nn.Linear(20,1)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def classify(self, z):
        h4 = F.relu(self.fc5(z))
        return torch.sigmoid(self.fc6(h4))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 300))
        z = self.reparameterize(mu, logvar)
        ans = self.classify(mu)
        return self.decode(z), mu, logvar, ans

# Reconstruction + KL divergence losses summed over all elements and batch

class ENDmodel():
    def __init__(self, dat, label, args):
        self.dataset = Menddataset(dat,label)
        self.batch_size = args.batch_size
        self.dataloader = data.DataLoader(dataset=self.dataset, batch_size = self.batch_size, shuffle = True)
        self.epoch_num = args.epochs
        self.latent_num = args.latent_variable_num
        self.module=END(self.latent_num).to(device)
        self.optimizer = optim.Adam(self.module.parameters(), lr=1e-3)

    def loss_function(self,recon_x, x, mu, logvar, label, ans):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 300), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        CLA = 10*F.binary_cross_entropy(ans, label)
        return BCE + KLD + CLA

    def train(self):
        self.module.train()
        for epoch in range(1,self.epoch_num+1):
            train_loss = 0
            for batch_idx, data in enumerate(self.dataloader):
                label=torch.tensor(data[1], dtype=torch.float32).to(device)
                data=torch.tensor(data[0], dtype=torch.float32).to(device)
                self.optimizer.zero_grad()
                recon_batch, mu, logvar, ans = self.module(data)
                loss = self.loss_function(recon_batch, data, mu, logvar, label, ans)
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()
                if batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.dataloader.dataset),
                        100. * batch_idx / len(self.dataloader),
                        loss.item() / len(data)))

            print('====> Epoch: {} Average loss: {:.4f}'.format(
                epoch, train_loss / len(self.dataloader.dataset)))