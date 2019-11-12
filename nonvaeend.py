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


class nEND(nn.Module):
    def __init__(self):
        super(nEND, self).__init__()
        self.conv1 = nn.Conv1d(1,4,3,stride=1,padding=1)
        self.conv2 = nn.Conv1d(4,4,5,stride=1,padding=2,groups=4)
        self.fc1 = nn.Linear(1200,256)
        self.fc2 = nn.Linear(256,32)
        self.fc3 = nn.Linear(32,1)

    def forward(self, x):
        x=F.relu(self.conv1(x.view(x.size()[0],1,-1)))
        x=F.relu(self.conv2(x))
        x=x.view(x.size()[0], -1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return F.sigmoid(x)

# Reconstruction + KL divergence losses summed over all elements and batch

class nENDmodel():
    def __init__(self, dat, label, args):
        self.dataset = Menddataset(dat,label)
        self.batch_size = args.batch_size
        self.dataloader = data.DataLoader(dataset=self.dataset, batch_size = self.batch_size, shuffle = True)
        self.epoch_num = args.epochs
        self.module=nEND().to(device)
        self.optimizer = optim.Adam(self.module.parameters(), lr=1e-3)

    def loss_function(self, label, ans):
        return F.binary_cross_entropy(ans, label)

    def train(self):
        self.module.train()
        for epoch in range(1,self.epoch_num+1):
            train_loss = 0
            for batch_idx, data in enumerate(self.dataloader):
                label=torch.tensor(data[1], dtype=torch.float32).to(device)
                data=torch.tensor(data[0], dtype=torch.float32).to(device)
                self.optimizer.zero_grad()
                ans = self.module(data)
                loss = self.loss_function(label, ans)
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