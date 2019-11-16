import argparse
import torch
from torch.utils import data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np

parser = argparse.ArgumentParser(description='AE M-protain')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 4)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 300)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--latent-variable-num', type=int, default=9, metavar='N',
                    help='how many latent variables')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

class MNISTaedataset(data.Dataset):
    def __init__(self, dat, train=True):
        self.train = train
        if self.train:
            self.train_features=dat
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


class CNNAE(nn.Module):
    def __init__(self,ln):
        super(CNNAE, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # batch_size, 16, 10, 10
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2) # b, 16, 5, 5
        )
        
        self.conv2=nn.Sequential(
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=1) # b, 8, 2, 2
        )

        self.deconv1=nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2, padding=0),  # b, 16, 5, 5
            nn.ReLU(),
        )
        
        self.deconv2=nn.Sequential(
            nn.ConvTranspose2d(16, 8, 3, stride=3, padding=0),  #b,8,15,15
        )

        self.deconv3=nn.Sequential(
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  #b,1,31,31
        )

        self.fc1 = nn.Linear(32, ln)
        self.fc2 = nn.Linear(ln, 32)

    def encode(self, x):
        h1=self.conv1(x)
        h2=self.conv2(h1)
        return F.relu(self.fc1(h2.view(h2.size(0),-1)))

    def decode(self, z):
        z=F.relu(self.fc2(z))
        z=z.view(z.size(0),8,2,2)
        h1=self.deconv1(z)
        #print(h1.shape)
        h2=self.deconv2(h1)
        #print(h2.shape)
        h3=self.deconv3(h2)
        #print(h3.shape)
        return torch.tanh(h3)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z),z

class CNNAEmodel():
    def __init__(self, dat, args=args):
        self.dataset = MNISTaedataset(dat)
        self.batch_size = args.batch_size
        self.dataloader = data.DataLoader(dataset=self.dataset, batch_size = self.batch_size, shuffle = True)
        self.epoch_num = args.epochs
        self.latent_num = args.latent_variable_num
        self.module=CNNAE(self.latent_num).to(device)
        self.optimizer = optim.Adam(self.module.parameters(), lr=1e-3,weight_decay=0)
        self.criterion = nn.MSELoss()

    def train(self):
        self.module.train()
        for epoch in range(1,self.epoch_num+1):
            train_loss = 0
            for batch_idx, data in enumerate(self.dataloader):
                data = data.to(device)
                data=data.float()
                data=data.view(data.size(0),1,data.size(1),data.size(2))
                self.optimizer.zero_grad()
                recon_batch, mu = self.module(data)
                loss = self.criterion(recon_batch, data)
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

    def decode(self,z):
        svmfeatures=torch.from_numpy(z).float()
        realpic=self.module.decode(svmfeatures).detach().numpy()
        return realpic

    def save(self,path="cnnae.pth"):
    #save
        torch.save(self.module.state_dict(), path)
        return 0

    def load(self,path="cnnae.pth"):
        self.module.load_state_dict(torch.load(path))