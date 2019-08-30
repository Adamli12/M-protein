
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

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 4)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

class Mvaedataset(data.Dataset):
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


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(300, 150)
        self.fc21 = nn.Linear(150, 9)
        self.fc22 = nn.Linear(150, 9)
        self.fc3 = nn.Linear(9, 150)
        self.fc4 = nn.Linear(150, 300)

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

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 300))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Reconstruction + KL divergence losses summed over all elements and batch

class VAEmodel():
    def __init__(self, dat,args):
        self.dataset = Mvaedataset(dat)
        self.batch_size = args.batch_size
        self.dataloader = data.DataLoader(dataset=self.dataset, batch_size = self.batch_size, shuffle = True)
        self.epoch_num = args.epochs
        self.module=VAE().to(device)
        self.optimizer = optim.Adam(self.module.parameters(), lr=1e-3)

    def loss_function(self,recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 300), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    def train(self):
        self.module.train()
        for epoch in range(1,self.epoch_num+1):
            train_loss = 0
            for batch_idx, data in enumerate(self.dataloader):
                data = data.to(device)
                data=torch.tensor(data, dtype=torch.float32)
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.module(data)
                loss = self.loss_function(recon_batch, data, mu, logvar)
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

    def svmtest(self,testdata,testlabel,svm,scaler):
        self.module.eval()
        with torch.no_grad():
            td=torch.tensor(testdata,dtype=torch.float32).to(device)
            recon_batch, mu, logvar = self.module(td)
        svmdat=mu.view(len(testdata),9).detach().numpy()
        svm.partial_fit(svmdat,testlabel,classes=[0,1])
        #svm.fit(svmdat,testlabel)
        print("the latest train dataset score",svm.score(svmdat,testlabel))
        for i in range(5):
            dat=np.reshape(np.array(scaler.inverse_transform(testdata[i*5].reshape(1,-1)),dtype=np.uint8),(-1,1))
            rec=np.reshape(np.array(scaler.inverse_transform(recon_batch[i*5].reshape(1,-1)),dtype=np.uint8),(-1,1))
            dimg=np.tile(dat,50,)
            drec=np.tile(rec,50)
            cv2.imshow("origin",dimg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imshow("reconstructed",drec)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    

"""def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))"""

if __name__ == "__main__":
    unlabeledpath="vaetrain.csv"
    featurepath="vaetrain.csv"
    labelpath="vaelabel.csv"
    testfeaturepath="vaetrain.csv"
    testlabelpath="vaelabel.csv"
    feature=np.loadtxt(featurepath,delimiter="\t")
    label=np.loadtxt(labelpath,delimiter="\t")
    ufeature=np.loadtxt(unlabeledpath,delimiter="\t")
    testfeature=np.loadtxt(testfeaturepath,delimiter="\t")
    testlabel=np.loadtxt(testlabelpath,delimiter="\t")

    scaler=MinMaxScaler()
    ufeature_sc=scaler.fit_transform(ufeature)#using large unlabeled data to normalize
    feature_sc=scaler.transform(feature)
    testfeature_sc=scaler.transform(testfeature)

    svm=SGDClassifier(max_iter=1000)
    #svm=SVC(kernel="linear")

    vamo=VAEmodel(ufeature_sc,args)
    vamo.train()
    print()
    vamo.svmtest(testfeature_sc,testlabel,svm,scaler)
        
    #test(epoch)
    """
    with torch.no_grad():
        sample = torch.randn(64, 9).to(device)
        sample = model.decode(sample).cpu()
    """