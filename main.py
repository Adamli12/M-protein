import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt
import os,sys
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pickle
import utils
import copy
import argparse
from gmm import GMMmodel
from vae import VAEmodel
from bigan import BiGANmodel

parser = argparse.ArgumentParser(description='M-prtain main')
parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 4)')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
                    help='number of epochs to train (default: 300)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--latent-variable-num', type=int, default=9, metavar='N',
                    help='how many latent variables for vae')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

def generate_balance(folderpath,svm,g_num,dis_filter,svmscaler,Gscaler):
    mindis=dis_filter*dismean
    deletelist=[]
    for i in range(len(ufeature_sc)):
        #print(self.decision_distance(svm,torch.tensor(ufeature_sc[i])))
        if (utils.decision_distance(svm,ufeature_sc[i]))<mindis:
            deletelist.append(i)
    chosenf=np.delete(ufeature_sc,deletelist,axis=0)
    li=range(len(chosenf))
    chli=np.random.choice(li,int(g_num),replace=False)
    chosenf=chosenf[chli]
    bfsc_chosen_f=scaler.inverse_transform(chosenf)
    label=svm.predict(chosenf).reshape(-1,1)
    chosenf=np.hstack((chosenf,label))
    for feature in bfsc_chosen_f:
        for i in range(len(feature)):
            if i>5 and feature[i]<0:
                feature[i-6]=1
                feature[i]=1000
                continue
            if feature[i]<0:
                feature[i]=0
        utils.showpic(feature)

    utils.save_in_train_all(chosenf,0,"data")
    return 0

#add residual feature
def train_svm(svm,traindatab,traindatag):
    if type(traindatab)==type(None):
        labeleddata=traindatag
    else:
        labeleddata=np.vstack((traindatab,traindatag))
    feature_sc=labeleddata[:,:-1]
    label=labeleddata[:,-1]
    svm.partial_fit(feature_sc,label,classes=[0,1])
    """pred=svm.predict(feature_sc)
    truelabel=pred==label
    for i in range(len(truelabel)):
        if truelabel[i]==0:
            print("picture",i//5,"column",i%5)"""
    testf=feature_sc
    testl=label
    print("the latest train dataset score",svm.score(testf,testl))
    return 0

def expert_label(folderpath,g_num,feature_num,svmscaler,Gscaler,model):
    path=folderpath + "/expert/generated.txt"
    generated=np.loadtxt(path,delimiter="\t")
    trainadd=np.zeros((g_num,feature_num+1))
    trainadd[:,:feature_num]=generated
    generated=svmscaler.inverse_transform(generated)
    denses=model.module.decode(generated)
    denses=np.array(Gscaler.inverse_transform(denses),dtype="uint8")
    j=0
    for dense in denses:
        dimg=np.tile(np.reshape(dense,(-1,1)),50)
        cv2.imshow("generated",dimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        ans=input("please enter answer:(1 for abnormal, 0 for normal)")
        trainadd[j,-1]=ans
        j+=1
    utils.save_in_train_all(trainadd,1,folderpath)
    return 0

def sample(path, svm, expert_batch_num):
    samp=1
    return samp

def active(folderpath, Gmodel):
    #data path
    featurepath = os.path.join(folderpath, "feature.csv")
    labelpath = os.path.join(folderpath, "label.csv")
    ufeaturepath = os.path.join(folderpath, "feature.csv")
    feature = np.loadtxt(featurepath, delimiter = "\t")
    label = np.loadtxt(labelpath, delimiter = "\t")
    ufeature = np.loadtxt(ufeaturepath, delimiter = "\t")

    if Gmodel == "vae":
        Gscaler = MinMaxScaler()
        ufeature_sc = Gscaler.fit_transform(ufeature)#using large unlabeled data to normalize
        feature_sc = Gscaler.transform(feature)
        Gmo = VAEmodel(ufeature_sc, args)
        Gmo.train()
        Gmo.module.eval()
        with torch.no_grad():
            feature_sc = torch.tensor(feature_sc, dtype = torch.float32).to(device)
            recon, mu, _ = Gmo.module(feature_sc)
            utils.recon_error(Gscaler.inverse_transform(feature_sc), Gscaler.inverse_transform(recon))
        svmfeature = mu

#看VAE每个variable都表示什么
        d=20
        a=mu[d]
        c=np.array(Gscaler.inverse_transform(feature_sc)[d],dtype=np.uint8)
        b=5
        delta=np.ones(a.shape)
        img=np.tile(np.reshape(c,(-1,1)),50)
        cv2.imshow("i",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        for i in range(10):
            delta[b]+=2*i
            with torch.no_grad():
                tes=torch.tensor(np.multiply(a,delta),dtype=torch.float32).to(device)
                img=np.tile(np.reshape(Gmo.module.decode(tes).detach().numpy(),(-1,1)),50)
            cv2.imshow("i",img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        
    elif Gmodel == "gmm":
        Gmo = GMMmodel(visualization=0)
        svmfeature = Gmo.module.encode(feature)
        recon=Gmo.module.decode(svmfeature)
        utils.recon_error(feature,recon)
        ans = Gmo.module.rule_classication(svmfeature)
        #print([label[i] == ans[i] for i in range(len(label))])
        acc = sum([label[i] == ans[i] for i in range(len(label))])/ans.shape[0]
        print("rule based acc: ", acc)
    elif Gmodel == "bigan":
        pass

    #compute the cov matrix
    """np.set_printoptions(precision=3)
    plt.imshow(np.cov(svmfeature,rowvar=False))
    plt.show()"""

    #scaling
    svmscaler = StandardScaler()
    svmfeature_sc = svmscaler.fit_transform(svmfeature)#using large unlabeled data to normalize

    #preparing
    label = np.reshape(label, (-1,1))
    labeleddata = np.hstack((svmfeature_sc, label))
    utils.save_in_train_all(labeleddata, 0, os.path.join(folderpath, Gmodel))
    svm = SGDClassifier(max_iter = 10000)

    #begin training process step1
    traindata = np.loadtxt(os.path.join(folderpath, Gmodel) + "/train/balancing.txt")

    feature_sc=traindata[:,:-1]
    label=traindata[:,-1]
    svm.partial_fit(feature_sc,label,classes=[0,1])
    print("coef", svm.coef_)
    pred=svm.predict(feature_sc)
    truelabel=pred==label
    for i in range(len(truelabel)):
        if truelabel[i]==0:
            print("data(from 0)",i,feature_sc[i])
            fea=svmscaler.inverse_transform(feature_sc[i])
            image=utils.toimage(fea[0:3],fea[3:6],fea[6:9])
            cv2.imshow("generated",image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    testf=feature_sc
    testl=label
    print("the latest train dataset score",svm.score(testf,testl))

    train_svm(svm,None,traindata)
    """dissum=0
    for feature in ufeature_sc:
        dissum+=(utils.decision_distance(svm,feature))
    dismean=dissum/len(ufeature_sc)"""

    #begin training process step2
    expert_batch_num=10
    balancing_ratio=0.5
    iter_num=10
    for i in range(iter_num):
        samp = sample(os.path.join(folderpath, Gmodel), svm, expert_batch_num)#put data in generated.txt in expert folder
        generate_balance(os.path.join(folderpath, Gmodel), svm, expert_batch_num*balancing_ratio, 1, svmscaler, Gscaler)#put data in balancing.txt
        expert_label(os.path.join(folderpath, Gmodel), expert_batch_num, (labeleddata.shape[1]-1), svmscaler, Gscaler, Gmo)#put data in generated.txt
        traindatab=np.loadtxt(os.path.join(folderpath, Gmodel) + "train/balancing.txt")
        traindatag=np.loadtxt(os.path.join(folderpath, Gmodel) + "train/generated.txt")
        train_svm(svm,traindatab,traindatag)

    #test
    print("train score:",svm.score(feature_sc,label))
    return svm

active("data", "vae")