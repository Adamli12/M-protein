import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt
import os,sys
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pickle
import utils
import copy
import argparse
from gmm import GMMmodel
from vae import VAEmodel
from bigan import BiGANmodel
from ae import AEmodel
from end import ENDmodel

parser = argparse.ArgumentParser(description='M-prtain main')
parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 4)')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train (default: 15)')
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

def active(folderpath, Gmodel, Cmodel):
    #data path
    testfpath = os.path.join(folderpath, "g100feature.csv")
    testlpath = os.path.join(folderpath, "g100label.csv")
    ufeaturepath = os.path.join(folderpath, "gufeature.csv")
    featurepath = os.path.join(folderpath, "r60feature.csv")
    labelpath = os.path.join(folderpath, "r60label.csv")
    feature = np.loadtxt(featurepath, delimiter = "\t")
    label = np.loadtxt(labelpath, delimiter = "\t")
    ufeature = np.loadtxt(ufeaturepath, delimiter = "\t")
    """tfeature = np.loadtxt(testfpath, delimiter = "\t")
    tlabel = np.loadtxt(testlpath, delimiter = "\t")"""
    feature, tfeature, label, tlabel = train_test_split(feature,label,test_size=0.33)


    if Cmodel == "end_to_end":
        Gscaler = MinMaxScaler()
        ufeature_sc = Gscaler.fit_transform(ufeature)#using large unlabeled data to normalize
        feature_sc = Gscaler.transform(feature)
        tfeature_sc = Gscaler.transform(tfeature)
        Gmo = ENDmodel(feature_sc, label, args)
        Gmo.train()
        Gmo.module.eval()
        with torch.no_grad():
            feature_sc = torch.tensor(feature_sc, dtype = torch.float32).to(device)
            recon, mu, _, _ = Gmo.module(feature_sc)
            print("train recon error")
            utils.recon_error(Gscaler.inverse_transform(feature_sc), Gscaler.inverse_transform(recon))
            tfeature_sc = torch.tensor(tfeature_sc, dtype = torch.float32).to(device)
            trecon, tmu, _, _ = Gmo.module(tfeature_sc)
            print("test recon error")
            utils.recon_error(Gscaler.inverse_transform(tfeature_sc), Gscaler.inverse_transform(trecon))
            ufeature_sc = torch.tensor(ufeature_sc, dtype = torch.float32).to(device)
            urecon, umu, _, _= Gmo.module(ufeature_sc)
            print("unlabeled recon error")
            utils.recon_error(Gscaler.inverse_transform(ufeature_sc), Gscaler.inverse_transform(urecon))
            _, _, _, ans = Gmo.module(feature_sc)
            ans=np.array(ans.detach())
            print("train classify acc", sum([label[i] == round(ans[i][0]) for i in range(len(label))])/len(ans))
            _, _, _, tans = Gmo.module(tfeature_sc)
            tans=np.array(tans.detach())
            t_acc=sum([tlabel[i] == round(tans[i][0]) for i in range(len(tlabel))])/len(tans)
            print("test classify acc", t_acc)
        return t_acc

    if Gmodel == "pca":
        Gmo = PCA(9)
        usvmfeature = Gmo.fit_transform(ufeature)
        svmfeature = Gmo.transform(feature)
        tsvmfeature = Gmo.transform(tfeature)
        recon=Gmo.inverse_transform(svmfeature)
        trecon=Gmo.inverse_transform(tsvmfeature)
        urecon=Gmo.inverse_transform(usvmfeature)
        print("train")
        utils.recon_error(feature,recon)

        #return utils.recon_error(feature,recon)

        print("test")
        utils.recon_error(tfeature,trecon)
        print("unlabel")
        utils.recon_error(ufeature,urecon)

    if Gmodel == "ae":
        Gscaler = MinMaxScaler()
        ufeature_sc = Gscaler.fit_transform(ufeature)#using large unlabeled data to normalize
        feature_sc = Gscaler.transform(feature)
        tfeature_sc = Gscaler.transform(tfeature)
        Gmo = AEmodel(ufeature_sc, args)
        Gmo.train()
        Gmo.module.eval()
        with torch.no_grad():
            feature_sc = torch.tensor(feature_sc, dtype = torch.float32).to(device)
            recon, mu = Gmo.module(feature_sc)
            print("train recon error")
            utils.recon_error(Gscaler.inverse_transform(feature_sc), Gscaler.inverse_transform(recon))

            #return utils.recon_error(Gscaler.inverse_transform(feature_sc), Gscaler.inverse_transform(recon))

            tfeature_sc = torch.tensor(tfeature_sc, dtype = torch.float32).to(device)
            trecon, tmu = Gmo.module(tfeature_sc)
            print("test recon error")
            utils.recon_error(Gscaler.inverse_transform(tfeature_sc), Gscaler.inverse_transform(trecon))
            ufeature_sc = torch.tensor(ufeature_sc, dtype = torch.float32).to(device)
            urecon, umu = Gmo.module(ufeature_sc)
            print("unlabeled recon error")
            utils.recon_error(Gscaler.inverse_transform(ufeature_sc), Gscaler.inverse_transform(urecon))
        svmfeature = mu
        tsvmfeature = tmu
        usvmfeature = umu

        """#看AE每个variable都表示什么
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
            cv2.destroyAllWindows()"""

    if Gmodel == "vae":
        Gscaler = MinMaxScaler()
        ufeature_sc = Gscaler.fit_transform(ufeature)#using large unlabeled data to normalize
        feature_sc = Gscaler.transform(feature)
        tfeature_sc = Gscaler.transform(tfeature)
        Gmo = VAEmodel(ufeature_sc, args)
        Gmo.train()
        Gmo.module.eval()
        with torch.no_grad():
            feature_sc = torch.tensor(feature_sc, dtype = torch.float32).to(device)
            recon, mu, _ = Gmo.module(feature_sc)
            print("train recon error")
            train_recon = utils.recon_error(Gscaler.inverse_transform(feature_sc), Gscaler.inverse_transform(recon))

            #return train_recon

            tfeature_sc = torch.tensor(tfeature_sc, dtype = torch.float32).to(device)
            trecon, tmu, _ = Gmo.module(tfeature_sc)
            print("test recon error")
            utils.recon_error(Gscaler.inverse_transform(tfeature_sc), Gscaler.inverse_transform(trecon))
            ufeature_sc = torch.tensor(ufeature_sc, dtype = torch.float32).to(device)
            urecon, umu, _ = Gmo.module(ufeature_sc)
            print("unlabeled recon error")
            utils.recon_error(Gscaler.inverse_transform(ufeature_sc), Gscaler.inverse_transform(urecon))
        svmfeature = mu
        tsvmfeature = tmu
        usvmfeature = umu

        """#看VAE每个variable都表示什么
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
            cv2.destroyAllWindows()"""

        
    elif Gmodel == "gmm":
        Gmo = GMMmodel(visualization=0)
        svmfeature = Gmo.module.encode(feature)
        recon=Gmo.module.decode(svmfeature)
        tsvmfeature = Gmo.module.encode(tfeature)
        trecon=Gmo.module.decode(tsvmfeature)
        print("train")
        utils.recon_error(feature,recon)

        #return utils.recon_error(feature,recon)

        print('test')
        utils.recon_error(tfeature,trecon)

        ans = Gmo.module.rule_classication(svmfeature)
        tans = Gmo.module.rule_classication(tsvmfeature)
        #print([label[i] == ans[i] for i in range(len(label))])
        acc = sum([label[i] == ans[i] for i in range(len(label))])/ans.shape[0]
        tacc = sum([tlabel[i] == tans[i] for i in range(len(tlabel))])/tans.shape[0]
        print("rule based train acc: ", acc)
        print("rule based test acc: ", tacc)
        ans = utils.derivative_filter(feature)
        tans = utils.derivative_filter(tfeature)
        mans = utils.mean_filter(feature)
        mtans = utils.mean_filter(tfeature)
        acc = sum([label[i] == mans[i] for i in range(len(label))])/len(mans)
        tacc = sum([tlabel[i] == mtans[i] for i in range(len(tlabel))])/len(mtans)
        print("mean rule based train acc: ", acc)
        print("mean rule based test acc: ", tacc)
        acc = sum([label[i] == ans[1][i] for i in range(len(label))])/len(ans[1])
        tacc = sum([tlabel[i] == tans[1][i] for i in range(len(tlabel))])/len(tans[1])
        print("2nd derivative rule based train acc: ", acc)
        print("2nd derivative rule based test acc: ", tacc)

    elif Gmodel == "bigan":
        pass

    #compute the cov matrix
    """np.set_printoptions(precision=3)
    plt.imshow(np.cov(svmfeature,rowvar=False))
    plt.show()"""

    #scaling
    svmscaler = StandardScaler()
    svmscaler.fit(svmfeature)#should be using large unlabeled data to normalize
    svmfeature_sc = svmscaler.transform(svmfeature)
    tsvmfeature_sc = svmscaler.transform(tsvmfeature)

    if Cmodel == "linear_svm":
        classifier = SVC(kernel="linear")
        classifier.fit(svmfeature_sc, label)
        print(classifier.coef_)
        print("train score: ", classifier.score(svmfeature_sc, label), "test score: ", classifier.score(tsvmfeature_sc, tlabel))
        return classifier.score(tsvmfeature_sc, tlabel)
    
    if Cmodel == "rbf_svm":
        classifier = SVC()
        classifier.fit(svmfeature_sc, label)
        print("train score: ", classifier.score(svmfeature_sc, label), "test score: ", classifier.score(tsvmfeature_sc, tlabel))
        return classifier.score(tsvmfeature_sc, tlabel)
    
    if Cmodel == "decision_tree":
        classifier = DecisionTreeClassifier()
        classifier.fit(svmfeature_sc, label)
        print("train score: ", classifier.score(svmfeature_sc, label), "test score: ", classifier.score(tsvmfeature_sc, tlabel))
        return classifier.score(tsvmfeature_sc, tlabel)

    if Cmodel == "random_forest":
        classifier = RandomForestClassifier()
        classifier.fit(X=svmfeature_sc, y=label)
        print("train score: ", classifier.score(svmfeature_sc, label), "test score: ", classifier.score(tsvmfeature_sc, tlabel))
        return classifier.score(tsvmfeature_sc, tlabel)

    """#preparing
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
    dissum=0
    for feature in ufeature_sc:
        dissum+=(utils.decision_distance(svm,feature))
    dismean=dissum/len(ufeature_sc)"""

    """#begin training process step2
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
    return svm"""
    return 0

test_acc=np.zeros(10)
for i in range(10):
    t=active("data", "vae", "linear_svm")
    print(t)
    test_acc[i]=t
std=np.sqrt(np.cov(test_acc,rowvar=False))
halfci=1.96*std/np.sqrt(10)
print("test_acc",np.mean(test_acc),"+-",halfci)
