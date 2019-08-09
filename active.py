import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture
import os,sys
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from generate import *
import pickle
from main import *
import copy

def save_in_train_all(dat,mode):
    if mode==1:
        np.savetxt("data/train/generated.txt",dat)
        gen_str=pickle.dumps(dat)
        f=open("data/all/generated.txt","a")
        f.write(gen_str+"\n")
        f.close()
    if mode==0:
        np.savetxt("data/train/balancing.txt",dat)
        gen_str=pickle.dumps(dat)
        f=open("data/all/balancing.txt","a")
        f.write(gen_str+"\n")
        f.close()
    return 0

def showpic(feature):
    weights=feature[:3]
    means=feature[3:6]
    covs=feature[6:9]
    img=toimage(weights,means,covs)
    cv2.imshow("generated",img)
    cv2.waitKey(0)
    return 0


#need to implement an SVM with incremental property and yield margin distance
#add residual feature
def train_svm(svm,labeleddata):
    feature_sc=labeleddata[:,:-1]
    label=labeleddata[:,-1]
    svm.fit(feature_sc,label)
    pred=svm.predict(scaler.transform(feature_sc))
    truelabel=pred==label
    for i in range(len(truelabel)):
        if truelabel[i]==0:
            print("picture",i//5,"column",i%5)
    print("the train dataset score",svm.score(feature_sc,label))
    return 0

def expert_label(g_num,feature_num,iter_num):
    path="data/expert/generated.txt"
    generated=np.loadtxt(path,delimiter="\t")
    trainadd=np.zeros((g_num,feature_num+1))
    i=0
    for feature in generated:
        showpic(feature)
        ans=input("please enter answer:(1 for abnormal, 0 for normal)")
        trainadd[i,:-1]=feature
        trainadd[i,-1]=ans
        i+=1
    save_in_train_all(trainadd,1)

#data path
unlabeledpath="ttrain.csv"
featurepath="ttrain.csv"
labelpath="tlabel.csv"
feature=np.loadtxt(featurepath,delimiter="\t")
label=np.loadtxt(labelpath,delimiter="\t")
ufeature=np.loadtxt(unlabeledpath,delimiter="\t")

#scaling
scaler=StandardScaler()
feature_sc=scaler.fit_transform(feature)
ufeature_sc=scaler.transform(ufeature)

#preparing
labeleddata=np.hstack(feature_sc,label)
save_in_train_all(labeleddata,0)
svm=SGDClassifier()


#begin training process step1
traindata=np.loadtxt("data/train/balancing.txt")
train_svm(svm,traindata)
gan1=GANmodel(ufeature_sc)
gan1.init_train()#用来生成generate
gan2=copy.deepcopy(gan1)#用来生成balancing



