import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture
import os,sys
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from generate import GANmodel
import pickle
from main import toimage
import copy

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
    svm.partial_fit(feature_sc,label,classes=np.unique(label))
    """pred=svm.predict(feature_sc)
    truelabel=pred==label
    for i in range(len(truelabel)):
        if truelabel[i]==0:
            print("picture",i//5,"column",i%5)"""
    print("the latest train dataset score",svm.score(feature_sc,label))
    return 0

def expert_label(g_num,feature_num):
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
    return 0

#data path
unlabeledpath="ttrain.csv"
featurepath="ttrain.csv"
labelpath="tlabel.csv"
testfeaturepath="ttrain.csv"
testlabelpath="tlabel.csv"
feature=np.loadtxt(featurepath,delimiter="\t")
label=np.loadtxt(labelpath,delimiter="\t")
ufeature=np.loadtxt(unlabeledpath,delimiter="\t")
testfeature=np.loadtxt(testfeaturepath,delimiter="\t")
testlabel=np.loadtxt(testlabelpath,delimiter="\t")

#scaling
scaler=StandardScaler()
ufeature_sc=scaler.fit_transform(ufeature)#using large unlabeled data to normalize
feature_sc=scaler.transform(feature)
testfeature_sc=scaler.transform(testfeature)

#preparing
label=np.reshape(label,(60,1))
labeleddata=np.hstack((feature_sc,label))
save_in_train_all(labeleddata,0)
svm=SGDClassifier()


#begin training process step1
traindata=np.loadtxt("data/train/balancing.txt")
train_svm(svm,traindata)
gan1=GANmodel(ufeature_sc)
gan1.init_train()#用来生成generate
gan2=copy.deepcopy(gan1)#用来生成balancing

#begin training process step2
expert_batch_num=10
balancing_ratio=0.5
iter_num=10
for i in range(iter_num):
    gan1.G_train(svm)
    gan1.generate(expert_batch_num)#put data in generated.txt in expert folder
    gan2.generate_balance(svm,expert_batch_num*balancing_ratio,0.5)#put data in balancing.txt
    expert_label(expert_batch_num,(labeleddata.shape[1]-1))#put data in generated.txt
    traindata1=np.loadtxt("data/train/balancing.txt")
    traindata2=np.loadtxt("data/train/generated.txt")
    traindata=np.vstack((traindata1,traindata2))
    train_svm(svm,traindata)

#test
pred=svm.predict(scaler.transform(feature_sc))
print("test score:",svm.score(testfeature_sc,testlabel))


