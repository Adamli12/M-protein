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
import main
import copy

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

#compute the cov matrix
"""np.set_printoptions(precision=3)
plt.imshow(np.cov(feature,rowvar=False))
plt.show()"""

#scaling
scaler=StandardScaler()
ufeature_sc=scaler.fit_transform(ufeature)#using large unlabeled data to normalize
feature_sc=scaler.transform(feature)
testfeature_sc=scaler.transform(testfeature)


#preparing
label=np.reshape(label,(-1,1))
labeleddata=np.hstack((feature_sc,label))
main.save_in_train_all(labeleddata,0,"data")
svm=SGDClassifier(max_iter=10000)

#begin training process step1
traindata=np.loadtxt("data/train/balancing.txt")
main.train_svm(svm,None,traindata)
dissum=0
for feature in ufeature_sc:
     dissum+=(main.decision_distance(svm,feature))
dismean=dissum/len(ufeature_sc)
gan1=GANmodel(ufeature_sc)
gan1.init_train()#用来生成generate

#begin training process step2
expert_batch_num=10
balancing_ratio=0.5
iter_num=10
for i in range(iter_num):
    gan1.G_train(svm)
    gan1.generate(expert_batch_num,scaler)#put data in generated.txt in expert folder
    main.generate_balance(ufeature_sc,svm,dismean,expert_batch_num*balancing_ratio,1,scaler)#put data in balancing.txt
    main.expert_label(expert_batch_num,(labeleddata.shape[1]-1),scaler)#put data in generated.txt
    traindatab=np.loadtxt("data/train/balancing.txt")
    traindatag=np.loadtxt("data/train/generated.txt")
    main.train_svm(svm,traindatab,traindatag)

#test
pred=svm.predict(scaler.transform(feature_sc))
print("test score:",svm.score(testfeature_sc,testlabel))


