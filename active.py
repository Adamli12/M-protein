import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture
import os,sys
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from generate import *

#need to implement an SVM with incremental property and yield margin distance
#add residual feature
def init_train_G(gan,realdata):
    

def init_train_svm(svm,labeleddata):

def train_G(svm,gan):

def expert_label(gtrain_path,alltrain_path,glabel_path):



trainpath="ttrain.csv"
labelpath="tlabel.csv"
train=np.loadtxt(trainpath,delimiter="\t")
label=np.loadtxt(labelpath,delimiter="\t")
scaler=StandardScaler()
train_sc=scaler.fit_transform(train)
svm=SGDClassifier()
svm.fit(train_sc,label)
pred=svm.predict(scaler.transform(train))
truelabel=pred==label
for i in range(len(truelabel)):
    if truelabel[i]==0:
        print("picture",i//5,"column",i%5)
print(svm.score(train,label))

