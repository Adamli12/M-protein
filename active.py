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
def train_svm(svm,labeleddata):
    feature_sc,label=labeleddata
    svm.fit(feature_sc,label)
    pred=svm.predict(scaler.transform(feature_sc))
    truelabel=pred==label
    for i in range(len(truelabel)):
        if truelabel[i]==0:
            print("picture",i//5,"column",i%5)
    print("the train dataset score",svm.score(feature_sc,label))
    return 0

def expert_label(gfeature_path,allfeature_path,glabel_path):
    


featurepath="ttrain.csv"
labelpath="tlabel.csv"
feature=np.loadtxt(featurepath,delimiter="\t")
label=np.loadtxt(labelpath,delimiter="\t")
scaler=StandardScaler()
feature_sc=scaler.fit_transform(feature)
labeleddata=(feature_sc,label)
svm=SGDClassifier()
train_svm(svm,labeleddata)

