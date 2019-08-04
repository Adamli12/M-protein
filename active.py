import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture
import os,sys
from sklearn.svm import SVC

#need to implement an SVM with incremental property and yield margin distance
#add residual feature
trainpath="ttrain.csv"
labelpath="tlabel.csv"
train=np.loadtxt(trainpath,delimiter="\t")
label=np.loadtxt(labelpath,delimiter="\t")
svm=SVC()
svm.fit(train,label)
print(svm.predict(train))


