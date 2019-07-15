# -*- coding: UTF-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture

def tosample(dense):
    n=len(dense)
    sample=[]
    for i in range(n):
        for j in range (int(dense[i])):
            sample.append(i)
    return sample


def todensity(img):
    #hi=img.shape[0]
    wi=img.shape[1]
    """cv2.imshow("d",img)
    cv2.waitKey(0)"""
    img = cv2.fastNlMeansDenoising(img,None,20,7,21)
    """plt.subplot(121),plt.imshow(img,cmap="gray")
    plt.subplot(122),plt.imshow(img1,cmap="gray")
    plt.show()"""
    img2 = img[:,int(wi*0.25):int(wi*0.75)]
    dense=255-np.mean(img2,axis=1)
    """plt.plot(dense)
    plt.show()"""
    return dense

def finddense(path):
    img = cv2.imread(path,0)
    hi=img.shape[0]
    wi=img.shape[1]
    horizontalmean=np.mean(img,axis=1)
    verticalmean=np.mean(img,axis=0)
    hmeso=np.argsort(verticalmean)
    vmeso=np.argsort(horizontalmean)
    vbounds=[]
    for i in vmeso:
        if i>hi/10 and i<hi-hi/10:
            if len(vbounds)==0:
                vbounds.append(i)
            elif abs(vbounds[0]-i)>hi/10:
                vbounds.append(i)
            if len(vbounds)==2:
                break
    vbounds.sort()
    hbounds=[]
    for i in hmeso:
        if i>wi/30 and i<wi-wi/30:
            if len(hbounds)==0:
                hbounds.append(i)
            elif min(abs(hbounds-i))>wi/50:
                hbounds.append(i)
            if len(hbounds)==12:
                break
    hbounds.sort()
    denses=[]
    for i in range(6):
        denses.append(todensity(img[vbounds[0]:vbounds[1],hbounds[i*2]:hbounds[i*2+1]]))
    return denses,wi

def finddensefromcut(path):
    img = cv2.imread(path,0)
    #hi=img.shape[0]
    wi=img.shape[1]
    hbounds=[]
    wiband=int(wi/6)
    flag=0
    mi=int(wi/20)
    for i in range(6):
        hbounds.append(flag+mi)
        flag+=wiband
        hbounds.append(flag-mi)
    denses=[]
    for i in range(6):
        denses.append(todensity(img[:,hbounds[i*2]:hbounds[i*2+1]]))
        #cv2.imshow("density",img[:,hbounds[i*2]:hbounds[i*2+1]])
        #cv2.waitKey(0)
    return denses,wi

def GMMreport(path):
    denses,_=finddensefromcut(path)
    samples=list()
    for i in range(1,6):
        samples.append(np.array(tosample(denses[i])).reshape(-1,1))
    no=[0,0,0,0,0]
    for i in range(len(samples)):
        if len(samples[i])<500:
            no[i]=1
    allmeans=[]
    allcovs=[]
    allweights=[]
    for i in range(5):
        GM=GaussianMixture(n_components=3,covariance_type='spherical')
        GM.fit(samples[i])
        means=GM.means_
        allmeans.append(means)
        covs=GM.covariances_
        allcovs.append(covs)
        weights=GM.weights_
        allweights.append(weights)
    return 0

def onepeakreport(path):
    t1=130
    t2=15
    i=0
    denses,_=finddensefromcut(path)
    tail=int(len(denses[0])/50)
    ders=[]
    ders1=[]
    for dense in denses:
        i+=1
        if max(dense)>50:
#做一个归一化，如果太小就根本不考虑
            dense=dense/max(dense)*255
        ders.append(cv2.Sobel(cv2.Sobel(dense,cv2.CV_64F,0,1,ksize=3),cv2.CV_64F,0,1,ksize=3))
#用二阶导来确定五个column有没有异常，一阶导来确定峰的位置，二阶导因为有双重性所以不能用来断定峰的位置
        ders1.append(cv2.Sobel(dense,cv2.CV_64F,0,1,ksize=3))
        #plt.subplot(2,3,i),plt.plot(ders1[i-1])
        plt.subplot(2,3,i),plt.plot(dense)
        plt.ylim(-255,255)
        plt.xlim(tail,len(dense)-tail)
    plt.show()
    peakpo=[]
    dermax=[]
    for der in ders[1:]:
        dernt=der[tail:len(dense)-tail]
        dermax.append((np.max(dernt)-np.min(dernt))/2)
    for der1 in ders1[1:]:
        dernt=der1[tail:len(dense)-tail]
        peakpo.append((np.argmax(dernt)+np.argmin(dernt))/2)
    abn2=[1 if x>t1 else 0 for x in dermax]
    abn1=[]
    pod=len(ders[0])/t2
    for i in [0,1,2]:
        for j in [3,4]:
            if abs(peakpo[i]-peakpo[j])<pod and dermax[i]>t1 and dermax[j]>t1:
                abn1.append(1)
            else:
                abn1.append(0)
    if max(abn2)==1:
        abn3=1
    else:
        abn3=0
    abn=[abn3]+abn1+abn2
    return abn

print(GMMreport("l1.jpg"))#在i1时就是背景有些太大了，结果阈值不够了，需要尖角识别器，可以运行来观看1阶导有尖角但是二阶导不大