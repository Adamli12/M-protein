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

def toGM(x,components,means,covs,weights):
    ys=np.zeros((components,len(x)))
    for i in range(components):
        ys[i]=weights[i]/np.sqrt(2*np.pi*covs[i])*np.exp(-(x-means[i])**2/(2*covs[i]))
    return ys

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

def GMMreport(path):#maybe combine this with rulebased
    n_components=3
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
        GM=GaussianMixture(n_components=n_components,covariance_type='spherical')
        GM.fit(samples[i])
        means=GM.means_
        allmeans.append(means)
        covs=GM.covariances_
        allcovs.append(covs)
        weights=GM.weights_
        allweights.append(weights)
    return 0

def BGMreport(path):
    t1=130
    t2=15
    t3=0.6
    n_components=3
    denses,_=finddensefromcut(path)
    maxd=[]
    for dense in denses[1:]:
        maxd.append(max(dense))
    for i in range(len(denses)):
        if max(denses[i])>50:#做一个归一化，如果太小就根本不考虑
            denses[i]=denses[i]/max(denses[i])*255
    lofd=len(denses[0])
    samples=list()
    for i in range(1,6):
        samples.append(np.array(tosample(denses[i])).reshape(-1,1))
    allmeans=[]
    allcovs=[]
    allweights=[]
    for i in range(5):
        BGM=BayesianGaussianMixture(n_components=n_components,covariance_type='spherical',weight_concentration_prior=0.000000000001,max_iter=200)
        BGM.fit(samples[i])
        means=np.reshape(BGM.means_,(-1,))
        allmeans.append(means)
        covs=BGM.covariances_
        allcovs.append(covs)
        weights=BGM.weights_
        allweights.append(weights)
    for i in range(5):
        plt.subplot(2,n_components,i+1),plt.plot(denses[i+1])
        X=np.linspace(0,lofd,num=200,endpoint=False)
        Ys=toGM(X,n_components,allmeans[i],allcovs[i],allweights[i])
        for j in range(n_components):
            #plt.subplot(1,5,i+1),plt.plot([allmeans[i][j],allmeans[i][j]],[0,255])
            plt.subplot(2,n_components,i+1),plt.plot(X,len(samples[i])*Ys[j])
            #plt.subplot(2,n_components,i+1),plt.plot(X,Ys[j])
            plt.ylim(0,255)
    plt.show()
    ans=np.zeros((12,))
    for i in [0,1,2]:
        for j in [3,4]:
            if maxd[i]<50 or maxd[j]<50:
                continue
            else:
                for k in range(len(allmeans[i])):
                    for l in range(len(allmeans[j])):
                        if abs(allmeans[i][k]-allmeans[j][l])>lofd/t2:
                            continue
                        else:
                            if allweights[i][k]<0.1 or allweights[j][l]<0.1:
                                continue
                            else:
                                if allcovs[i][k]/maxd[i]>t3 or allcovs[j][l]/maxd[j]>t3:###将sample数量考虑进来，这样能够更准确的衡量峰的形状
                                    continue
                                else:
                                    ans[i*2+j-2]=1 
                                    ans[7+i]=1
                                    ans[7+j]=1  
                                    ans[0]=1   
    for i in range(5):
        for j in range(n_components):
            if maxd[i]<50:
                continue
            elif allweights[i][j]<0.05:
                continue
            elif allcovs[i][j]/maxd[i]>t3:###将sample数量考虑进来，这样能够更准确的衡量峰的形状
                continue
            else:
                ans[7+i]=1
                ans[0]=1
    return ans

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

print(BGMreport("pics/e1.jpg"))#依然对背景中的不行，高斯无法近似平台背景