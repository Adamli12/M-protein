import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt
import os,sys
import utils
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture

def rulebasefinal(path,visualize=0,cut_n=6):
    t2=15
    t3=0.07
    n_components=3
    denses,_=utils.finddensefromcut(path,cut_n)
    maxd=[]
    for dense in denses[(cut_n-5):]:
        maxd.append(max(dense))
    lofd=len(denses[0])
    samples=list()
    for i in range((cut_n-5),cut_n):#sampling for BGM
        samples.append(np.array(utils.tosample(denses[i])).reshape(-1,1))
    allmeans=[]
    allcovs=[]
    allweights=[]
    BGM45=np.zeros((45))
    for i in range(5):
        BGM=BayesianGaussianMixture(n_components=n_components,covariance_type='spherical',weight_concentration_prior=0.000000000001,max_iter=500)
        BGM.fit(samples[i])
        means=np.reshape(BGM.means_,(-1,))
        permu=np.argsort(means)
        means=means[permu]
        BGM45[i*9+3:i*9+6]=means
        allmeans.append(means)
        covs=BGM.covariances_
        covs=covs[permu]
        BGM45[i*9+6:i*9+9]=covs
        allcovs.append(covs)
        weights=BGM.weights_
        weights=weights[permu]
        BGM45[i*9:i*9+3]=weights*len(samples[i])
        allweights.append(weights)
    if visualize==1:
        l=0
        for i in range(cut_n-5,cut_n):#visualization
            l+=1
            plt.subplot(2,n_components,l),plt.plot(denses[i])
            X=np.linspace(0,lofd,num=200,endpoint=False)
            Ys=utils.toGM(X,n_components,allmeans[l-1],allcovs[l-1],allweights[l-1])
            for j in range(n_components):
                #plt.subplot(1,5,l),plt.plot([allmeans[l-1][j],allmeans[l-1][j]],[0,255])
                plt.subplot(2,n_components,l),plt.plot(X,len(samples[l-1])*Ys[j])
                #plt.subplot(2,n_components,l),plt.plot(X,Ys[j])
                plt.ylim(0,255)
        plt.show()
    ans=np.zeros((12,))
    pre=np.zeros((5,n_components))
    _,delist=utils.derivative_filter(denses[(cut_n-5):])
    for i in range(5):###preprocessing the data to avoid peak overlapping(far overlap and near overlap) influence: identify far/near overlap cases and suppress far overlap peaks, amplify near overlap peaks
        ###如果很理想的情况应该能把两个far overlap的peak合并成一个在中间mean的，但是现在可以先直接把两个抑制掉，毕竟就不太可能是单克隆峰了。far overlap也就是两个峰实际上在图里面是同一个，BGM将其拆分从而更好的拟合高斯模型，我们这里将其抑制因为能够拆分为两个峰的基本上cov都比较大，不尖。
        for j in range(n_components):
            for l in range(n_components):
                if delist[i]==0:
                    pre[i][j]=1
                if j<l:
                    if allweights[i][j]/allweights[i][l]>3 or allweights[i][j]/allweights[i][l]<0.3333:#ignore when weight difference is too large
                        continue
                    if allcovs[i][j]/allweights[i][j]/allcovs[i][l]*allweights[i][l]/abs(allmeans[i][j]-allmeans[i][l])*utils.mean(np.sqrt(allcovs[i][j]),np.sqrt(allcovs[i][l]))>2 or allcovs[i][l]/allweights[i][l]/allcovs[i][j]*allweights[i][j]/abs(allmeans[i][j]-allmeans[i][l])*utils.mean(np.sqrt(allcovs[i][j]),np.sqrt(allcovs[i][l]))>2:#if the cov difference is large than it will be ignored from far overlap because there should be two peaks in the original density plot
                    #near overlap situation is when a sharp peak is on a mild one. it happens when monoclonal peak has a background polyclonal peak. here we amplify the sharp peaks' weight when their cov difference is large enough or their distance is close enough so that it will be detected as abnormal in the classification step
                        if abs(allmeans[i][j]-allmeans[i][l])<3.5*np.sqrt(max(allcovs[i][j],allcovs[i][l])):
                            neww=allweights[i][j]+allweights[i][l]
                            if allcovs[i][l]/allweights[i][l]/allcovs[i][j]*allweights[i][j]>1 and allweights[i][j]>0.15:
                                if allcovs[i][j]<400:
                                    allweights[i][j]=neww
                            else:
                                if allcovs[i][l]<400:
                                    allweights[i][l]=neww
                        continue
                    if allcovs[i][j]/allweights[i][j]/len(samples[i])<t3/2.5 or allcovs[i][l]/allweights[i][l]/len(samples[i])<t3/2.5:#if one of the considered peak has very small variance, then it should not be far overlap situation where the original peak is mild
                        continue
                    if allcovs[i][j]<70 or allcovs[i][l]<70:
                        continue
                    elif abs(allmeans[i][j]-allmeans[i][l])<3.5*np.sqrt(max(allcovs[i][j],allcovs[i][l])):#far overlap situation where there is only a mild peak in the original density plot, and GMM model break it down to two sharper peaks to fit the guassian curves more accurately. here we just suppress the peaks and thus we cannot determine the column is abnormal because of the two considered components
                        pre[i][j]=pre[i][l]=1    
    for i in [0,1,2]:
        for j in [3,4]:
            if maxd[i]<50 or maxd[j]<50:
                continue
            else:
                for k in range(len(allmeans[i])):
                    for l in range(len(allmeans[j])):
                        if pre[i][k]==1 or pre[j][l]==1:
                            continue
                        if abs(allmeans[i][k]-allmeans[j][l])>lofd/t2:
                            continue
                        else:
                            if allweights[i][k]<0.1 or allweights[j][l]<0.1:
                                continue
                            else:
                                if allcovs[i][k]/allweights[i][k]/len(samples[i])>t3 or allcovs[j][l]/allweights[j][l]/len(samples[j])>t3:###the t figure, represents the sharpness of the peak. just variance is not enough, we need to consider n_samples and weights too.
                                    continue
                                else:
                                    ans[i*2+j-2]=1 
                                    ans[7+i]=1
                                    ans[7+j]=1  
                                    ans[0]=1   
    for i in range(5):
        for j in range(n_components):
            if pre[i][j]==1:
                continue
            if maxd[i]<80:
                continue
            elif allweights[i][j]<0.05:
                continue
            if allcovs[i][j]/allweights[i][j]/len(samples[i])>t3:###t-figure
                continue
            else:
                ans[7+i]=1
                ans[0]=1
    return ans

ans=rulebasefinal("pics/trainpics/b.jpg",1,cut_n=6)
print(ans)