# -*- coding: UTF-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture
def toGM(x,components,means,covs,weights):
    ys=np.zeros((components,len(x)))
    for i in range(components):
        ys[i]=weights[i]/np.sqrt(2*np.pi*covs[i])*np.exp(-(x-means[i])**2/(2*covs[i]))
    return ys

def toimage(weights,means,covs,n_samples):
    X=np.linspace(0,300,num=300,endpoint=False)
    Ys=toGM(X,len(means),means,covs,weights)
    for i in range(len(means)):
        plt.subplot(1,len(means),i),plt.plot(X,n_samples[i]*Ys[i])
        plt.show()
    return 0

toimage([0.2,0.2],[170,230],[80,200],2000)

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
    dense[150]=dense[150]+1###BGM will not be able to calculate 0 sample situation
    dense[151]=dense[151]+1
    dense[152]=dense[152]+1
    """plt.plot(dense)
    plt.show()"""
    return dense
"""
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
"""
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
"""
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
"""
def BGMreport(path):
    t2=15
    t3=0.07
    n_components=3
    denses,_=finddensefromcut(path)
    maxd=[]
    for dense in denses[1:]:
        maxd.append(max(dense))
    lofd=len(denses[0])
    samples=list()
    for i in range(1,6):#sampling for BGM
        samples.append(np.array(tosample(denses[i])).reshape(-1,1))
    allmeans=[]
    allcovs=[]
    allweights=[]
    for i in range(5):
        BGM=BayesianGaussianMixture(n_components=n_components,covariance_type='spherical',weight_concentration_prior=0.0000000000001,max_iter=500)
        BGM.fit(samples[i])
        means=np.reshape(BGM.means_,(-1,))
        allmeans.append(means)
        covs=BGM.covariances_
        allcovs.append(covs)
        weights=BGM.weights_
        allweights.append(weights)
    for i in range(5):#visualization
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
    pre=np.zeros((5,n_components))
    for i in range(5):###preprocessing the data to avoid peak overlapping(far overlap and near overlap) influence: identify far/near overlap cases and suppress far overlap peaks, amplify near overlap peaks
        ###如果很理想的情况应该能把两个far overlap的peak合并成一个在中间mean的，但是现在可以先直接把两个抑制掉，毕竟就不太可能是单克隆峰了。far overlap也就是两个峰实际上在图里面是同一个，BGM将其拆分从而更好的拟合高斯模型，我们这里将其抑制因为能够拆分为两个峰的基本上cov都比较大，不尖。
        for j in range(n_components):
            for l in range(n_components):
                if j<l:
                    if allweights[i][j]/allweights[i][l]>2.5 or allweights[i][j]/allweights[i][l]<0.4:#ignore when weight difference is too large
                        continue
                    if abs(allmeans[i][j]-allmeans[i][l])<lofd/t2:#near overlap situation is when a sharp peak is on a mild one. it happens when monoclonal peak has a background polyclonal peak. here we amplify both peaks' weights so that it will be detected as abnormal in the classification step
                        neww=allweights[i][j]+allweights[i][l]
                        allweights[i][j]=neww*2
                        allweights[i][l]=neww*2
                        continue
                    if allcovs[i][j]/allcovs[i][l]>2.5 or allcovs[i][j]/allcovs[i][l]<0.4:#if the cov difference is large than it will be ignored from far overlap because there should be two peaks in the original density plot
                        continue
                    if allcovs[i][j]<70 or allcovs[i][l]<70:#if one of the considered peak has very small variance, then it should not be far overlap situation where the original peak is mild
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
                                if allcovs[i][k]/allweights[i][k]/len(samples[i])>t3 or allcovs[j][l]/allweights[j][l]/len(samples[j])>t3:###the t figure, represents the sharpness of the peak. just variance is not enough, we need to consider n_samples and weights too. actually I am going to change variance to the square root of variance to make sure the dimension is right.
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
"""
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
"""
a=[1,1,0,0,0,0,0,1,0,0,1,0]
b=[1,0,0,1,0,0,0,0,1,0,1,0]
c=[1,0,0,0,0,1,0,0,0,1,1,0]
d=[1,0,1,0,0,0,0,1,0,0,0,1]
e=[1,0,0,0,1,0,0,0,1,0,0,1]
f=[1,0,0,0,0,0,1,0,0,1,0,1]
g=[1,0,0,0,0,0,0,0,0,0,0,1]
h=[1,0,0,0,0,0,0,0,0,0,1,0]
i=[1,1,0,0,0,0,0,1,0,0,1,0]
j=[1,0,1,0,0,0,0,1,0,0,0,1]
k=[1,1,1,0,0,0,0,1,0,0,1,1]
l=[1,1,0,1,0,1,0,1,1,1,1,0]
m=[1,0,1,0,0,0,0,1,0,0,0,1]
n=[1,0,0,0,1,0,0,0,1,0,0,1]
o=[1,1,1,1,0,0,0,1,1,0,1,1]
p=[1,0,0,0,1,0,1,0,1,1,0,1]
q=[1,0,0,0,0,0,0,0,1,0,1,1]
print(BGMreport("pics/l1.jpg")==l)