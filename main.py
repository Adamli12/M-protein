# -*- coding: UTF-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture
import os,sys
import pickle

def decision_distance(svm,x):
        w=svm.coef_
        b=svm.intercept_
        dis=abs(sum(np.matmul(w,x))+b)
        dis=dis/np.linalg.norm(w)
        return dis

def showpic(feature):
    weights=feature[:3]
    means=feature[3:6]
    covs=feature[6:9]
    img=toimage(weights,means,covs)
    cv2.imshow("generated",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0

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

def generate_balance(ufeature_sc,svm,dismean,g_num,dis_filter,scaler):
    mindis=dis_filter*dismean
    deletelist=[]
    for i in range(len(ufeature_sc)):
        #print(self.decision_distance(svm,torch.tensor(ufeature_sc[i])))
        if (decision_distance(svm,ufeature_sc[i]))<mindis:
            deletelist.append(i)
    chosenf=np.delete(ufeature_sc,deletelist,axis=0)
    li=range(len(chosenf))
    chli=np.random.choice(li,int(g_num),replace=False)
    chosenf=chosenf[chli]
    bfsc_chosen_f=scaler.inverse_transform(chosenf)
    label=svm.predict(chosenf).reshape(-1,1)
    chosenf=np.hstack((chosenf,label))
    for feature in bfsc_chosen_f:
        for i in range(len(feature)):
            if i>5 and feature[i]<0:
                feature[i-6]=1
                feature[i]=1000
                continue
            if feature[i]<0:
                feature[i]=0
        showpic(feature)

    save_in_train_all(chosenf,0)
    return 0

#add residual feature
def train_svm(svm,traindatab,traindatag):
    if type(traindatab)==type(None):
        labeleddata=traindatag
    else:
        labeleddata=np.vstack((traindatab,traindatag))
    feature_sc=labeleddata[:,:-1]
    label=labeleddata[:,-1]
    svm.partial_fit(feature_sc,label,classes=[0,1])
    """pred=svm.predict(feature_sc)
    truelabel=pred==label
    for i in range(len(truelabel)):
        if truelabel[i]==0:
            print("picture",i//5,"column",i%5)"""
    testf=traindatag[:,:-1]
    testl=traindatag[:,-1]
    print("the latest train dataset score",svm.score(testf,testl))
    return 0

def expert_label(g_num,feature_num,scaler):#为什么一直会生成负的cov啊啊啊啊啊啊！！！
    path="data/expert/generated.txt"
    generated=np.loadtxt(path,delimiter="\t")
    trainadd=np.zeros((g_num,feature_num+1))
    trainadd[:,:feature_num]=generated
    generated=scaler.inverse_transform(generated)
    j=0
    for feature in generated:
        for i in range(len(feature)):
            if i<3 and feature[i]<0:
                feature[i]=0
        showpic(feature)
        ans=input("please enter answer:(1 for abnormal, 0 for normal)")
        trainadd[j,-1]=ans
        j+=1
    save_in_train_all(trainadd,1)
    return 0

def mean(a,b):
    return (a+b)/2

def mean_generator():
    return 180+40*np.random.randn(3)

def weight_generator():
    a=np.random.rand(2)
    a.sort()
    b=a[0]
    c=a[1]-a[0]
    d=1-a[1]
    return[b,c,d]

def cov_generatorn():
    a=700+400*np.random.randn(3)
    for i in range(3):
        a[i]=max(a[i],600)
    return a

def cov_generatorgk():
    a=100+30*np.random.randn(3)
    for i in range(3):
        a[i]=max(a[i],10)
    return a

def sample_generator():
    return int(11000+50*np.random.randn())

def toGM(x,components,means,covs,weights):
    ys=np.zeros((components,len(x)))
    for i in range(components):
        ys[i]=weights[i]/np.sqrt(2*np.pi*covs[i])*np.exp(-(x-means[i])**2/(2*covs[i]))
    return ys

def toimage(weightsn,means,covs):#weightsn is how much sample that one component got
    X=np.linspace(0,300,num=300,endpoint=False)
    Ys=toGM(X,len(means),means,covs,weightsn)
    dense=np.sum(Ys,axis=0)
    """
    for i in range(len(means)):
        plt.plot(X,Ys[i])
    plt.plot(X,dense)
    plt.show()
    """
    denseno=np.reshape(np.uint8(255-np.clip(dense,0,220)),(300,1))
    img=np.tile(denseno,50)
    #cv2.imshow("generated",img)
    #cv2.waitKey(0)
    return img

def showgeneratedimg(imgs):
    hi,wi=np.shape(imgs[0])
    interval=np.uint8(np.zeros((hi,int(wi/6))))+255
    gimg=np.hstack((imgs[0],interval,imgs[1],interval,imgs[2],interval,imgs[3],interval,imgs[4]))
    rgbgimg=np.zeros((gimg.shape[0],gimg.shape[1],3))
    rgbgimg=rgbgimg+255
    rgbgimg[:,:,1]+=gimg
    rgbgimg[:,:,2]+=gimg
    rgbgimg=np.uint8(rgbgimg)
    #cv2.imshow("gimg",rgbgimg)
    #cv2.waitKey(0)
    return rgbgimg

def tosample(dense):
    n=len(dense)
    sample=[]
    for i in range(n):
        for j in range (int(dense[i])):
            sample.append(i)
    return sample

def todensity(img):
    wi=img.shape[1]
    img=cv2.resize(img,(wi,300))
    img = cv2.fastNlMeansDenoising(img,None,20,7,21)
    #plt.imshow(img,cmap="gray")
    #plt.show()
    img2 = img[:,int(wi*0.25):int(wi*0.75)]
    dense=255-np.mean(img2,axis=1)
    dense[150]=dense[150]+1###BGM will not be able to calculate 0 sample situation
    dense[151]=dense[151]+1
    dense[152]=dense[152]+1
    #plt.plot(dense)
    #plt.show()
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
def finddensefromcut(path,cut_n):
    img = cv2.imread(path,0)
    #hi=img.shape[0]
    wi=img.shape[1]
    hbounds=[]
    wiband=int(wi/cut_n)
    flag=0
    mi=int(wi/20)
    for i in range(cut_n):
        hbounds.append(flag+mi)
        flag+=wiband
        hbounds.append(flag-mi)
    denses=[]
    for i in range(cut_n):
        denses.append(todensity(img[:,hbounds[i*2]:hbounds[i*2+1]]))
        #cv2.imshow("density",img[:,hbounds[i*2]:hbounds[i*2+1]])
        #cv2.waitKey(0)
    return denses,wi

def BGMreport(path,visualize=1,cut_n=6):
    t2=15
    t3=0.07
    n_components=3
    denses,_=finddensefromcut(path,cut_n)
    maxd=[]
    for dense in denses[(cut_n-5):]:
        maxd.append(max(dense))
    lofd=len(denses[0])
    samples=list()
    for i in range((cut_n-5),cut_n):#sampling for BGM
        samples.append(np.array(tosample(denses[i])).reshape(-1,1))
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
            Ys=toGM(X,n_components,allmeans[l-1],allcovs[l-1],allweights[l-1])
            for j in range(n_components):
                #plt.subplot(1,5,l),plt.plot([allmeans[l-1][j],allmeans[l-1][j]],[0,255])
                plt.subplot(2,n_components,l),plt.plot(X,len(samples[l-1])*Ys[j])
                #plt.subplot(2,n_components,l),plt.plot(X,Ys[j])
                plt.ylim(0,255)
        plt.show()
    ans=np.zeros((12,))
    pre=np.zeros((5,n_components))
    for i in range(5):###preprocessing the data to avoid peak overlapping(far overlap and near overlap) influence: identify far/near overlap cases and suppress far overlap peaks, amplify near overlap peaks
        ###如果很理想的情况应该能把两个far overlap的peak合并成一个在中间mean的，但是现在可以先直接把两个抑制掉，毕竟就不太可能是单克隆峰了。far overlap也就是两个峰实际上在图里面是同一个，BGM将其拆分从而更好的拟合高斯模型，我们这里将其抑制因为能够拆分为两个峰的基本上cov都比较大，不尖。
        for j in range(n_components):
            for l in range(n_components):
                if j<l:
                    if allweights[i][j]/allweights[i][l]>3 or allweights[i][j]/allweights[i][l]<0.3333:#ignore when weight difference is too large
                        continue
                    if allcovs[i][j]/allweights[i][j]/allcovs[i][l]*allweights[i][l]/abs(allmeans[i][j]-allmeans[i][l])*mean(np.sqrt(allcovs[i][j]),np.sqrt(allcovs[i][l]))>2 or allcovs[i][l]/allweights[i][l]/allcovs[i][j]*allweights[i][j]/abs(allmeans[i][j]-allmeans[i][l])*mean(np.sqrt(allcovs[i][j]),np.sqrt(allcovs[i][l]))>2:#if the cov difference is large than it will be ignored from far overlap because there should be two peaks in the original density plot
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
    return ans,BGM45
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
def read_label(path,ans):
    label=np.loadtxt(path,delimiter="\t")
    wr=[]
    for i in range(len(label)):
        for j in range(5):
            if ans[j]!=label[i][-5:][j]:
                wr.append(i)
                break
    return wr
    
def classify_folder(path,pre,gt=None,testflag=0,cut_n=6,numsort=0):
    i=0
    if numsort==0:
        #train=[]
        label=[]
    else:
        #train=np.zeros((len(os.listdir(path)),45))
        label=np.zeros((len(os.listdir(path)),12))
    for img in os.listdir(path):
        if numsort==1:
            nu=int(img[:-4])
        path1=os.path.join(path,img)
        ans=BGMreport(path1,0,cut_n)
        if testflag==1:
            if numsort==0:
                label.append(ans[0]==gt[i])
            else:
                label[nu]=(ans[0]==gt[i])
        else:
            if numsort==0:
                label.append(ans[0])
            else:
                label[nu]=(ans[0])
        i+=1
        if i%20==0:
            print(i)
    #np.savetxt(pre+"components.csv",train,delimiter="\t",fmt="%.4f")
    np.savetxt(pre+"labels.csv",label,delimiter="\t",fmt="%d")
    return 0

def folder_to_data(path,pre,cut_n=6):
    i=0
    train=np.zeros((len(os.listdir(path)*5),9))
    label=np.zeros((len(os.listdir(path)*5)))
    for img in os.listdir(path):
        path1=os.path.join(path,img)
        ans=BGMreport(path1,0,cut_n)
        for j in range(5):
            train[i*5+j]=ans[1][j*9:j*9+9]
            label[i*5+j]=ans[0][7+j]
        i+=1
        if i%20==0:
            print(i)
    np.savetxt(pre+"train.csv",train,delimiter="\t",fmt="%.4f")
    np.savetxt(pre+"label.csv",label,delimiter="\t",fmt="%d")
    return 0

def folder_to_vae_data(path,pre,cut_n=6):
    i=0
    train=np.zeros((len(os.listdir(path)*5),300))
    label=np.zeros((len(os.listdir(path)*5)))
    for img in os.listdir(path):
        path1=os.path.join(path,img)
        ans=finddensefromcut(path1,cut_n)[0]
        ans1=BGMreport(path1,0,cut_n)
        for j in range(5):
            train[i*5+j]=ans[j]
            label[i*5+j]=ans1[0][7+j]
        i+=1
        if i%20==0:
            print(i)
    np.savetxt(pre+"train.csv",train,delimiter="\t",fmt="%.4f")
    np.savetxt(pre+"label.csv",label,delimiter="\t",fmt="%d")
    return 0

def generate_pics(pathgk,pathno,num):
    for i in range(num):
        G=toimage(np.array(weight_generator())*sample_generator(),mean_generator(),cov_generatorgk())
        A=toimage(np.array(weight_generator())*sample_generator(),mean_generator(),cov_generatorn())
        M=toimage(np.array(weight_generator())*sample_generator(),mean_generator(),cov_generatorn())
        K=toimage(np.array(weight_generator())*sample_generator(),mean_generator(),cov_generatorgk())
        L=toimage(np.array(weight_generator())*sample_generator(),mean_generator(),cov_generatorn())
        img=showgeneratedimg([G,A,M,K,L])
        cv2.imwrite(pathgk+"/"+str(i)+".jpg",img)

    for i in range(num):
        G=toimage(np.array(weight_generator())*sample_generator(),mean_generator(),cov_generatorn())
        A=toimage(np.array(weight_generator())*sample_generator(),mean_generator(),cov_generatorn())
        M=toimage(np.array(weight_generator())*sample_generator(),mean_generator(),cov_generatorn())
        K=toimage(np.array(weight_generator())*sample_generator(),mean_generator(),cov_generatorn())
        L=toimage(np.array(weight_generator())*sample_generator(),mean_generator(),cov_generatorn())
        img=showgeneratedimg([G,A,M,K,L])
        cv2.imwrite(pathno+"/"+str(i)+".jpg",img)
    return 0

gt=[[1,1,0,0,0,0,0,1,0,0,1,0],
[1,0,0,1,0,0,0,0,1,0,1,0],
[1,0,0,0,0,1,0,0,0,1,1,0],
[1,0,1,0,0,0,0,1,0,0,0,1],
[1,0,0,0,1,0,0,0,1,0,0,1],
[1,0,0,0,0,0,1,0,0,1,0,1],
[1,0,0,0,0,0,0,0,0,0,0,1],
[1,0,0,0,0,0,0,0,0,0,1,0],
[1,1,0,0,0,0,0,1,0,0,1,0],
[1,0,1,0,0,0,0,1,0,0,0,1],
[1,1,1,0,0,0,0,1,0,0,1,1],
[1,1,0,1,0,1,0,1,1,1,1,0],
[1,0,1,0,0,0,0,1,0,0,0,1],
[1,0,0,0,1,0,0,0,1,0,0,1],
[1,1,1,1,0,0,0,1,1,0,1,1],
[1,0,0,0,1,0,1,0,1,1,0,1],
[1,0,0,0,0,0,0,0,1,0,1,1]]

"""
generate_pics("generate_gkpics","generate_nopics",100)
folder_to_data("generate_gkpics","gk",5,)
folder_to_data("generate_nopics","no",5)
print(read_label("nolabels.csv",[0,0,0,0,0]))
"""
if __name__ == "__main__":
    folder_to_vae_data("pics/trainpics","vae",6)

"""
ans=BGMreport("pics/trainpics/b.jpg",1,cut_n=6)
print(ans[0])
#print(ans[1])

classify_folder("pics/trainpics","train",gt=gt,testflag=1,cut_n=6,numsort=0)"""
