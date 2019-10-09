import numpy as np
import utils
from sklearn.mixture import BayesianGaussianMixture
from matplotlib import pyplot as plt

class GMM():
    def __init__(self, visualization=0, lofd=300, n_components=3, t3=0.07):
        self.visualization=visualization
        self.lofd=lofd
        self.n_components=n_components
        self.t3=t3

    def encode(self,x):
        samples=list()
        for i in range(x.shape[0]):#sampling for BGM
            samples.append(np.array(utils.tosample(x[i])).reshape(-1,1))
        allmeans=[]
        allcovs=[]
        allweights=[]
        BGM45=np.zeros((x.shape[0],3*self.n_components))
        for i in range(x.shape[0]):
            #BGM=BayesianGaussianMixture(n_components=self.n_components,covariance_type='spherical',weight_concentration_prior=1e-10,max_iter=5000,tol=1e-7,n_init=5)
            BGM=BayesianGaussianMixture(n_components=self.n_components,covariance_type='spherical',weight_concentration_prior=1e-10,max_iter=500)
            BGM.fit(samples[i])
            means=np.reshape(BGM.means_,(-1,))
            permu=np.argsort(means)
            means=means[permu]
            BGM45[i][self.n_components:2*self.n_components]=means
            covs=BGM.covariances_
            covs=covs[permu]
            BGM45[i][2*self.n_components:3*self.n_components]=covs
            weights=BGM.weights_
            weights=weights[permu]
            BGM45[i][0:self.n_components]=weights*len(samples[i])
            if self.visualization==1:
                plt.plot(x[i])
                X=np.linspace(0,self.lofd,num=200,endpoint=False)
                Ys=utils.toGM(X,self.n_components,BGM45[i][self.n_components:2*self.n_components],BGM45[i][2*self.n_components:3*self.n_components],BGM45[i][0:self.n_components])
                for j in range(self.n_components):
                    plt.plot(X,Ys[j])
                    plt.ylim(0,255)
                plt.show()
        return BGM45

    def decode(self,z):
        means=z[:,self.n_components:2*self.n_components]
        covs=z[:,2*self.n_components:3*self.n_components]
        weights=z[:,0:self.n_components]
        data=np.zeros((len(means),self.lofd))
        for i in range(len(means)):
            X=np.linspace(0,300,num=300,endpoint=False)
            mean=means[i]
            cov=covs[i]
            weight=weights[i]
            Ys=utils.toGM(X,self.n_components,mean,cov,weight)
            data[i]=np.sum(Ys,axis=0)
        return data

    def rule_classication(self,z):
        data=self.decode(z)
        maxd=[]
        for dat in data:
            maxd.append(max(dat))
        ans=np.zeros((z.shape[0],))
        pre=np.zeros((z.shape[0],self.n_components))
        allmeans=z[:,self.n_components:2*self.n_components]
        allcovs=z[:,2*self.n_components:3*self.n_components]
        allweights=z[:,0:self.n_components]
        normedweights=np.zeros(allweights.shape)
        for i in range(len(allweights)):
            normedweights[i]=allweights[i]/sum(allweights[i])
        for i in range(z.shape[0]):
            ###preprocessing the data to avoid peak overlapping(far overlap and near overlap) influence: identify far/near overlap cases and suppress far overlap peaks, amplify near overlap peaks
            ###如果很理想的情况应该能把两个far overlap的peak合并成一个在中间mean的，但是现在可以先直接把两个抑制掉，毕竟就不太可能是单克隆峰了。far overlap也就是两个峰实际上在图里面是同一个，BGM将其拆分从而更好的拟合高斯模型，我们这里将其抑制因为能够拆分为两个峰的基本上cov都比较大，不尖。
            for j in range(self.n_components):
                for l in range(self.n_components):
                    if j<l:
                        if normedweights[i][j]/normedweights[i][l]>3 or normedweights[i][j]/normedweights[i][l]<0.3333:#ignore when weight difference is too large
                            continue
                        if allcovs[i][j]/normedweights[i][j]/allcovs[i][l]*normedweights[i][l]/abs(allmeans[i][j]-allmeans[i][l])*utils.mean(np.sqrt(allcovs[i][j]),np.sqrt(allcovs[i][l]))>2 or allcovs[i][l]/normedweights[i][l]/allcovs[i][j]*normedweights[i][j]/abs(allmeans[i][j]-allmeans[i][l])*utils.mean(np.sqrt(allcovs[i][j]),np.sqrt(allcovs[i][l]))>2:#if the cov difference is large than it will be ignored from far overlap because there should be two peaks in the original density plot
                        #near overlap situation is when a sharp peak is on a mild one. it happens when monoclonal peak has a background polyclonal peak. here we amplify the sharp peaks' weight when their cov difference is large enough or their distance is close enough so that it will be detected as abnormal in the classification step
                            if abs(allmeans[i][j]-allmeans[i][l])<3.5*np.sqrt(max(allcovs[i][j],allcovs[i][l])):
                                neww=normedweights[i][j]+normedweights[i][l]
                                if allcovs[i][l]/normedweights[i][l]/allcovs[i][j]*normedweights[i][j]>1 and normedweights[i][j]>0.15:
                                    if allcovs[i][j]<400:
                                        normedweights[i][j]=neww
                                else:
                                    if allcovs[i][l]<400:
                                        normedweights[i][l]=neww
                            continue
                        if allcovs[i][j]/normedweights[i][j]/sum(allweights[i])<self.t3/2.5 or allcovs[i][l]/normedweights[i][l]/sum(allweights[i])<self.t3/2.5:#if one of the considered peak has very small variance, then it should not be far overlap situation where the original peak is mild
                            continue
                        if allcovs[i][j]<70 or allcovs[i][l]<70:
                            continue
                        elif abs(allmeans[i][j]-allmeans[i][l])<3.5*np.sqrt(max(allcovs[i][j],allcovs[i][l])):#far overlap situation where there is only a mild peak in the original density plot, and GMM model break it down to two sharper peaks to fit the guassian curves more accurately. here we just suppress the peaks and thus we cannot determine the column is abnormal because of the two considered components
                            pre[i][j]=pre[i][l]=1           
        for i in range(z.shape[0]):
            for j in range(self.n_components):
                if pre[i][j]==1:
                    continue
                if maxd[i]<80:
                    continue
                elif normedweights[i][j]<0.05:
                    continue
                if allcovs[i][j]/normedweights[i][j]/sum(allweights[i])>self.t3:###t-figure
                    continue
                else:
                    ans[i]=1
        return ans

class GMMmodel():
    def __init__(self, visualization=0, lofd=300, n_components=3):
        self.module=GMM(visualization=visualization,lofd=lofd,n_components=n_components)