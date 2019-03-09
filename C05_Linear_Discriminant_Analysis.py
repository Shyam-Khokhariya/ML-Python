import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv("/home/ghanshyam/Machine Learning/wine.data",header=None)
df.columns=['ClassLable','Alcohol','Malic acid','Ash',
            'Alcalinity of ash','Magnesium','Total phenols',
            'Flavanoids','Nonflavanoid phenols','Proanthocyanins',
            'Color intensity','Hue','OD280/OD315 of diluted wines','Proline']

X,y=df.iloc[:,1:].values,df.iloc[:,0].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,stratify=y,random_state=0)
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_train_std=ss.fit_transform(X_train)
X_test_std=ss.transform(X_test)

np.set_printoptions(precision=4)
mean_vecs=[]
for label in range(1,4):
    mean_vecs.append(np.mean(X_train_std[y_train==label],axis=0))
    print("MV %s: %s\n"%(label,mean_vecs[label-1]))

d=13
S_W=np.zeros((d,d))
for label,mv in zip(range(1,4),mean_vecs):
    class_scatter=np.zeros((d,d))
    for row in X_train_std[y_train==label]:
        row,mv=row.reshape(d,1),mv.reshape(d,1)
        class_scatter+=(row-mv).dot((row-mv).T)
    S_W+=class_scatter
print("Within Class Matrix %sx%s"%(S_W.shape[0],S_W.shape[1]))
print("CLass Label Distribution:%s"%np.bincount(y_train)[1:])

d=13
S_W=np.zeros((d,d))
for label,mv in zip(range(1,4),mean_vecs):
    class_scatter=np.cov(X_train_std[y_train==label].T)
    S_W+=class_scatter
print("Within Class Matrix(using Covariance Matrix):%sx%s"%(S_W.shape[0],S_W.shape[1]))

mean_overall=np.mean(X_train_std,axis=0)
d=13
S_B=np.zeros((d,d))
for i,mean_vec in enumerate(mean_vecs):
    n=X_train[y_train==i+1,:].shape[0]
    mean_vec=mean_vec.reshape(d,1)
    mean_overall=mean_overall.reshape(d,1)
    S_B+=(mean_vec-mean_overall).dot((mean_vec-mean_overall).T)
print("Between Class Matrix:%sx%s"%(S_B.shape[0],S_B.shape[1]))


eigen_val,eigen_vec=np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
eigen_pair=[(np.abs(eigen_val[i]),eigen_vec[:,i]) for i in range(len(eigen_val))]
eigen_pair=sorted(eigen_pair,key=lambda k:k[0],reverse=True)
print("Eigen Val in Descending Order:")
for i in eigen_pair:
    print(i[0])

tot=sum(eigen_val.real)
discr=[(i/tot) for i in sorted(eigen_val.real,reverse=True)]
cum_discre=np.cumsum(discr)
plt.bar(range(1,14),discr,alpha=0.5,align="center",label="individual 'Discriminability'")
plt.step(range(1,14),cum_discre,where="mid",label="Cummulative 'Discriminability'")
plt.ylabel("'Discriminilability' Ratio")
plt.xlabel("Linear Discriminants")
plt.legend(loc="best")
plt.show(block=False)
plt.pause(1)
plt.close()

w=np.hstack((eigen_pair[0][1][:,np.newaxis].real,
             eigen_pair[1][1][:,np.newaxis].real))
print(w)

X_train_lda=X_train_std.dot(w)
color=['r','g','b']
marker=['x','*','o']
for l,c,m in zip(np.unique(y_train),color,marker):
    plt.scatter(X_train_lda[y_train==l,0],X_train_lda[y_train==l,1]*(-1),c=c,marker=m,label=l)
plt.xlabel("LD 1")
plt.ylabel("LD 2")
plt.legend()
plt.show(block=False)
plt.pause(1)
plt.close()


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components=2)
X_train_lda=lda.fit_transform(X_train_std,y_train)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr=lr.fit(X_train_lda,y_train)
import C05_Decision_Boundary as C05DR
C05DR.plot_decision_region(X_train_lda,y_train,classifier=lr)
plt.xlabel("LD 1")
plt.ylabel("LD 2")
plt.legend()
plt.show(block=False)
plt.pause(1)
plt.close()

X_test_lda=lda.transform(X_test_std)
C05DR.plot_decision_region(X_test_lda,y_test,classifier=lr)
plt.xlabel("LD 1")
plt.ylabel("LD 2")
plt.show(block=False)
plt.pause(1)
plt.close()
