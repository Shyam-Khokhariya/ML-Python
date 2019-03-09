import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("/home/ghanshyam/Machine Learning/wine.data",header=None)
df.columns=['ClassLable','Alcohol','Malic acid','Ash',
            'Alcalinity of ash','Magnesium','Total phenols',
            'Flavanoids','Nonflavanoid phenols','Proanthocyanins',
            'Color intensity','Hue','OD280/OD315 of diluted wines','Proline']

X,y=df.iloc[:,1:].values, df.iloc[:,0].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0,stratify=y)
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_train_std=ss.fit_transform(X_train)
X_test_std=ss.transform(X_test)

cov_mat=np.cov(X_train_std.T)
eigen_vals,eigen_vecs=np.linalg.eig(cov_mat)
# print("Eigen Values:",eigen_vals)

tot=sum(eigen_vals)
var_exp=[(i/tot) for i in sorted(eigen_vals,reverse=True)]
cum_var_exp=np.cumsum(var_exp)
plt.bar(range(1,14),var_exp,alpha=0.5,align='center',label='Individual Explained Variance')
plt.step(range(1,14),cum_var_exp,where='mid',label="Cumulative Explained Variance")
plt.xlabel("Principal Component Index")
plt.ylabel("Explained Variance Ratio")
plt.legend(loc="best")
plt.show(block=False)
plt.pause(1)
plt.close()

eigen_pair=[(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pair.sort(key=lambda k:k[0],reverse=True)

w=np.hstack((eigen_pair[0][1][:,np.newaxis],eigen_pair[1][1][:,np.newaxis]))
# print("matrix W:")
# print(w)
X_train_pca=X_train_std.dot(w)
# print("Hello")
# print(X_train_pca)
colors=['r','b','g']
markers=['x','o','s']
for l,c,m in zip(np.unique(y_train),colors,markers):
    plt.scatter(X_train_pca[y_train==l, 0],
                X_train_pca[y_train==l, 1],
                c=c,label=l,marker=m)
plt.xlabel("PC 1")
plt.ylabel("PC 2")
# plt.xlim(-4,4)
# plt.ylim(-4,4)
plt.legend(loc="lower left")
plt.show(block=False)
plt.pause(1)
plt.close()

from sklearn.linear_model import LogisticRegression
import C05_Decision_Boundary as C05DB
from sklearn.decomposition import PCA

pca=PCA(n_components=2)
lr=LogisticRegression()
X_train_pca=pca.fit_transform(X_train_std)
X_test_pca=pca.transform(X_test_std)
lr.fit(X_train_pca,y_train)
C05DB.plot_decision_region(X_train_pca,y_train,classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show(block=False)
plt.pause(1)
plt.close()

C05DB.plot_decision_region(X_test_pca,y_test,classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show(block=False)
plt.pause(1)
plt.close()

# pca=PCA(n_components=None)
# X_train_pca=pca.fit_transform(X_train_std)
# print(pca.explained_variance_ratio_)