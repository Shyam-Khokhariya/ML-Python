import C05_KPCA_Modified as C05KPCA
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
X,y=make_moons(100,random_state=123)
alpha,lambdas=C05KPCA.rbf_kernel_pca(X,15,1)

X_new=X[25]
print(X_new)
X_proj=alpha[25]
print(X_proj)

def project_x(X_new,X,gamma,alpha,lambdas):
    pair_dis=np.array([np.sum((X_new-row)**2) for row in X])
    k=np.exp(-gamma*pair_dis)
    return k.dot(alpha/lambdas)

X_reproj=project_x(X_new,X,15,alpha,lambdas)
print(X_reproj)

plt.scatter(alpha[y==0, 0], np.zeros((50)),color='red', marker='^',alpha=0.5)
plt.scatter(alpha[y==1, 0], np.zeros((50)),color='blue', marker='o', alpha=0.5)
plt.scatter(X_proj, 0, color='black',label='original projection of point X[25]',marker='^', s=100)
plt.scatter(X_reproj, 0, color='green',label='remapped point X[25]',marker='x', s=500)
plt.legend(scatterpoints=1)
plt.show(block=False)
plt.pause(1)
plt.close()

from sklearn.decomposition import KernelPCA
X,y=make_moons(n_samples=100,random_state=123)
kpca=KernelPCA(n_components=2,kernel='rbf',gamma=15)
X_kpca=kpca.fit_transform(X)

plt.scatter(X_kpca[y==0,0],X_kpca[y==0,1],marker='^',color='red',alpha=1)
plt.scatter(X_kpca[y==1,0],X_kpca[y==1,1],marker='o',color='blue',alpha=1)
plt.xlabel('PC1')
plt.ylabel("PC1")
plt.show(block=False)
plt.pause(1)
plt.close()