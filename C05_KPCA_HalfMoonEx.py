import C05_Kernel_PCA as C05KPCA
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np
X,y=make_moons(100,random_state=123)
plt.scatter(X[y==0,0],X[y==0,1],marker='x',color='blue',alpha=1)
plt.scatter(X[y==1,0],X[y==1,1],marker='o',color='green',alpha=1)
plt.show(block=False)
plt.pause(1)
plt.close()

from sklearn.decomposition import PCA
scikit_pca=PCA(n_components=2)
X_spca=scikit_pca.fit_transform(X)
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(10,4))
ax[0].scatter(X_spca[y==0,0],X_spca[y==0,1],marker='x',color='blue',alpha=1)
ax[0].scatter(X_spca[y==1,0],X_spca[y==1,1],marker='o',color='green',alpha=1)
ax[1].scatter(X_spca[y==0,0],np.zeros((50,1))+0.02,marker='x',color='blue')
ax[1].scatter(X_spca[y==1,0],np.zeros((50,1))-0.02,marker='o',color='green')
ax[0].set_xlabel("PC1")
ax[0].set_ylabel("PC2")
ax[1].set_ylim([-1,1])
ax[1].set_yticks([])
ax[1].set_xlabel("PC1")
plt.show(block=False)
plt.pause(1)
plt.close()


from sklearn.decomposition import PCA
scikit_pca=PCA(n_components=2)
X_kpca=C05KPCA.rbf_kernel_pca(X,15,2)
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(10,4))
ax[0].scatter(X_kpca[y==0,0],X_kpca[y==0,1],marker='x',color='blue',alpha=1)
ax[0].scatter(X_kpca[y==1,0],X_kpca[y==1,1],marker='o',color='green',alpha=1)
ax[1].scatter(X_kpca[y==0,0],np.zeros((50,1))+0.02,marker='x',color='blue')
ax[1].scatter(X_kpca[y==1,0],np.zeros((50,1))-0.02,marker='o',color='green')
ax[0].set_xlabel("KPC1")
ax[0].set_ylabel("KPC2")
ax[1].set_ylim([-1,1])
ax[1].set_yticks([])
ax[1].set_xlabel("KPC1")
plt.show(block=False)
plt.pause(1)
plt.close()
