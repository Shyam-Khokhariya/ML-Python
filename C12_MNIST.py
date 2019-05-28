import os
import struct
import numpy as np

def load_mnist(path,kind="train"):
    label_path=os.path.join(path,"%s-labels-idx1-ubyte"%kind)
    image_path=os.path.join(path,"%s-images-idx3-ubyte"%kind)
    with open(label_path,'rb') as lpath:
        magic,n=struct.unpack('>II',lpath.read(8))
        labels=np.fromfile(lpath,dtype=np.uint8)
    with open(image_path,'rb') as ipath:
        magic,num,rows,cols=struct.unpack('>IIII',ipath.read(16))
        images=np.fromfile(ipath,dtype=np.uint8).reshape(len(labels),784)
        images=((images/255.)-.5)*2
    return images,labels

X_train,y_train=load_mnist("","train")
X_test,y_test=load_mnist("","t10k")
print("Train:",X_train.shape[0],X_train.shape[1])
print("Test:",X_test.shape[0],X_test.shape[1])

import matplotlib.pyplot as plt
fig,ax=plt.subplots(nrows=2,ncols=5,sharex=True,sharey=True)
ax=ax.flatten()
for i in range(10):
    img=X_train[y_train==i][0].reshape(28,28)
    ax[i].imshow(img,cmap='gist_earth')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show(block=False)
plt.pause(1)
plt.close()

fig,ax=plt.subplots(nrows=5,ncols=5,sharex=True,sharey=True)
ax=ax.flatten()
for i in range(25):
    img=X_train[y_train==5][i].reshape(28,28)
    ax[i].imshow(img,cmap='gist_earth')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show(block=False)
plt.pause(1)
plt.close()

np.savez_compressed('mnist_scaled.npz',X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test)
