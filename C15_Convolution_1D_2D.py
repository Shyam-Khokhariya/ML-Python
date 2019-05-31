import numpy as np
import scipy.signal
def conv1d(x,w,p=0,s=1):
    w_rot=np.array(w[::-1])
    x_padded=np.array(x)
    if p>0:
        zeros=np.zeros(shape=p)
        x_padded=np.concatenate([zeros,x_padded,zeros])
    res=[]
    for i in range(0,int(len(x)/s),s):
        res.append(np.sum(x_padded[i:i+w_rot.shape[0]]*w_rot))
    return np.array(res)
x=[1,3,2,4,5,6,1,3]
w=[1,0,3,1,2]
print("Convolution 1D output : ",conv1d(x,w,p=2,s=1))
print("Numpy Conolution : ",np.convolve(x,w,mode="same"))


def conv2d(x,w,p=(0,0),s=(1,1)):
    w_rot=np.array(w)[::-1,::-1]
    x=np.array(x)
    n1=x.shape[0]+2*p[0]
    n2=x.shape[1]+2*p[1]
    x_padded=np.zeros(shape=(n1,n2))
    x_padded[p[0]:p[0]+x.shape[0],p[1]:p[1]+x.shape[1]]=x
    res=[]
    for i in range (0 , int((x_padded.shape[0]-w_rot.shape[0])/s[0])+1,s[0]):
        res.append([])
        for j in range(0,int((x_padded.shape[1]-w_rot.shape[1])/s[1])+1,s[1]):
            x_sub=x_padded[i:i+w_rot.shape[0],j:j+w_rot.shape[1]]
            res[-1].append(np.sum(x_sub*w_rot))
    return np.array(res)

x=[[1,2,3,4],[2,3,4,5],[1,4,3,2],[6,4,2,3]]
w=[[1,2,5],[3,2,4],[3,2,1]]
print("Convolution 2D Output : \n",conv2d(x,w,p=(1,1),s=(1,1)))
print("Scipy Result : \n",scipy.signal.convolve2d(x,w,mode="same"))
