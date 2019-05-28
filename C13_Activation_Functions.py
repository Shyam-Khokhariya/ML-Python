import numpy as np
import matplotlib.pyplot as plt


#------------------Logistic Function-------------------#

X=np.array([1,2,3])
w=np.array([0.2,0.3,0.4])
def net_input(X,w):
    return np.dot(X,w)

def logistic(z):
    return (1.0/(1.0+np.exp(-z)))

def log_act(X,w):
    z=net_input(X,w)
    return logistic(z)

print("P(y=1|x) = %.3f"%log_act(X,w))

W=np.array([[1.1,1.2,1.3,1.4],
            [.2,.4,.7,1.0],
            [1.2,.6,.9,1.0]])
A=np.array([[1.0,1.4,1.2,.7]])
Z=np.dot(W,A[0])
y_pro=logistic(Z)
print("Net Input :",Z)
print("Output Units :",y_pro)
print("Class :",np.argmax(Z,axis=0))


#---------------Softmax Function----------------#
def softmax(z):
    return np.exp(z)/np.sum(np.exp(z))
y_prob=softmax(Z)
print("Probabilities : ",y_prob)
print(np.sum(y_prob))


#----------------Tanh Function------------------#

def tanh(z):
    ep=np.exp(z)
    en=np.exp(-z)
    return (ep-en)/(ep+en)
z=np.arange(-5,5,0.005)
log_acti=logistic(z)
tanh_act=tanh(z)
plt.ylim([-1.5,1.5])
plt.xlabel("net input $z$")
plt.ylabel("activation $\phi (z) $")
plt.axhline(1,color="black",linestyle=":")
plt.axhline(.5,color="black",linestyle=":")
plt.axhline(0,color="black",linestyle=":")
plt.axhline(-0.5,color="black",linestyle=":")
plt.axhline(-1,color="black",linestyle=":")
plt.plot(z,tanh_act,label="tanh",color="blue",linestyle="--")
plt.plot(z,log_acti,label="Logistic",color="g")
plt.legend()
plt.show()