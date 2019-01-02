import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
z=np.arange(-9,9,0.1)
phi_z=sigmoid(z)
plt.plot(z,phi_z)
plt.axvline(0.0,color='k')
plt.ylim(-0.1,1.1)
plt.xlabel('z')
plt.ylabel('$\phi (z)$')
plt.yticks([0.0,0.5,1.0])
ax=plt.gca()
ax.yaxis.grid(True)
plt.show(block=False)
plt.pause(1)
plt.close()


def cost_1(z):
    return -np.log(sigmoid(z))
def cost_0(z):
    return -np.log(1-sigmoid(z))
z=np.arange(-10,10,0.1)
phi_z=sigmoid(z)
c1=[cost_1(x) for x in z]
plt.plot(phi_z,c1,label='J(w) if y=1')
c0=[cost_0(x) for x in z]
plt.plot(phi_z,c0,linestyle='--',label='J(w) if y=0')
plt.ylim(0.0,5.1)
plt.xlim([0,1])
plt.xlabel('$\phi$(z)')
plt.ylabel('J(w)')
plt.legend(loc='best')
plt.show()
