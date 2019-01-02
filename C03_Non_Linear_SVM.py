import  matplotlib.pyplot as plt
import numpy as np
np.random.seed(1)
X_xor=np.random.randn(200,2)
y_xor=np.logical_xor(X_xor[:,0]>0,X_xor[:,1]>0)
y_xor=np.where(y_xor,1,-1)
plt.scatter(X_xor[y_xor==1,0],X_xor[y_xor==1,1],c='b',marker='x',label='1')
plt.scatter(X_xor[y_xor==-1,0],X_xor[y_xor==-1,1],c='green',marker='o',label='-1')
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.legend(loc='best')
plt.show(block=False)
plt.pause(2)
plt.close()

import C03_Decision_region as DRC03
from sklearn.svm import SVC
svm=SVC(kernel='rbf',random_state=1,gamma=0.1,C=10.0)
svm.fit(X_xor,y_xor)
DRC03.plot_decision_region(X_xor,y_xor,classifier=svm)
plt.show(block=False)
plt.pause(2)
plt.close()