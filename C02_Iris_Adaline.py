import C02_Adaline as AC02
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import C02_Iris as IC02

df=pd.read_csv('/home/ghanshyam/Machine Learning/Python Machine Learning by Sebastian Raschka and Vahid Mirjalili-2nd Edition/iris_data.csv',header=None)
y=df.iloc[0:100,4].values
y=np.where(y=='Iris-setosa',-1,1)
X=df.iloc[0:100,[0,2]].values

fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(10,4))
ada1=AC02.Adaline(n_iter=10,eta=0.01).fit(X,y)
ax[0].plot(range(1,len(ada1.cost_)+1),np.log(ada1.cost_),marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Log(Sum - squared - error)')
ax[0].set_title('Adaline - Learning Rate 0.01')

ada2 = AC02.Adaline(n_iter=10000,eta=0.00001).fit(X,y)
ax[1].plot(range(1,len(ada2.cost_)+1),np.log(ada2.cost_))
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Log(Sum - squared - error)')
ax[1].set_title('Adaline - Learning Rate 0.0001')

plt.show(block=False)
plt.pause(1)
plt.close()



X_std=np.copy(X)
X_std[:,0]=(X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1]=(X[:,1] - X[:,1].mean()) / X[:,1].std()

ada=AC02.Adaline(eta=0.01,n_iter=15)
ada.fit(X_std,y)
IC02.plot_decision_regions(X_std,y,classifier=ada)
plt.title('Adaline Gradient Descent')
plt.xlabel("sepal - length [Standard]")
plt.ylabel("petal - length [Standard]")
plt.legend()
plt.tight_layout()
plt.show(block=False)
plt.pause(1)
plt.close()

plt.plot(range(1,len(ada.cost_)+1),ada.cost_,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-Squared-error')
plt.show(block=False)
plt.pause(1)
plt.close()