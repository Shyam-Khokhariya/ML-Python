import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import C02_Perceptron as C02

df=pd.read_csv('/home/ghanshyam/Machine Learning/Python Machine Learning by Sebastian Raschka and Vahid Mirjalili-2nd Edition/iris_data.csv',header=None)


y=df.iloc[0:100,4].values
y=np.where(y=='Iris-setosa',-1,1)

X=df.iloc[0:100,[0,2]].values
# plt.scatter(X[:50,0],X[:50,1],color='red',marker='o',label='setosa')
# plt.scatter(X[50:100,0],X[50:100,1],color='blue',marker='x',label='versicolor')
# plt.xlabel('sepal length [cm]')
# plt.ylabel('petal length [cm]')
# plt.legend(loc='upper left')
# plt.show()


ppn=C02.Perception()
ppn.fit(X,y)
# plt.plot(range(1,len(ppn.error_) + 1),ppn.error_,marker='o')
# plt.xlabel('EpochS')
# plt.ylabel('Number of updates')
# plt.show()

from matplotlib.colors import ListedColormap

def plot_decision_regions(X,y,classifier,resolution=0.02):
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap=ListedColormap(colors[:len(np.unique(y))])

    x1_min,x1_max=X[:,0].min()-1,X[:,0].max()+1
    x2_min,x2_max=X[:,1].min()-1,X[:,1].max()+1
    xx1,xx2=np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))

    Z=classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z=Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,Z,alpha=0.3,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0],y=X[y==cl,1],alpha=0.8,c=colors[idx],marker=markers[idx],label=cl,edgecolor='black')


plot_decision_regions(X,y,classifier=ppn)
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.legend()
plt.show(block=False)
plt.pause(2)
plt.close()
