import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import C02_Iris as IC02
import C02_Mini_Batch as MC02


df=pd.read_csv('/home/ghanshyam/Machine Learning/Python Machine Learning by Sebastian Raschka and Vahid Mirjalili-2nd Edition/iris_data.csv',header=None)
y=df.iloc[0:100,4].values
y=np.where(y=='Iris-setosa',-1,1)
X=df.iloc[0:100,[0,2]].values

X_std=np.copy(X)
X_std[:,0]=(X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1]=(X[:,1] - X[:,1].mean()) / X[:,1].std()


ada=MC02.AdalineSGD(0.01,15,random_state=1)
ada.fit(X_std,y)

IC02.plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show(block=False)
plt.pause(2)
plt.close()
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show(block=False)
plt.pause(2)
plt.close()