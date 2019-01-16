import numpy as np
import pandas as pd
import C04_Train_Test as TTC04

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(penalty='l1',C=1.0)
lr.fit(TTC04.X_train_std,TTC04.y_train)
print('Training Accuracy:',lr.score(TTC04.X_train_std,TTC04.y_train))
print('Test Accuracy:',lr.score(TTC04.X_test_std,TTC04.y_test))

print("Intercept:",lr.intercept_)
print("Coefficient:",lr.coef_)

import matplotlib.pyplot as plt
fig=plt.figure()
ax=plt.subplot(111)

colors=['yellow','green','blue','red','black','pink','magenta','cyan','indigo','lightgreen','lightblue','gray','orange']
weights,params=[],[]
for c in np.arange(-4.0,6.0):
    lr=LogisticRegression(penalty='l1',C=10.0**c,random_state=0)
    lr.fit(TTC04.X_train_std,TTC04.y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights=np.array(weights)
for column,color in zip(range(weights.shape[1]),colors):
    plt.plot(params,weights[:,column],label=TTC04.df.columns[column+1],color=color)
plt.axhline(0,color='black',linestyle='--',linewidth=3)
plt.xlim([10**(-5),10**5])
plt.ylabel('Weight Coefficient')
plt.xlabel("C")
plt.xscale('log')
plt.legend(loc="upper left")

ax.legend(loc='upper center',
          bbox_to_anchor=(1.38, 1.03),
          ncol=1, fancybox=True)

# ax.legend(loc="upper center",bbox_to_anchor=(1.38,1.03),ncol=1,fancybox=True)
plt.show()
