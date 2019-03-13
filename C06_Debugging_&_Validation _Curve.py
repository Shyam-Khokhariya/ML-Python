import numpy as np

import pandas as pd
df=pd.read_csv("/home/ghanshyam/Machine Learning/wdbc.data.csv",header=None)
X,y=df.iloc[:,2:].values,df.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1,stratify=y,test_size=.2)

from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

pipe_lr=make_pipeline(StandardScaler(),LogisticRegression(penalty='l2',random_state=1))
train_sizes,train_scores,test_scores=learning_curve(estimator=pipe_lr,X=X_train,y=y_train,
                                                     train_sizes=np.linspace(0.1,1.0,10),
                                                     cv=10,n_jobs=1)
train_mean=np.mean(train_scores,axis=1)
train_std=np.std(train_scores,axis=1)
test_mean=np.mean(test_scores,axis=1)
test_std=np.std(test_scores,axis=1)

plt.plot(train_sizes,train_mean,color='blue',marker='o',markersize=5,label='Training Accuracy')
plt.fill_between(train_sizes,train_mean+train_std,train_mean-train_std,alpha=0.15,color='blue')

plt.plot(train_sizes,test_mean,color='red',marker='^',markersize=5,label="Validation Accuracy")
plt.fill_between(train_sizes,test_mean+test_std,test_mean-test_std,alpha=0.15,color='red')

plt.grid()
plt.xlabel("Number Of Traing Examples")
plt.ylabel("Accuracy")
plt.legend()
plt.ylim([0.9,1.0])

plt.show()


from sklearn.model_selection import validation_curve
param_range=[0.001,0.01,0.1,1.0,10.0,100.0]
train_scores,test_scores=validation_curve(estimator=pipe_lr,X=X_train,y=y_train,
                                          param_name='logisticregression__C',
                                          param_range=param_range,cv=10)

train_mean=np.mean(train_scores,axis=1)
train_std=np.std(train_scores,axis=1)
test_mean=np.mean(test_scores,axis=1)
test_std=np.std(test_scores,axis=1)

plt.plot(param_range,train_mean,color='blue',marker='o',label='Accuracy')
plt.fill_between(param_range,train_mean+train_std,train_mean-train_std,alpha=0.15,color='blue')

plt.plot(param_range,test_mean,color='black',marker='*',label='Validation Accuracy')
plt.fill_between(param_range,test_mean+test_std,test_mean-test_std,alpha=0.15,color='black')

plt.grid()
plt.xscale('log')
plt.legend()
plt.xlabel("Parameter C")
plt.ylabel("Accuracy")
plt.ylim([0.8,1.03])
plt.show()