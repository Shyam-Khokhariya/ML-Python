import numpy as np
import pandas as pd
df=pd.read_csv("/home/ghanshyam/Machine Learning/wdbc.data.csv",header=None)
# print(df.head())
X,y=df.iloc[:,2:].values,df.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,test_size=0.2,random_state=1)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import  make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
pipe_lr=make_pipeline(StandardScaler(),PCA(n_components=2),LogisticRegression(random_state=1))
# pipe_lr.fit(X_train,y_train)

from sklearn.model_selection import StratifiedKFold
kfold=StratifiedKFold(n_splits=10,random_state=1).split(X_train,y_train)
scores=[]
for k,(train,test) in enumerate(kfold):
    pipe_lr.fit(X_train[train],y_train[train])
    score=pipe_lr.score(X_train[test],y_train[test])
    scores.append(score)
    # print(np.bincount(y_train[train])
    print("Fold :{0},class dist. :,Accur :{1}".format((k+1),score))
print("CV: %.3f +/- %.3f" % (np.mean(scores), np.std(scores)))

from sklearn.model_selection import cross_val_score
scores=cross_val_score(estimator=pipe_lr,
                       X=X_train,
                       y=y_train,
                       cv=10,
                       n_jobs=1)
print("CV Accuracy score : %s" %scores)
print("CV: %.3f +/- %.3f" %(np.mean(scores),np.std(scores)))