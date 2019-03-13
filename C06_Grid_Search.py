import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("/home/ghanshyam/Machine Learning/wdbc.data.csv",header=None)
X,y=df.iloc[:,2:].values,df.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,test_size=0.2,random_state=1)

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
pipe_svc=make_pipeline(StandardScaler(),SVC(random_state=1))
param_range=[0.0001,.001,.01,.1,1,10,100,1000]
param_grid=[{'svc__C':param_range,
             'svc__kernel':['linear']},
            {'svc__C':param_range,
             'svc__gamma':param_range,
             'svc__kernel':['rbf']}]
gs=GridSearchCV(estimator=pipe_svc,
                param_grid=param_grid,
                scoring='accuracy',cv=10,
                n_jobs=-1)
gs=gs.fit(X_train,y_train)
print(gs.best_score_)
print(gs.best_params_)
clf=gs.best_estimator_
print(clf)
clf.fit(X_train,y_train)
print('Test Accuracy:%.3f'%clf.score(X_test,y_test))

from sklearn.model_selection import cross_val_score
gs1=GridSearchCV(estimator=pipe_svc,
                 param_grid=param_grid,cv=2,scoring='accuracy')
scores=cross_val_score(gs1,X_train,y_train,scoring='accuracy',cv=5)
print("CV SCORE: %.3f +/- %.3f" %(np.mean(scores),np.std(scores)))

from sklearn.tree import DecisionTreeClassifier
gs1=GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                 param_grid=[{'max_depth':[1,2,3,4,5,6,7,None]}],
                 scoring='accuracy',cv=2)
scores=cross_val_score(gs1,X_train,y_train,scoring='accuracy',cv=5)
print("CV SCORE (DECISION TREE) : %.3f +/- %.3f"%(np.mean(scores),np.std(scores)))