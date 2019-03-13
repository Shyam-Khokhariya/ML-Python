import numpy as np
import pandas as pd
df=pd.read_csv("/home/ghanshyam/Machine Learning/wdbc.data.csv",header=None)
X,y=df.iloc[:,2:].values,df.iloc[:,1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,test_size=0.2,random_state=1)

from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
pipe_svc=make_pipeline(StandardScaler(),SVC(random_state=1))

pipe_svc.fit(X_train,y_train)
y_pred=pipe_svc.predict(X_test)
con=confusion_matrix(y_true=y_test,y_pred=y_pred)
print(con)
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(2.5, 2.5))
ax.matshow(con,cmap=plt.cm.Reds,alpha=0.3)
for i in range(con.shape[0]):
    for j in range(con.shape[1]):
        ax.text(x=j,y=i,s=con[i,j],va='center',ha='center')
plt.xlabel("Predicted Value")
plt.ylabel("True Value")
plt.show(block=False)
plt.pause(2)
plt.close()



from sklearn.metrics import precision_score
from sklearn.metrics import recall_score,f1_score
print("Precision : %.3f"%precision_score(y_true=y_test,y_pred=y_pred,pos_label='M'))
print("Recall : %.3f"%recall_score(y_test,y_pred,pos_label='M'))
print("f1 Score : %.3f"%f1_score(y_test,y_pred,pos_label='M'))


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer,f1_score
param_range=[0.0001,.001,.01,.1,1,10,100,1000]
param_grid=[{'svc__C':param_range,
             'svc__kernel':['linear']},
            {'svc__C':param_range,
             'svc__gamma':param_range,
             'svc__kernel':['rbf']}]

scorer =make_scorer(f1_score)
gs=GridSearchCV(estimator=pipe_svc,
                param_grid=param_grid,
                scoring=scorer,cv=10)
gs=gs.fit(X_train,y_train)
print(gs.best_score_)
print(gs.best_params_)
