import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("/home/ghanshyam/Machine Learning/wdbc.data.csv",header=None)
X,y=df.iloc[:,2:].values,df.iloc[:,1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,test_size=0.2,random_state=1)

from sklearn.metrics import roc_curve,auc
from scipy import interp
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

pipe_lr=make_pipeline(StandardScaler(),PCA(n_components=2),
                      LogisticRegression(penalty='l2',random_state=1,C=100.0))
X_train2=X_train[:,[4,14]]
from sklearn.model_selection import StratifiedKFold
cv=list(StratifiedKFold(n_splits=3,random_state=1).split(X_train,y_train))
fig=plt.figure(figsize=(7,5))
mean_tpr=np.zeros(100,dtype='float64')
# print(mean_tpr)
mean_fpr=np.linspace(0,1,100)
# print(mean_fpr.shape[0])
all_tpr=[]
for i,(train,test) in enumerate(cv):
    probas=pipe_lr.fit(X_train2[train],y_train[train]).predict_proba(X_train2[test])
    fpr,tpr,threshold=roc_curve(y_train[test],probas[:,1],pos_label='M')
    mean_tpr+=interp(mean_fpr,fpr,tpr)
    mean_tpr[0]=0.0
    roc_auc=auc(fpr,tpr)
    plt.plot(fpr,tpr,label='ROC Fold %d(area=%.2f)'%(i+1,roc_auc))
plt.plot([0,1],[0,1],linestyle='--',color=(.6,.6,.6),label='Random Guessing')
mean_tpr /= len(cv)
# print(mean_tpr)
mean_tpr[-1]=1.0
mean_auc=auc(mean_fpr,mean_tpr)
plt.plot(mean_fpr,mean_tpr,'k--',label="mean ROC(area=%.2f)"%mean_auc,lw=2)
plt.plot([0,0,1],[0,1,1],linestyle=':',color='black',label='Perfect Performance')
plt.xlim([-.05,1.05])
plt.ylim([-.05,1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show(block=False)
plt.pause(2)
plt.close()

# pre_scorer=make_scorer

###########Class Imbalance

X_imb=np.vstack((X[y==0],X[y==1][:40]))
y_imb=np.hstack((y[y==0],y[y==1][:40]))
print(y_imb)
for i in range(y_imb.shape[0]):
    if y_imb[i]=='B':
        y_imb[i]=0
    else:
        y_imb[i]=1
print("\n\n\n\n\n\n")
print(y_imb)
y_pred=np.zeros(y_imb.shape[0])
print("/n/n/n/nn/n/n/")
print(y_pred)
print(np.mean(y_pred==y_imb)*100)