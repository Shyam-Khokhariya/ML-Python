import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("/home/ghanshyam/Machine Learning/wine.data",header=None)
df.columns=['ClassLable','Alcohol','Malic acid','Ash',
            'Alcalinity of ash','Magnesium','Total phenols',
            'Flavanoids','Nonflavanoid phenols','Proanthocyanins',
            'Color intensity','Hue','OD280/OD315 of diluted wines','Proline']
X,y=df.iloc[:,1:].values,df.iloc[:,0].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=0,stratify=y)

from sklearn.ensemble import RandomForestClassifier

feat_labels=df.columns[1:]
feat_label=[]
forest=RandomForestClassifier(n_estimators=500,random_state=1)
forest.fit(X_train,y_train)
importances=forest.feature_importances_
indices=np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    feat_label.append(feat_labels[indices[f]])
    print("%2d %-*s %f"%(f+1,30,feat_labels[indices[f]],importances[indices[f]]))
plt.title('Features Importance')
plt.bar(range(X_train.shape[1]),importances[indices],align='center')
plt.xticks(range(X_train.shape[1]),feat_label, rotation=90)
plt.xlim([-1,X_train.shape[1]])
plt.tight_layout()
plt.show(block=False)
plt.pause(1)
plt.close()


from sklearn.feature_selection import SelectFromModel
sfm=SelectFromModel(forest,threshold=0.1,prefit=True)
X_selected=sfm.transform(X_train)
print('Number of sample that meet this criterion:',X_selected.shape[0])
for f in range(X_selected.shape[1]):
    print("%2d) %-*s %f"% (f+1,30,feat_labels[indices[f]],importances[indices[f]]))