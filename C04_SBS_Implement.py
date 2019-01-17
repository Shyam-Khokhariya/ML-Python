import C04_Sequential_Backward_Selection as SBSC04
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df=pd.read_csv("/home/ghanshyam/Machine Learning/wine.data",header=None)
df.columns=['ClassLable','Alcohol','Malic acid','Ash',
            'Alcalinity of ash','Magnesium','Total phenols',
            'Flavanoids','Nonflavanoid phenols','Proanthocyanins',
            'Color intensity','Hue','OD280/OD315 of diluted wines','Proline']
from sklearn.model_selection import train_test_split
X,y=df.iloc[:,1:].values,df.iloc[:,0].values
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.3,stratify=y)
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_train_std=ss.fit_transform(X_train)
X_test_std=ss.fit_transform(X_test)


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)

sbs=SBSC04.SBS(knn,k_features=1)
sbs.fit(X_train_std,y_train)

k_feat=[len(k) for k in sbs.subsets_]
# print(k_feat)
# print(sbs.subsets_)
plt.plot(k_feat,sbs.scores_,marker='o')
plt.ylim([0.7,1.02])
plt.xlim([0,14])
plt.ylabel("Accuracy")
plt.xlabel("No. of features")
plt.grid()
plt.show(block=False)
plt.pause(1)
plt.close()

k3=list(sbs.subsets_[10])
print(df.columns[1:][k3])

knn.fit(X_train_std,y_train)
print('Training accuracy:',knn.score(X_train_std,y_train))
print('Test accuracy:',knn.score(X_test_std,y_test))

knn.fit(X_train_std[:,k3],y_train)
print('Training Accuracy:',knn.score(X_train_std[:,k3],y_train))
print('Testing Accuracy:',knn.score(X_test_std[:,k3],y_test))