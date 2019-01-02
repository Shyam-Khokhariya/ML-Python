import C03_Adaline_into_Logistic_Regression as ALRC03
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as sd
import C03_Decision_region as DRC03

iris=sd.load_iris()
X=iris.data[:,[2,3]]
y=iris.target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=3,random_state=1,stratify=y)

X_train_01_subset=X_train[(y_train==1)|(y_train==0)]
y_train_01_subset=y_train[(y_train==1)|(y_train==0)]
lrgd=ALRC03.LogisticRegressionGD(0.05,1000,1)
lrgd.fit(X_train_01_subset,y_train_01_subset)
DRC03.plot_decision_region(X_train_01_subset,y_train_01_subset,lrgd)
plt.xlabel('Petal Size[Standardized]')
plt.ylabel('Petal Width[Standardized]')
plt.legend(loc='upper left')
plt.show(block=False)
plt.pause(2)
plt.close()

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)
X_combined_std=np.vstack((X_train_std,X_test_std))
y_combined_std=np.hstack((y_train,y_test))

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(C=100.0,random_state=1)
lr.fit(X_train_std,y_train)
DRC03.plot_decision_region(X_combined_std,y_combined_std,lr,range(105,150))
plt.xlabel('Petal Height')
plt.ylabel('Petal Wigth')
plt.legend(loc='upper left')
plt.show(block=False)
plt.pause(2)
plt.close()

a=lr.predict_proba(X_test_std[:3,:])
print(a)
b=lr.predict_proba(X_test_std[:3,:]).argmax(axis=1)
print(b)
b=lr.predict(X_test_std[:3,:])
print(b)
c=lr.predict(X_test_std[1,:].reshape(1,-1))
print(c)

 # ..............................
 # Overfitting Via Regularization
 # ..............................

weights, params = [], []
for c in np.arange(-5, 5):
 lr = LogisticRegression(C=10.**c, random_state=1)
 lr.fit(X_train_std, y_train)
 weights.append(lr.coef_[1])
 params.append(10.**c)
weights = np.array(weights)
plt.plot(params, weights[:, 0],label='petal length')
plt.plot(params, weights[:, 1], linestyle='--',label='petal width')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()