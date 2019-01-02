from sklearn import datasets
import numpy as np
import C03_Decision_region as D_C03
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target
print('Class Lable:',np.unique(y))

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)
print('Lable count in y:',np.bincount(y))
print('Lable count in y_train:',np.bincount(y_train))
print('Lable count in y_test:',np.bincount(y_test))


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)

from sklearn.linear_model import Perceptron
ppn = Perceptron(max_iter=40,eta0=0.01,random_state=1)
ppn.fit(X_train_std,y_train)
y_pred = ppn.predict(X_test_std)
print("Misclassified Sample:%d"%(y_pred!=y_test).sum())

from sklearn.metrics import accuracy_score
print("Accuracy:%f"%accuracy_score(y_test,y_pred))

X_combined_std=np.vstack((X_train_std,X_test_std))
y_combined_std=np.hstack((y_train,y_test))
D_C03.plot_decision_region(X=X_combined_std,y=y_combined_std,classifier=ppn,test_idx=range(105,150))
plt.xlabel('Petal Length ')
plt.ylabel('Petal Width')
plt.legend(loc='upper left')
plt.show(block=False)
plt.pause(1)
plt.close()