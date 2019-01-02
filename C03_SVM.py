import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.datasets as sd
import C03_Decision_region as DRC03

iris = sd.load_iris()
X=iris.data[:,[2,3]]
y=iris.target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=3,random_state=1,stratify=y)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)
X_combined_std=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))

from sklearn.svm import SVC
svm=SVC(kernel='linear',C=1.0,random_state=1)
svm.fit(X_train_std,y_train)
DRC03.plot_decision_region(X_combined_std,y_combined,classifier=svm,test_idx=range(105,150))
plt.xlabel('Petal Length[standardized]')
plt.ylabel('Petal Width[standardized]')
plt.show(block=False)
plt.pause(2)
plt.close()


from sklearn.linear_model import SGDClassifier
ppn=SGDClassifier(loss='perceptron')
lr=SGDClassifier(loss='log')
svm=SGDClassifier(loss='hinge')


svm1=SVC(kernel='rbf',gamma=100,C=1.0,random_state=1)
svm1.fit(X_train_std,y_train)
DRC03.plot_decision_region(X_combined_std,y_combined,classifier=svm1,test_idx=range(105,150))
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend(loc='best')
plt.show(block=False)
plt.pause(2)
plt.close()

#---------------------------------------#
#         Building Decision Tree        #
#---------------------------------------#

from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier(criterion='gini',max_depth=4,random_state=1)
tree.fit(X_train,y_train)
X_combined=np.vstack((X_train,X_test))
y_combined=np.hstack((y_train,y_test))
DRC03.plot_decision_region(X_combined,y_combined,classifier=tree,test_idx=range(105,150))
plt.xlabel('Petal Length [cm]')
plt.ylabel('Petal Width [cm]')
plt.legend()
plt.show(block=False)
plt.pause(2)
plt.close()

from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
dot_data=export_graphviz(tree,filled=True,rounded=True,class_names=['Setosa','Versicolor','Virginica'],feature_names=['petal length','petal width'],out_file=None)
graph=graph_from_dot_data(dot_data)
graph.write_png('tree.png')




#-----------------------------#
#       Random Forest         #
#-----------------------------#

from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier(criterion='gini',n_estimators=25,random_state=1,n_jobs=2)
forest.fit(X_train,y_train)
DRC03.plot_decision_region(X_combined,y_combined,classifier=forest,test_idx=range(105,150))
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend()
plt.show(block=False)
plt.pause(2)
plt.close()


#------------------------------#
#       K-Nearest Neighbour    #
#------------------------------#

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski')
knn.fit(X_train_std,y_train)
DRC03.plot_decision_region(X_combined_std,y_combined,classifier=knn,test_idx=range(105,150))
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend(loc='upper left')
plt.show(block=False)
plt.pause(2)
plt.close()