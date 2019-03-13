import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("/home/ghanshyam/Machine Learning/wdbc.data.csv",header=None)
# print(df.head())
X,y=df.iloc[:,2:].values,df.iloc[:,1].values

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)
# print(le.transform(le.classes_),le.classes_)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1,stratify=y)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
pipe_lr=make_pipeline(StandardScaler(),
                      PCA(n_components=2),
                      LogisticRegression(random_state=1))
pipe_lr.fit(X_train,y_train)
y_pred=pipe_lr.predict(X_test)
print("Test Accuracy = %.3f" % pipe_lr.score(X_test,y_test))
