import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("/home/ghanshyam/Machine Learning/Housing.csv")
df.columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

X,y=df[["LSTAT"]].values,df["MEDV"].values

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
tree=DecisionTreeRegressor(max_depth=3)
tree.fit(X,y)
sort_idx=X.flatten().argsort()
# lin_regplot=LinearRegression()
# lin_regplot(X[sort_idx],y[sort_idx],tree)

plt.scatter(X[sort_idx],y[sort_idx],c="blue",edgecolors="white",s=70)
plt.plot(X[sort_idx],tree.predict(X[sort_idx]),color='black',lw=2)


plt.xlabel("% Lower Status of Population [LSTAT]")
plt.ylabel("Price in $1000s [MEDV]")
plt.show(block=False)
plt.pause(1)
plt.close()


X,y=df.iloc[:,:-1].values,df.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1,test_size=0.4)

from sklearn.ensemble import RandomForestRegressor
forest=RandomForestRegressor(n_estimators=1000,
                             criterion="mse",
                             random_state=1,
                             n_jobs=-1)
forest.fit(X_train,y_train)
y_train_pred=forest.predict(X_train)
y_test_pred=forest.predict(X_test)
from sklearn.metrics import mean_squared_error,r2_score
print("MSE Train: %.3f Test: %.3f"%(mean_squared_error(y_train,y_train_pred),mean_squared_error(y_test,y_test_pred)))
print("R^2 Score Train: %.3f Test: %.3f"%(r2_score(y_train,y_train_pred),r2_score(y_test,y_test_pred)))
plt.scatter(y_train_pred,y_train_pred-y_train,c="steelblue",edgecolors="white",marker='o',s=35,alpha=0.9,label="Training Data")
plt.scatter(y_test_pred,y_test_pred-y_test,c="limegreen",edgecolors="white",marker="s",s=35,alpha=.9,label="Test Data")
plt.xlabel("Predicted Value")
plt.ylabel("Residual")
plt.hlines(y=0,xmin=-10,xmax=50,lw=2,color="black")
plt.xlim([-10,50])
plt.legend()
plt.show()

