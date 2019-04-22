import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("/home/ghanshyam/Machine Learning/Housing.csv")
df.columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

X=df[['LSTAT']].values
y=df[["MEDV"]].values

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
lr=LinearRegression()

from sklearn.preprocessing import PolynomialFeatures
pr2=PolynomialFeatures(degree=2)
pr3=PolynomialFeatures(degree=3)

X_2=pr2.fit_transform(X)
X_3=pr3.fit_transform(X)

X_fit=np.arange(X.min(),X.max(),1)[:,np.newaxis]
lr=lr.fit(X,y)
y_lr=lr.predict(X_fit)
lr_r2=r2_score(y,lr.predict(X))

lr=lr.fit(X_2,y)
y_2=lr.predict(pr2.fit_transform(X_fit))
pr2_r2=r2_score(y,lr.predict(X_2))

lr=lr.fit(X_3,y)
y_3=lr.predict(pr3.fit_transform(X_fit))
pr3_r2=r2_score(y,lr.predict(X_3))

plt.scatter(X,y,c='lightgrey',marker='o',label="Training Points")
# plt.plot(X_fit,y_lr,linestyle=':',color='blue',label="Linear(d=1),$R^2=%.2f$"%lr_r2)
plt.plot(X_fit,y_2,linestyle='--',color='green',label="Quadratic(d=2),$R^2=%.2f$"%pr2_r2)
plt.plot(X_fit,y_3,color='red',label="Cubic(d=3),$R^2=%.2f$"%pr3_r2)
plt.legend()
plt.show(block=False)
plt.pause(1)
plt.close()


X_log=np.log(X)
y_sqrt=np.sqrt(y)

X_fit=np.arange(X_log.min()-1,X_log.max()+1,1)[:,np.newaxis]
regr=lr.fit(X_log,y_sqrt)
y_lin_fit=regr.predict(X_fit)
lin_r2=r2_score(y_sqrt,regr.predict(X_log))
plt.scatter(X_log,y_sqrt,c='lightgrey',marker='o',label="Training Point")
plt.plot(X_fit,y_lin_fit,color='black',label="Linear($R^2=%.3f$)"%lin_r2)
plt.legend()
plt.show()