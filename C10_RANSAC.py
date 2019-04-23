import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("/home/ghanshyam/Machine Learning/Housing.csv")
df.columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

X,y=df[['RM']].values,df['MEDV'].values

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
ransac=RANSACRegressor(LinearRegression(),max_trials=100,min_samples=50,loss='absolute_loss',residual_threshold=5.0,random_state=0)
ransac.fit(X,y)

inlier_mask=ransac.inlier_mask_
outlier_mask=np.logical_not(inlier_mask)
line_X=np.arange(3,10,1)
line_y_ransac=ransac.predict(line_X[:,np.newaxis])
plt.scatter(X[inlier_mask],y[inlier_mask],c="Green",marker='o',edgecolors="white",label="Inlier")
plt.scatter(X[outlier_mask],y[outlier_mask],c="blue",marker='+',label="Outlier")
plt.plot(line_X,line_y_ransac,color="black",lw=2)
plt.show(block=False)
plt.pause(1)
plt.close()

print("Slope:%.3f"%ransac.estimator_.coef_[0])
print("Intercept:%.3f"%ransac.estimator_.intercept_)




from sklearn.model_selection import train_test_split
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.3)
slr=LinearRegression()
slr.fit(X_train,y_train)
y_train_pred=slr.predict(X_train)
y_test_pred=slr.predict(X_test)
plt.scatter(y_train_pred,y_train_pred-y_train,c="steelblue",marker='o',edgecolors="white",label="Training Data")
plt.scatter(y_test,y_test_pred-y_test,c="green",marker="+",label="Test Data")
plt.xlabel("Predicted Value")
plt.ylabel("Residuals")
plt.legend()
plt.hlines(y=0,xmin=-10,xmax=50,colors="black",lw=2)
plt.show()

from sklearn.metrics import mean_squared_error
print("Mean Squre Error Train:%.3f\nTest:%.3f"%(mean_squared_error(y_train,y_train_pred)
                                                ,mean_squared_error(y_test,y_test_pred)))

from sklearn.metrics import r2_score
print("R^2 Train:%.3f   \n     Test:%.3f"%(r2_score(y_train,y_train_pred),r2_score(y_test,y_test_pred)))


from sklearn.linear_model import Ridge,Lasso,ElasticNet
ridge=Ridge(alpha=1.0)
lasso=Lasso(alpha=1.0)
elanet=ElasticNet(alpha=1.0,l1_ratio=0.5)