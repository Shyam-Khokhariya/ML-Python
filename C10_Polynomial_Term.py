import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
X=np.array([259,270,300,345,385,425,462,475,510,534])[:,np.newaxis]
y=np.array([215,220,248,255,260,275,256,275,285,295])
lr=LinearRegression()
pr=LinearRegression()
quadratic=PolynomialFeatures(degree=2)
X_quad=quadratic.fit_transform(X)
lr.fit(X,y)
X_fit=np.arange(250,600,10)[:,np.newaxis]
y_fit=lr.predict(X_fit)
pr.fit(X_quad,y)
y_quad_fit=pr.predict(quadratic.fit_transform(X_fit))
plt.scatter(X,y,label='training point')
plt.plot(X_fit,y_fit,label='Linear Fit',linestyle='--')
plt.plot(X_fit,y_quad_fit,label="Poly Fit")
plt.legend()
plt.show(block=False)
plt.pause(1)
plt.close()

y_pred=lr.predict(X)
y_quad_pred=pr.predict(X_quad)
from sklearn.metrics import mean_squared_error,r2_score
print("Training MSE Linear: %.3f, quadratic: %.3f"%(mean_squared_error(y,y_pred),mean_squared_error(y,y_quad_pred)))
print("Training R^2 Linear: %.3f, quadratic: %.3f"%(r2_score(y,y_pred),r2_score(y,y_quad_pred)))


