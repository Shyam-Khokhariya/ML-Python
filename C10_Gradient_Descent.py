import C10_Linear_Regression_GD as C10LRGD
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv("/home/ghanshyam/Machine Learning/Housing.csv")
df.columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']


X=df[['RM']].values
y=df['MEDV'].values
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
X_std=sc_x.fit_transform(X)
y_std=sc_y.fit_transform(y[:,np.newaxis]).flatten()
lr=C10LRGD.LinearRegressionGD()
lr.fit(X_std,y_std)

sns.reset_orig()
plt.plot(range(1,lr.n_iter+1),lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.show(block=False)
plt.pause(1)
plt.close()

def lin_regplot(X,y,model):
    plt.scatter(X,y,c='steelblue',edgecolors='white',s=70)
    plt.plot(X,model.predict(X),color='black',lw=2)
    return None

lin_regplot(X_std,y_std,lr)
plt.xlabel('Avarage number of room [RM] (Standardized)')
plt.ylabel('Price in $1000s [MEDV] (Standardized)')
plt.show(block=False)
plt.pause(5)
plt.close()

# num_room_std=sc_x.transform([5.0])
# price_std=lr.predict(num_room_std)
# print("Price in $1000: %.3f"%sc_y.inverse_transform(price_std))

print("Slope:%.3f"%lr.w_[1])
print("Intercept:%.3f"%lr.w_[0])


from sklearn.linear_model import LinearRegression
slr=LinearRegression()
slr.fit(X,y)
print("\n\nLinear Model")
print("Slope:%.3f"%slr.coef_[0])
print("Intercept:%.3f"%slr.intercept_)

lin_regplot(X,y,slr)
plt.xlabel("Average Number of Rooms ")
plt.ylabel("Price in $1000")
plt.show()

Xb=np.hstack((np.ones((X.shape[0],1)),X))
w=np.zeros(X.shape[1])
z=np.linalg.inv(np.dot(Xb.T,Xb))
w=np.dot(z,np.dot(Xb.T,y))
print("\n\nDirect Method Using Matrices....")
print("Slope:%.3f"%w[1])
print("Intercept:%.3f"%w[0])