import numpy as np
class Perception(object):
    def __init__(self,eta=0.01,n_iter=50,random_state=1):
        self.eta=eta
        self.n_iter=n_iter
        self.random_state=random_state

    def fit(self,X,y):
        rgen=np.random.RandomState(self.random_state)
        self.w_=rgen.normal(loc=0.0,scale=0.01,size=1+X.shape[1])
        # print(self.w_)
        self.error_=[]
        for _ in range(self.n_iter):
            error=0
            for xi,target in zip(X,y):
                update=self.eta * (target - self.predict(xi))
                self.w_[1:]+=update*xi
                self.w_[0]+=update
                error += int(update != 0.0)
            self.error_.append(error)
        return self

    def net_input(self,X):
        return np.dot(X,self.w_[1:]) + self.w_[0]

    def predict(self,X):
        return np.where(self.net_input(X) >= 0.0,1,-1)




