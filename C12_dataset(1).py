import numpy as np
mnist=np.load('mnist_scaled.npz')
print(mnist.files)
X_train,y_train,X_test,y_test=[mnist[f] for f in mnist.files]

from C12_neuralnet import NeuralNetMLP
nn=NeuralNetMLP(n_hidden=100,l2=0.01,epochs=200,eta=0.0005,minibatch_size=100,suffle=True,seed=1)
nn.fit(X_train[:55000],y_train[:55000],X_train[55000:],y_train[55000:])