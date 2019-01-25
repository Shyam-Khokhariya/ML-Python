from scipy.spatial.distance import squareform,pdist
from scipy import exp
from scipy.linalg import eigh
import numpy as np

def rbf_kernel_pca(X, gamma, n_component):
    """

    :param X: {numpy ndarray} shape: {n_sample,n_features}
    :param gamma:float
                 tuning parameter for RBF Kernel
    :param n_component:int
                 return number of principle component
    :return:
    X_pc: {numpy ndarray} shape:{n_sample,n_features}
        Projected Dataset
    """

    #Calculate pairwise Euclidean Distance
    sq_dis=pdist(X,'sqeuclidean')

    #Convert into square matrix
    mat_sq_dis=squareform(sq_dis)

    #Compute symmetric Kernel matrix
    K=exp(-gamma*mat_sq_dis)

    #Center of Kernel Matrix
    N=K.shape[0]

    one_np = np.ones((N,N))/N
    K=K-one_np.dot(K)-K.dot(one_np)+one_np.dot(K).dot(one_np)

    #obtaining eigen pair from center kernel matrix
    #scipy.linalg.eig return in ascending order
    eigen_val,eigen_vec=eigh(K)
    eigen_val , eigen_vec = eigen_val[::-1] , eigen_vec[:,::-1]

    #collecting top k elements
    X_pc=np.column_stack((eigen_vec[:,i] for i in range(n_component)))

    return X_pc