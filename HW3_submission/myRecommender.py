from unittest import registerResult
import numpy as np


def my_recommender(rate_mat, lr, with_reg):
    """

    :param rate_mat:
    :param lr:
    :param with_reg:
        boolean flag, set true for using regularization and false otherwise
    :return:
    """

    # TODO pick hyperparams
    max_iter = 1000
    learning_rate = 0.0005

    if with_reg == True:
        reg_coef = 3
    else:
        reg_coef = 0

    n_user, n_item = rate_mat.shape[0], rate_mat.shape[1]

    U = np.random.rand(n_user, lr) / lr
    V = np.random.rand(n_item, lr) / lr

    # TODO implement your code here


    
    M = rate_mat
    mask = M > 0

    epsilon = 0.01

    for _ in range(max_iter):
        # update learning rate

        X = M - np.matmul(U, V.T) * mask
        U_grad = -2 * np.matmul(X, V) + 2 * reg_coef * U
        V_grad = -2 * np.matmul(X.T, U) + 2 * reg_coef * V
        U = U - learning_rate * U_grad
        V = V - learning_rate * V_grad
        
        if np.max([np.linalg.norm(U_grad), np.linalg.norm(V_grad)]) < epsilon:
            break

        # stoping criteria

    return U, V