import numpy as np
from .Qproduct import Qproduct

def Q_minus_Lambda_product(X, Lambda, problem_data, use_Cholesky=True):
    """
    Compute and return the product (Q - Lambda) * X.

    :param X: Input matrix
    :param Lambda: Lagrange multiplier
    :param problem_data: Problem data containing necessary matrices
    :param use_Cholesky: Boolean indicating whether to use Cholesky decomposition or not
    :return: Product (Q - Lambda) * X
    """
    
    Lambda_X = np.zeros((problem_data['d'] * problem_data['n'], X.shape[1]))
    for i in range(problem_data['n']):
        rmin = problem_data['d'] * i
        rmax = problem_data['d'] * (i + 1)
        Lambda_X[rmin:rmax, :] = np.dot(Lambda[:, rmin:rmax], X[rmin:rmax, :])

    Q_minus_Lambda_X = Qproduct(X, problem_data, use_Cholesky) - Lambda_X

    return Q_minus_Lambda_X