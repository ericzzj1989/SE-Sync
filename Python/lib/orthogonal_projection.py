import numpy as np
from scipy.sparse.linalg import lsqr, spsolve

def orthogonal_projection(X, problem_data, Cholesky=True):
    """
    This function computes and returns the orthogonal projection of X onto
    ker(A * Omega^(1/2), using either the Cholesky factor L for the reduced
    Laplacian L(W^tau) or by applying an orthogonal (QR) decomposition for
    Omega^(1/2) * Ared'

    :param X: Input matrix
    :param problem_data: Problem data containing necessary matrices
    :param Cholesky: Boolean indicating whether to use Cholesky decomposition or not
    :return: Orthogonal projection of X
    """

    if Cholesky:
        # Compute the projection using a sequence of matrix multiplications and
        # upper-triangular solves

        P1 = np.matmul(problem_data["sqrt_Omega_AredT"].t(), X)
        P2 = np.linalg.solve(problem_data["L"], P1)
        P3 = np.linalg.solve(problem_data["L"].t(), P2)
        P4 = np.matmul(problem_data["sqrt_Omega_AredT"], P3)

        PiX = X - P4

    else:
        # Use QR decomposition
        # print(f'omega: ', problem_data["sqrt_Omega_AredT"].shape)
        # print(f'X: ', X.shape)
        vstar = np.linalg.lstsq(problem_data["sqrt_Omega_AredT"].toarray(), X, rcond=None)[0]
        # vstar = lsqr(problem_data["sqrt_Omega_AredT"], X)
        # vstar = spsolve(problem_data["sqrt_Omega_AredT"], X)
        # PiX = X - np.matmul(problem_data["sqrt_Omega_AredT"], vstar)
        # print(f'vstar: ', vstar.shape)
        PiX = X - (problem_data["sqrt_Omega_AredT"].toarray()).dot(vstar)

    return PiX