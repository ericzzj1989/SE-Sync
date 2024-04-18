from .orthogonal_projection import orthogonal_projection

def Qproduct(X, problem_data, use_Cholesky=True):
    """
    This function computes and returns the matrix product P = Q*X, where

    Q = L(G^rho) + Q^tau

    is the quadratic form that defines the objective function for the
    pose-graph optimization problem.
    """
    if use_Cholesky is None:
        use_Cholesky = True

    # Compute the translational term Qtau
    # Qtau = T' * Omega^(1/2) * Pi * Omega^(1/2) * T * X
    # Avoid storing large dense matrices, compute this product associatively from right to left
    print('sqrt_Omega_T: ', problem_data['sqrt_Omega_T'].T.shape)
    print('X: ', X.shape)
    print('orth: ', orthogonal_projection(problem_data['sqrt_Omega_T'] @ X, problem_data, use_Cholesky).shape)
    QtauX = problem_data['sqrt_Omega_T'].T @ orthogonal_projection(problem_data['sqrt_Omega_T'] @ X, problem_data, use_Cholesky)

    QX = problem_data['ConLap'] @ X + QtauX

    return QX