from .orthogonal_projection import orthogonal_projection

def Qproduct(X, problem_data, use_Cholesky=False, verbose=False):
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
    if verbose == True:
        print('sqrt_Omega_T: ', problem_data['sqrt_Omega_T'].shape)
        print('X: ', X)
        # print('sqrt_Omega_T * X: ', problem_data['sqrt_Omega_T'] @ X)
        
    # print('orth: ', orthogonal_projection(problem_data['sqrt_Omega_T'] @ X, problem_data, use_Cholesky).shape)
    QtauX = problem_data['sqrt_Omega_T'].T @ orthogonal_projection(problem_data['sqrt_Omega_T'] @ X, problem_data, use_Cholesky, verbose)

    QX = problem_data['ConLap'] @ X + QtauX

    return QX