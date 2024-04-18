from .Qproduct import Qproduct


def Euclidean_gradient(Y, problem_data, use_Cholesky=True):
    """
    This function computes and returns the value of the Euclidean gradient of
    the objective function: nabla F(Y) = 2YQ.

    :param Y: Input matrix
    :param problem_data: Problem data containing necessary matrices
    :param use_Cholesky: Boolean indicating whether to use Cholesky decomposition or not
    :return: Euclidean gradient of the objective function
    """

    egrad = 2 * Qproduct(Y.T, problem_data, use_Cholesky).T

    return egrad