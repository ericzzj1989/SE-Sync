from .Qproduct import Qproduct


def Euclidean_Hessian_vector_product(Y, Ydot, problem_data, use_Cholesky):
    """
    This function computes and returns the value of the Euclidean Hessian at
    the point Y evaluated along the tangent direction Ydot.

    :param Y: Point in the Euclidean space
    :param Ydot: Tangent direction at point Y
    :param problem_data: Problem data containing necessary matrices
    :param use_Cholesky: Boolean indicating whether to use Cholesky decomposition or not
    :return: Euclidean Hessian vector product
    """

    Hvec = 2 * Qproduct(Ydot.T, problem_data, use_Cholesky).T

    return Hvec