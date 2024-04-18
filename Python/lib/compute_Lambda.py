import numpy as np
from .Qproduct import Qproduct


def compute_Lambda(Yopt, problem_data, use_Cholesky=True):
    """
    Given an estimated local minimum Yopt for the (possibly lifted) relaxation,
    this function computes and returns the block-diagonal elements of the
    corresponding Lagrange multiplier.

    :param Yopt: Estimated local minimum
    :param problem_data: Problem data containing necessary matrices
    :param use_Cholesky: Boolean indicating whether to use Cholesky decomposition or not
    :return: Block-diagonal elements of the Lagrange multiplier
    """

    Lambda_blocks = np.zeros((problem_data["d"], problem_data["d"] * problem_data["n"]))

    QYt = Qproduct(Yopt.T, problem_data, use_Cholesky)

    for k in range(1, problem_data["n"] + 1):
        imin = problem_data["d"] * (k - 1)
        imax = problem_data["d"] * k
        B = QYt[imin:imax, :] * Yopt[:, imin:imax]
        Lambda_blocks[:, imin:imax] = 0.5 * (B + B.T)

    return Lambda_blocks