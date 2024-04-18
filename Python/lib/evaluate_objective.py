import numpy as np
from .Qproduct import Qproduct


def evaluate_objective(Y, problem_data, use_Cholesky=True):
    Yt = Y.T
    YQ = Qproduct(Yt, problem_data, use_Cholesky).T
    trQYtY = np.trace(YQ @ Yt)
    return trQYtY, YQ