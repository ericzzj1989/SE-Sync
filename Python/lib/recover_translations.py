import numpy as np

def recover_translations(R, problem_data):
    """
    Recover translations from the rotation matrix R.

    :param R: Rotation matrix
    :param problem_data: Problem data
    :return: Translations matrix
    """

    # V_transpose = problem_data['V'].T
    LWtau_pinv = np.linalg.pinv(problem_data['LWtau'])
    t = -np.dot(LWtau_pinv, np.dot(problem_data['V'], R.T)).T
    return t