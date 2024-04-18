import numpy as np
from .project_to_SOd import project_to_SOd


def round_solution(Yopt, problem_data):
    """
    Round the solution Yopt to an element of SO(d)^n.

    :param Yopt: Solution in St(d, r)^n
    :param problem_data: Problem data
    :return: R: Rounded solution in SO(d)^n
             singular_values: Singular values of Yopt
             determinants: Determinants of R subblocks
    """
    
    r = Yopt.shape[0]
    
    U, Xi, V = np.linalg.svd(Yopt, full_matrices=False)
    singular_values = np.diag(Xi)
    
    Xi_d = Xi[:problem_data['d'], :problem_data['d']]   # Xi_d is the upper-left dxd submatrix of Xi
    V_d = V[:, :problem_data['d']]  # V_d contains the first d columns of V
    
    R = np.dot(Xi_d, V_d.T)
    
    determinants = np.zeros(problem_data['n'])
    
    for k in range(problem_data['n']):
        submatrix = R[:, problem_data['d'] * k: problem_data['d'] * (k + 1)]
        determinants[k] = np.linalg.det(submatrix)
    
    ng0 = np.sum(determinants > 0)
    
    reflector = np.diag(np.concatenate((np.ones(problem_data['d'] - 1), [-1]))) # Orthogonal matrix that we can use for reversing the orientations of the orthogonal matrix subblocks of R
    
    if ng0 == 0:
        # This solution converged to a reflection of the correct solution
        R = np.dot(reflector, R)
        determinants = -determinants
    elif ng0 < problem_data['n']:
        print('WARNING: SOLUTION HAS INCONSISTENT ORIENTATIONS!')

        # If more than half of the determinants have negative sign, reverse them
        if ng0 < problem_data['n'] / 2:
            determinants = -determinants
            R = np.dot(reflector, R)
    
    # Finally, project each element of R to SO(d)
    for i in range(problem_data['n']):
        submatrix = R[:, problem_data['d'] * i: problem_data['d'] * (i + 1)]
        R[:, problem_data['d'] * i: problem_data['d'] * (i + 1)] = project_to_SOd(submatrix)
    
    return R, singular_values, determinants