import numpy as np


def project_to_SOd(M):
    """
    Given a square d x d matrix M, this function computes and returns a 
    closest matrix R belonging to SO(d).
    
    Args:
    - M (numpy.ndarray): Input square matrix of shape (d, d).
    
    Returns:
    - R (numpy.ndarray): Closest matrix belonging to SO(d).
    """
    # d = M.shape[0]
    # U, S, Vt = np.linalg.svd(M)
    # R = U @ np.diag(np.concatenate((np.ones(d-1), [np.sign(np.linalg.det(U @ Vt))]))) @ Vt
    d = M.shape[0]
    
    U, S, Vt = np.linalg.svd(M)
    V = Vt.T
    
    det_UVt = np.linalg.det(U @ Vt)
    sign_det = np.sign(det_UVt)
    
    diag_values = np.ones(d-1)
    diag_values = np.append(diag_values, sign_det)
    diag_matrix = np.diag(diag_values)
    
    R = U @ diag_matrix @ Vt
    return R