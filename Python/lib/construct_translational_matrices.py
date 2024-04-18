import numpy as np
from scipy.sparse import csr_matrix, spdiags

def construct_translational_matrices(measurements):
    """
    This function computes and returns the matrix T containing the raw
    translational measurements and the diagonal matrix Omega containing
    the translational measurement precisions on its main diagonal.

    Args:
    - measurements: dict containing measurements information

    Returns:
    - T: scipy.sparse.csr_matrix, the matrix containing the raw translational measurements
    - Omega: scipy.sparse.diags, the diagonal matrix containing the translational measurement precisions
    """
    D = len(measurements['t'][0])       # D = dimension of SE(d)
    N = np.max(np.max(measurements['edges']))  # N = number of nodes in the pose graph
    M = measurements['edges'].shape[0]  # M = number of edges in the pose graph

    # Allocate storage for sparse matrix
    rows = np.zeros(D * M, dtype=int)
    cols = np.zeros(D * M, dtype=int)
    vals = np.zeros(D * M)

    omega = np.zeros(M)

    # Iterate over the measurements in the pose graph
    for e in range(M):
        # Extract measurement data
        k = measurements['edges'][e, 0]  # The node that this edge leaves
        tij = measurements['t'][e]       # The translation corresponding to this observation
        omega[e] = measurements['tau'][e]  # The precision for this translational observation

        # Process measurement data
        rows[D * e : D * (e + 1)] = e * np.ones(D)
        cols[D * e : D * (e + 1)] = np.arange(D * (k-1), D * (k-1) + D)
        vals[D * e : D * (e + 1)] = -tij * np.ones(D)

    T = csr_matrix((vals, (rows, cols)), shape=(M, D * N))
    Omega = spdiags(omega, [0], M, M, format='csr')

    return T, Omega