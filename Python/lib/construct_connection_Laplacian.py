import numpy as np
from scipy.sparse import csr_matrix

from lib.rcvize_matrix import rcvize_matrix

def construct_connection_Laplacian(measurements):
    D = len(measurements['t'][0])  # Dimension of SE(d)
    N = np.max(np.max(measurements['edges'])) # Number of nodes in the pose graph
    M = measurements['edges'].shape[0]  # Number of edges in the pose graph

    D2 = D ** 2
    off_diag_inc = 2 * D2
    NNZ = off_diag_inc * M + D * N

    rows = np.zeros(NNZ, dtype=int)
    cols = np.zeros(NNZ, dtype=int)
    vals = np.zeros(NNZ)

    degs = np.zeros(N)  # Vector to store the degrees of the nodes

    for k in range(M):
        i = measurements['edges'][k, 0]  # Node that this edge leaves
        j = measurements['edges'][k, 1]  # Node that this edge enters
        Rij = measurements['R'][k]  # Rotation matrix for this observation
        kappa = measurements['kappa'][k]  # Precision for this rotational observation

        degs[i - 1] += kappa
        degs[j - 1] += kappa

        r, c, Rvect = rcvize_matrix(Rij, i, j)

        rows[off_diag_inc * k : off_diag_inc * k + D2] = r
        cols[off_diag_inc * k : off_diag_inc * k + D2] = c
        vals[off_diag_inc * k : off_diag_inc * k + D2] = -kappa * Rvect * np.ones(D2)

        r, c, Rvect = rcvize_matrix(np.transpose(Rij), j, i)

        rows[off_diag_inc * k + D2 : off_diag_inc * k + off_diag_inc] = r
        cols[off_diag_inc * k + D2 : off_diag_inc * k + off_diag_inc] = c
        vals[off_diag_inc * k + D2 : off_diag_inc * k + off_diag_inc] = -kappa * Rvect * np.ones(off_diag_inc-D2)

    # for d in range(D):
    #     rows[off_diag_inc * M + d * N : off_diag_inc * M + (d + 1) * N] = np.arange(1, N + 1)
    #     cols[off_diag_inc * M + d * N : off_diag_inc * M + (d + 1) * N] = np.arange(1, N + 1)
    #     vals[off_diag_inc * M + d * N : off_diag_inc * M + (d + 1) * N] = degs
    rows[off_diag_inc * M : NNZ] = np.arange(0, D*N)
    cols[off_diag_inc * M : NNZ] = np.arange(0, D*N)
    vals[off_diag_inc * M : NNZ] = np.kron(degs, np.ones(D))

    Lrho = csr_matrix((vals, (rows, cols)), shape=(D * N, D * N))
    return Lrho