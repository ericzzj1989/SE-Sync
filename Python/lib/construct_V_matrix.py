import numpy as np
from scipy.sparse import csr_matrix

def construct_V_matrix(measurements):
    D = len(measurements['t'][0])  # Dimension of SE(d)
    N = np.max(np.max(measurements['edges']))  # Number of nodes in the pose graph
    M = measurements['edges'].shape[0]  # Number of edges in the pose graph

    # Number of non-zero elements in V
    NNZ = D * M + D * N

    rows = np.zeros(NNZ)
    cols = np.zeros(NNZ)
    vals = np.zeros(NNZ)

    for e in range(M):
        i = measurements['edges'][e, 0]  # The node that this edge leaves
        j = measurements['edges'][e, 1]  # The node that this edge enters
        tij = measurements['t'][e]
        tau_ij = measurements['tau'][e]

        # for k in range(D):
        #     cmin = D * e + k
        #     cmax = D * e + D
        #     rows[cmin:cmax] = j
        #     cols[cmin:cmax] = D * (i - 1) + np.arange(D)
        #     vals[cmin:cmax] = -tau_ij * tij.T
        cmin = D * e
        cmax = D * e + D
        rows[cmin:cmax] = (j - 1) * np.ones(D)
        cols[cmin:cmax] = np.arange(D * (i - 1), D * i)
        vals[cmin:cmax] = -tau_ij * tij.T * np.ones(D)

        cmin = D * M + D * (i - 1)
        cmax = D * M + D * i
        vals[cmin:cmax] += tau_ij * tij.T * np.ones(D)  # Add observation to the weighted sum

    for i in range(N):
        cmin = D * M + D * i
        cmax = D * M + D * (i + 1)
        rows[cmin:cmax] = i * np.ones(D)
        cols[cmin:cmax] = np.arange(D * i, D * i + D)

    # print(f'rows: {rows}')
    # return

    V = csr_matrix((vals, (rows, cols)), shape=(N, D * N))
    return V