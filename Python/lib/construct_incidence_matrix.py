import numpy as np
from scipy.sparse import csr_matrix

def construct_incidence_matrix(measurements):
    """
    This function computes and returns the oriented incidence matrix of the
    underlying directed graph of measurements:

    A_{ie} =  -1, if edge e leaves node i,
              +1, if edge e enters node i,
              0, otherwise.

    Args:
    - measurements: dict containing measurements information

    Returns:
    - A: numpy.ndarray, the oriented incidence matrix
    """
    N = np.max(np.max(measurements['edges']))  # Number of nodes in the pose graph
    M = measurements['edges'].shape[0]      # Number of edges in the pose graph

    # Extract out_nodes and in_nodes from measurements
    out_nodes = measurements['edges'][:, 0] - 1  # out_nodes[e] = i if edge e leaves node i
    in_nodes = measurements['edges'][:, 1] - 1   # in_nodes[e] = j if edge e enters node j

    # Construct node_indices, edge_indices, and vals
    node_indices = np.concatenate((out_nodes, in_nodes))
    edge_indices = np.concatenate((np.arange(0, M), np.arange(0, M)))
    vals = np.concatenate((-np.ones(M), np.ones(M)))

    # Create the sparse oriented incidence matrix A
    A = csr_matrix((vals, (node_indices, edge_indices)), shape=(N, M))

    return A