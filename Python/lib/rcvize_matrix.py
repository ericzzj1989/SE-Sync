import numpy as np

def rcvize_matrix(M, i, j):
    """
    Given a DxD matrix M that we wish to insert into the (i,j)th DxD
    block of a sparse matrix, this function computes and returns the
    corresponding {rows, cols, vals} vectors describing this block.
    
    Parameters:
        M (numpy.ndarray): The DxD matrix to be inserted.
        i (int): The row index of the block.
        j (int): The column index of the block.
    
    Returns:
        rows (numpy.ndarray): The row indices of the block.
        cols (numpy.ndarray): The column indices of the block.
        vals (numpy.ndarray): The values of the block.
    """
    D = M.shape[0]

    vals = M.T.reshape(1, -1)  # Vectorize M by concatenating its columns

    rows = np.tile(np.arange(D*(i-1), D*i), D)
    cols = np.kron(np.arange(D*(j-1), D*j), np.ones(D))

    return rows, cols, vals