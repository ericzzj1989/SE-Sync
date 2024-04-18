import numpy as np
from scipy.sparse import csr_matrix, vstack, coo_matrix, bmat


def construct_B_matrices(measurements, nargout_flag = 1):
    d = len(measurements['t'][0])  # Dimension of observations
    n = np.max(np.max(measurements['edges']))  # Number of poses
    m = measurements['edges'].shape[0]  # Number of measurements

    # B3 matrix:
    B3_nnz = (d**3 + d**2) * m
    B3_rows = np.zeros(B3_nnz, dtype=int)
    B3_cols = np.zeros(B3_nnz, dtype=int)
    B3_vals = np.zeros(B3_nnz)

    for e in range(m):
        
        tail = measurements['edges'][e, 0]
        head = measurements['edges'][e, 1]
        # print(f'tail: ', {tail})
        # print(f'head: ', {head})

        sqkappa = np.sqrt(measurements['kappa'][e])
        Rt = measurements['R'][e].T

        # print('Rt: ', measurements['R'][e])

        for r in range(d):
            for c in range(d):
                # idxs = np.arange((d**3 + d**2) * (e-1) + d**2 * (r-1) + d * (c-1), (d**3 + d**2) * (e-1) + d**2 * (r-1) + d * c + 1)
                # B3_rows[idxs] = np.arange(d**2 * (e-1) + d * (r-1), d**2 * (e-1) + d * r + 1)
                # B3_cols[idxs] = np.arange(d**2 * (tail - 1) + d * (c-1), d**2 * (tail - 1) + d * c + 1)
                # B3_vals[idxs] = -sqkappa * Rt[r-1, c-1]

                idxs = np.arange((d**3 + d**2) * e + d**2 * r + d * c, (d**3 + d**2) * e + d**2 * r + d * c + d)
                B3_rows[idxs] = np.arange(d**2 * e + d * r, d**2 * e + d * r + d)
                B3_cols[idxs] = np.arange(d**2 * (tail-1) + d * c, d**2 * (tail-1) + d * c + d)
                B3_vals[idxs] = -sqkappa * Rt[r, c] * np.ones(d)

        idxs = np.arange(d**3 * (e+1) + d**2 * e, (d**3 + d**2) * (e+1))
        B3_rows[idxs] = np.arange(d**2 * e, d**2 * (e+1))
        B3_cols[idxs] = np.arange(d**2 * (head-1), d**2 * head)
        B3_vals[idxs] = sqkappa * np.ones(d**2)

    B3 = csr_matrix((B3_vals, (B3_rows, B3_cols)), shape=(d**2 * m, d**2 * n))
    # B3 = coo_matrix((B3_vals, (B3_rows, B3_cols)), shape=(d**2*m, d**2*n))

    if nargout_flag > 1:
        # B2 matrix:
        B2_nnz = m * d**2
        B2_rows = np.zeros(B2_nnz, dtype=int)
        B2_cols = np.zeros(B2_nnz, dtype=int)
        B2_vals = np.zeros(B2_nnz)

        for e in range(m):
            tail = measurements['edges'][e, 0]

            sqtau = np.sqrt(measurements['tau'][e])
            tij = measurements['t'][e]

            for idx in range(d):
                # print(f'idx: {idx}')
                # print(f'd: {d}')
                # print(f'e: {e}')
                # print(f'arange: {np.arange(d * e, d * e)}')
                # print(f'minus1: {d**2 * e + d * idx}')
                # print(f'minus2: {d**2 * e + d * (idx+1)}')
                # B2_rows[d**2 * (e-1) + d * (idx-1): d**2 * (e-1) + d * idx + 1] = np.arange(d * (e-1), d * e + 1)
                # B2_cols[d**2 * (e-1) + d * (idx-1): d**2 * (e-1) + d * idx + 1] = np.arange(d**2 * (tail-1) + d * (idx - 1), d**2 * (tail-1) + d * idx + 1)
                # B2_vals[d**2 * (e-1) + d * (idx-1): d**2 * (e-1) + d * idx + 1] = -sqtau * tij[idx-1]

                B2_rows[d**2 * e + d * idx: d**2 * e + d * (idx+1)] = np.arange(d * e, d * (e+1))
                B2_cols[d**2 * e + d * idx: d**2 * e + d * (idx+1)] = np.arange(d**2 * (tail-1) + d * idx, d**2 * (tail-1) + d * (idx+1))
                B2_vals[d**2 * e + d * idx: d**2 * e + d * (idx+1)] = -sqtau * tij[idx] * np.ones(d)

        B2 = csr_matrix((B2_vals, (B2_rows, B2_cols)), shape=(d * m, d**2 * n))
        # B2 = coo_matrix((B2_vals, (B2_rows, B2_cols)), shape=(d * m, d**2 * n))

        # B1 matrix:
        B1_nnz = 2 * d * m
        B1_rows = np.zeros(B1_nnz, dtype=int)
        B1_cols = np.zeros(B1_nnz, dtype=int)
        B1_vals = np.zeros(B1_nnz)

        for e in range(m):
            tail = measurements['edges'][e, 0]
            head = measurements['edges'][e, 1]

            B1_rows[2 * d * e: 2 * d * (e+1) - d] = np.arange(d * e, d * (e+1))
            B1_cols[2 * d * e: 2 * d * (e+1) - d] = np.arange(d * (tail-1), d * tail)
            B1_vals[2 * d * e: 2 * d * (e+1) - d] = -np.sqrt(measurements['tau'][e]) * np.ones(d)

            B1_rows[2 * d * (e+1) - d: 2 * d * (e+1)] = np.arange(d * e, d * (e+1))
            B1_cols[2 * d * (e+1) - d: 2 * d * (e+1)] = np.arange(d * (head-1), d * head)
            B1_vals[2 * d * (e+1) - d: 2 * d * (e+1)] = np.sqrt(measurements['tau'][e]) * np.ones(d)

        B1 = csr_matrix((B1_vals, (B1_rows, B1_cols)), shape=(d * m, d * n))
        # B1 = coo_matrix((B1_vals, (B1_rows, B1_cols)), shape=(d * m, d * n))

        if nargout_flag >= 4:
            zero_block = csr_matrix((B3.shape[0], B1.shape[1]))
            B = bmat([[B1, B2],
            [zero_block, B3]])
            # B = vstack([np.hstack([B1, B2]), np.hstack([coo_matrix((d**3 * m, B1.shape[1])), B3])])

            return B3, B2, B1, B
        
        return B3, B2, B1
    else:
        return B3