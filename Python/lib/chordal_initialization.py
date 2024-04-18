import numpy as np
from scipy.sparse.linalg import lsqr

from lib.construct_B_matrices import construct_B_matrices
from lib.project_to_SOd import project_to_SOd


def chordal_initialization(measurements, nargout_flag = 1):
    # Function to compute chordal initialization for SE-Sync problem
    measurements['edges'] = np.array(measurements['edges'])

    # print('measurements[edges]: ', measurements['edges'].shape[0])
    
    d = len(measurements['t'][0])
    n = np.max(np.max(measurements['edges']))
    m = measurements['edges'].shape[0]

    if nargout_flag > 1:
        B3, B2, B1 = construct_B_matrices(measurements, 3)
    else:
        B3 = construct_B_matrices(measurements)

    # First, estimate the rotations using only the rotational observations
    Id = np.eye(d)
    Id_vec = Id.flatten().reshape(d**2, 1)

    # Compute the constant vector cR induced by fixing the first orientation
    # estimate to be the identity Id.
    # cR = np.dot(B3[:, :d**2], Id_vec)
    B3_9 = B3[slice(None), slice(0, d**2)]
    # cR = np.dot(B3_9.toarray(), Id_vec)
    cR = B3_9.dot(Id_vec)

    # Compute an estimate of the remaining rotations by solving the resulting
    # least squares problem without enforcing the constraint that the
    # estimates lie in SO(d)
    # r2vec = -np.linalg.lstsq(B3[:, d**2:], cR, rcond=None)[0]
    # print(f'test: {B3.tocsr()[slice(None), slice(d**2, None)].shape}')
    # print(f'cR: {cR.shape}')
    B3_submatrix = B3[slice(None), slice(d**2, None)]
    # r2vec = -np.linalg.lstsq(B3_submatrix.toarray(), cR, rcond=None)[0]
    r2vec = -lsqr(B3_submatrix, cR)[0]
    r2vec = r2vec.reshape(-1, 1)
    rvec = np.concatenate((Id_vec, r2vec))
    R_LS = np.reshape(rvec, (d, d*n), order='F')

    # Now reproject these estimates onto SO(d)
    Rchordal = np.zeros((d, d*n))
    for i in range(n):
        Rchordal[:, d*i:d*(i+1)] = project_to_SOd(R_LS[:, d*i:d*(i+1)])

    if nargout_flag > 1:
        # Solve for the translations in terms of the rotations
        # Constant vector induced by fixing R
        cT = B2.dot(Rchordal.flatten())

        # Solve for t_2, ... t_n assuming that t_1 = 0
        B1_submatrix = B1[slice(None), slice(d, None)]
        # t2 = -np.linalg.lstsq(B1[:, d:], cT, rcond=None)[0]
        t2 = -lsqr(B1_submatrix, cT)[0]
        t2 = t2.reshape(-1, 1)
        print(f't2 shape: {t2.shape}')
        tvec = np.concatenate((np.zeros((d, 1)), t2), axis=1)
        tchordal = np.reshape(tvec, (d, n), order='F')

        print(f'tchordal shape: {tchordal}')
        
        return Rchordal, tchordal
    else:
        return Rchordal