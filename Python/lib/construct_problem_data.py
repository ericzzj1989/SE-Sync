import numpy as np
from scipy.sparse import spdiags, csr_matrix
from scipy.linalg import cholesky
import time

from lib.construct_connection_Laplacian import construct_connection_Laplacian
from lib.construct_incidence_matrix import construct_incidence_matrix
from lib.construct_translational_matrices import construct_translational_matrices
from lib.construct_V_matrix import construct_V_matrix

def construct_problem_data(measurements):
    # Set additional variables
    measurements['edges'] = np.array(measurements['edges'])

    problem_data_d = len(measurements['t'][0])
    problem_data_n = np.max(np.max(measurements['edges']))
    problem_data_m = measurements['edges'].shape[0]

    # Construct connection Laplacian for the rotational measurements
    t = time.time()
    problem_data_ConLap = construct_connection_Laplacian(measurements)
    t = time.time() - t
    print('Constructed rotational connection Laplacian in {} seconds'.format(t))

    # Construct the oriented incidence matrix for the underlying directed graph
    # of measurements
    t = time.time()
    problem_data_A = construct_incidence_matrix(measurements)
    t = time.time() - t
    print('Constructed oriented incidence matrix in {} seconds'.format(t))

    # Construct the reduced oriented incidence matrix
    problem_data_Ared = problem_data_A[:problem_data_n-1, :]

    # Construct translational observation and measurement precision matrices
    t = time.time()
    T, Omega = construct_translational_matrices(measurements)
    V = construct_V_matrix(measurements)
    t = time.time() - t
    print('Constructed translational observation and measurement precision matrices in {} seconds'.format(t))

    problem_data_T = T
    problem_data_Omega = Omega
    problem_data_V = V

    # Construct the Laplacian for the translational weight graph
    t = time.time()
    LWtau = problem_data_A.dot(problem_data_Omega).dot(problem_data_A.T)
    t = time.time() - t
    print('Constructed Laplacian for the translational weight graph in {} seconds'.format(t))
    problem_data_LWtau = LWtau

    # Construct the Cholesky factor for the reduced translational weight graph Laplacian
    t = time.time()
    L = cholesky(LWtau[:-1, :-1].toarray(), lower=True)
    problem_data_L = csr_matrix(L)
    t = time.time() - t
    print('Computed lower-triangular factor of reduced translational weight graph Laplacian in {} seconds'.format(t))

    # Cache a couple of various useful products
    print('Caching additional product matrices ...')

    problem_data_sqrt_Omega = spdiags(np.sqrt(Omega.diagonal()), 0, problem_data_m, problem_data_m)
    problem_data_sqrt_Omega = problem_data_sqrt_Omega.tocsr()
    problem_data_sqrt_Omega_AredT = problem_data_sqrt_Omega.dot(problem_data_Ared.T)
    problem_data_sqrt_Omega_T = problem_data_sqrt_Omega.dot(problem_data_T)

    print(f'd: ', problem_data_d)
    print(f'n: ', problem_data_n)
    print(f'm: ', problem_data_m)
    print(f'ConLap: ', problem_data_ConLap.shape)
    print(f'A: ', problem_data_A.shape)
    print(f'Ared: ', problem_data_Ared.shape)
    print(f'T: ', problem_data_T.shape)
    print(f'Omega: ', problem_data_Omega.shape)
    print(f'V: ', problem_data_V.shape)
    print(f'LWtau: ', problem_data_LWtau.shape)
    print(f'L: ', problem_data_L.shape)
    print(f'sqrt_Omega: ', problem_data_sqrt_Omega.shape)
    print(f'sqrt_Omega_AredT: ', problem_data_sqrt_Omega_AredT.shape)
    print(f'sqrt_Omega_T: ', problem_data_sqrt_Omega_T.shape)

    problem_data = {
        'd': problem_data_d,
        'n': problem_data_n,
        'm': problem_data_m,
        'ConLap': problem_data_ConLap,
        'A': problem_data_A,
        'Ared': problem_data_Ared,
        'T': problem_data_T,
        'Omega': problem_data_Omega,
        'V': problem_data_V,
        'LWtau': problem_data_LWtau,
        'L': problem_data_L,
        'sqrt_Omega': problem_data_sqrt_Omega,
        'sqrt_Omega_AredT': problem_data_sqrt_Omega_AredT,
        'sqrt_Omega_T': problem_data_sqrt_Omega_T
    }

    return problem_data