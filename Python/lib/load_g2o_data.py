import numpy as np
from scipy.linalg import sqrtm

def load_g2o_data(g2o_data_file):
    measurements = {'edges': [], 'R': [], 't': [], 'kappa': [], 'tau': []}
    edge_id = 0

    with open(g2o_data_file, 'r') as fid:
        for read_line in fid:
            tokens = read_line.split()
            
            if tokens[0] == 'EDGE_SE3:QUAT':
                edge_id += 1
                
                id1, id2, dx, dy, dz, dqx, dqy, dqz, dqw, \
                I11, I12, I13, I14, I15, I16, \
                I22, I23, I24, I25, I26, \
                I33, I34, I35, I36, \
                I44, I45, I46, \
                I55, I56, \
                I66 = map(float, tokens[1:])
                
                # Store the connectivity of this edge
                measurements['edges'].append([int(id1) + 1, int(id2) + 1])
                
                # Store the translational measurement
                measurements['t'].append(np.array([dx, dy, dz]))
                
                # Reconstruct quaternion for relative measurement
                q = np.array([dqw, dqx, dqy, dqz])
                q /= np.linalg.norm(q)
                
                # Compute and store corresponding rotation matrix
                R = quat2rot(q)
                measurements['R'].append(R)
                
                # Reconstruct the information matrix
                measurement_info = np.array([[I11, I12, I13, I14, I15, I16],
                                             [I12, I22, I23, I24, I25, I26],
                                             [I13, I23, I33, I34, I35, I36],
                                             [I14, I24, I34, I44, I45, I46],
                                             [I15, I25, I35, I45, I55, I56],
                                             [I16, I26, I36, I46, I56, I66]])
                
                # Compute and store the optimal value of the parameter tau
                tau = 3 / np.trace(np.linalg.inv(measurement_info[:3, :3]))
                measurements['tau'].append(tau)
                
                # Extract and store the optimal value of the parameter kappa
                kappa = 3 / (2 * np.trace(np.linalg.inv(measurement_info[3:, 3:])))
                measurements['kappa'].append(kappa)
                
            elif tokens[0] == 'EDGE_SE2':
                edge_id += 1
                
                id1, id2, dx, dy, dth, I11, I12, I13, I22, I23, I33 = map(float, tokens[1:])
                
                # Store the connectivity of this edge
                measurements['edges'].append([int(id1) + 1, int(id2) + 1])
                
                # Store the translational measurement
                measurements['t'].append(np.array([dx, dy]))
                
                # Reconstruct and store the rotational measurement
                R = np.array([[np.cos(dth), -np.sin(dth)],
                              [np.sin(dth), np.cos(dth)]])
                measurements['R'].append(R)
                
                # Reconstruct the information matrix
                measurement_info = np.array([[I11, I12, I13],
                                             [I12, I22, I23],
                                             [I13, I23, I33]])
                
                # Extract and store an outer approximation for the translational measurement precision
                tau = 2 / np.trace(np.linalg.inv(measurement_info[:2, :2]))
                measurements['tau'].append(tau)
                
                # Extract and store an outer approximation for the rotational measurement precision
                kappa = I33
                measurements['kappa'].append(kappa)
                
    return measurements

def quat2rot(q):
    qw, qx, qy, qz = q
    R = np.array([[1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
                  [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
                  [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]])
    return R
