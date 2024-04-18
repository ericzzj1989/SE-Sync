import numpy as np
from scipy.spatial.transform import Rotation as R

def save_results(t_hat, Rhat, filename):
    num_poses = t_hat.shape[1]
    
    # Open file for writing
    with open(filename, 'w') as fileID:
        # Iterate through poses and write translation and quaternion to file
        for index in range(num_poses):
            t = t_hat[:, index]
            R_mat = Rhat[:, (index * 3):((index + 1) * 3)]
            quat = R.from_matrix(R_mat).as_quat()

            # Write data with identifier and index to file
            fileID.write(f"VERTEX_SE3:QUAT {index} ")  # Writing index starting from 0

            # Write translation to file
            fileID.write(' '.join(map(str, t)) + ' ')

            # Write quaternion to file
            fileID.write(' '.join(map(str, quat[1:])) + ' ' + str(quat[0]))

            # Add newline at the end of each pose's data
            fileID.write('\n')