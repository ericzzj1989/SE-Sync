import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('./lib')
from load_g2o_data import load_g2o_data
from chordal_initialization import chordal_initialization
from plot_poses import plot_poses
from save_results import save_results
from SE_Sync import SE_Sync

# Reset environment
np.random.seed(0)
plt.close('all')

# Import SE-Sync (assuming it's a Python module)
# from import_SE_Sync import run_import_SE_Sync

# Select dataset to run
data_dir = '../data/'  # Relative path to directory containing example datasets

# 3D datasets
sphere2500 = 'sphere2500'
torus = 'torus3D'
grid = 'grid3D'
garage = 'parking-garage'
cubicle = 'cubicle'
rim = 'rim'

# 2D datasets
CSAIL = 'CSAIL'
manhattan = 'manhattan'
city10000 = 'city10000'
intel = 'intel'
ais = 'ais2klinik'

# Pick the dataset to run here
file = garage

g2o_file = data_dir + file + '.g2o'

# Read in .g2o file
print('Loading file: %s ...' % g2o_file)
measurements = load_g2o_data(g2o_file)
# print(measurements['R'][10])
num_poses = int(np.max(np.max(measurements['edges'])))
num_measurements = len(measurements['kappa'])
d = len(measurements['t'][0])
print('Number of poses:', num_poses)
print('Number of measurements:', num_measurements)

# Set Manopt options (if desired)
Manopt_opts = {
    'tolgradnorm': 1e-2,
    'rel_func_tol': 1e-5,
    'miniter': 1,
    'maxiter': 300,
    'maxinner': 500
}

# Set SE-Sync options (if desired)
SE_Sync_opts = {
    'r0': 5,
    'rmax': 10,
    'eig_comp_rel_tol': 1e-4,
    'min_eig_lower_bound': -1e-3,
    'Cholesky': False
}

use_chordal_initialization = True

# Run SE-Sync
print('Computing chordal initialization...')
R = chordal_initialization(measurements)
print(f"main R: ", R.shape)
Y0 = np.vstack((R, np.zeros((SE_Sync_opts['r0'] - d, num_poses * d))))
# Y0 = Y0.reshape(-1, num_poses, d)
# print(f'Y0: {Y0}')
# print(f'main: Y0: {Y0.shape}')
SDPval, Yopt, xhat, Fxhat, SE_Sync_info, problem_data = SE_Sync(measurements, Manopt_opts, SE_Sync_opts, Y0)

# Plot resulting solution
plot_loop_closures = True

if plot_loop_closures:
    plot_poses(xhat['t'], xhat['R'], measurements['edges'], '-b', .25)
else:
    plot_poses(xhat['t'], xhat['R'])
plt.axis('tight')
plt.show()

# Save results
save_results(xhat['t'], xhat['R'], 'se_sync_estimated_rim.g2o')