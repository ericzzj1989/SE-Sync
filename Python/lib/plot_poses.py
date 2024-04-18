import numpy as np
import matplotlib.pyplot as plt

def plot_poses(t_hat, Rhat, edges=None, lc_linestyle='-b', lc_alpha=1.0):
    """
    Given translational and rotation state estimates returned by SE_Sync,
    this function plots the corresponding solution, applying the gauge
    symmetry that maps the first pose to the identity element of SE(d). The
    'edges' argument is optional; if supplied, it will plot loop closures;
    otherwise only odometric links are shown. lc_linestyle is an optional
    string specifying the line style to be used for plotting the loop closure
    edges (default: '-b'); similarly, lc_alpha is an optional alpha value for
    the loop closure edges (default: alpha = 1.0)
    
    Args:
    - t_hat (numpy.ndarray): Translational state estimates of shape (D, n).
    - Rhat (numpy.ndarray): Rotational state estimates of shape (D, D, n).
    - edges (numpy.ndarray, optional): Matrix encoding the edges in the measurement network.
    - lc_linestyle (str, optional): Line style for plotting the loop closure edges. Default is '-b'.
    - lc_alpha (float, optional): Alpha value for the loop closure edges. Default is 1.0.
    
    Returns:
    - f (matplotlib.figure.Figure): The generated figure.
    """
    D = t_hat.shape[0]  # Get dimension of solutions, should be either 2 or 3
    
    # 'Unrotate' these vectors by premultiplying by the inverse of the first orientation estimate
    t_hat_rotated = np.transpose(np.matmul(np.transpose(Rhat[:, :, 0]), np.transpose(t_hat)))
    # Translate the resulting vectors to the origin
    t_hat_anchored = t_hat_rotated - np.tile(t_hat_rotated[:, 0], (t_hat_rotated.shape[1], 1)).T
    
    x = t_hat_anchored[0, :]
    y = t_hat_anchored[1, :]
    
    if D == 3:
        z = t_hat_anchored[2, :]
        
        # Plot odometric links
        f = plt.figure()
        ax = f.add_subplot(111, projection='3d')
        ax.plot(x, y, z, lc_linestyle)
        
        if edges is not None:
            for k in range(edges.shape[0]):
                id1 = edges[k, 0]
                id2 = edges[k, 1]
                
                if abs(id1 - id2) > 1:
                    # This is a loop closure measurement
                    lc_plot = ax.plot(t_hat_anchored[0, [id1, id2]], t_hat_anchored[1, [id1, id2]], t_hat_anchored[2, [id1, id2]], lc_linestyle, linewidth=1)
                    lc_plot[0].set_alpha(lc_alpha)  # Set transparency of loop closure edges
        
        ax.set_aspect('equal')
    
    elif D == 2:
        # Plot odometric links
        f = plt.figure()
        plt.plot(x, y, '-b')
        
        if edges is not None:
            for k in range(edges.shape[0]):
                id1 = edges[k, 0]
                id2 = edges[k, 1]
                
                if abs(id1 - id2) > 1:
                    # This is a loop closure measurement
                    lc_plot = plt.plot(t_hat_anchored[0, [id1, id2]], t_hat_anchored[1, [id1, id2]], lc_linestyle)
                    lc_plot[0].set_alpha(lc_alpha)  # Set transparency of loop closure edges
        
        plt.axis('equal')
    
    return f