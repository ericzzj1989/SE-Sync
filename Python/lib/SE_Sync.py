import numpy as np
from scipy.sparse import spdiags 
from scipy.sparse.linalg import svds, eigsh, spilu, spsolve
from scipy.linalg import cholesky
from scipy.linalg import solve
import time
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers

from lib.construct_problem_data import construct_problem_data
from lib.chordal_initialization import chordal_initialization
from lib.evaluate_objective import evaluate_objective
from lib.Euclidean_gradient import Euclidean_gradient
from lib.Euclidean_Hessian_vector_product import Euclidean_Hessian_vector_product
from lib.relative_func_decrease_stopfun import relative_func_decrease_stopfun
from lib.log_iterates import log_iterates
from lib.compute_Lambda import compute_Lambda
from lib.Q_minus_Lambda_min_eig import Q_minus_Lambda_min_eig
from lib.round_solution import round_solution
from lib.recover_translations import recover_translations
from lib.Qproduct import Qproduct


def SE_Sync(measurements, Manopt_opts=None, SE_Sync_opts=None, Y0=None):
    """
    SE-Sync: A certifiably correct algorithm for synchronization over the special Euclidean group

    Args:
    - measurements: A dictionary containing the data describing the special Euclidean synchronization problem.
      Specifically, measurements must contain the following keys:
        * 'edges': An (mx2)-dimensional matrix encoding the edges in the measurement network.
                   Each row [i,j] represents a relative transform x_i^{-1} x_j.
        * 'R': An m-dimensional list whose kth element is the rotational part of the kth measurement.
        * 't': An m-dimensional list whose kth element is the translational part of the kth measurement.
        * 'kappa': An m-dimensional list whose kth element gives the precision of the rotational part of the kth measurement.
        * 'tau': An m-dimensional list whose kth element gives the precision of the translational part of the kth measurement.
    - Manopt_opts (optional): A dictionary containing various options that determine the behavior of Manopt's Riemannian truncated-Newton trust-region method.
    - SE_Sync_opts (optional): A dictionary determining the behavior of the SE-Sync algorithm.
    - Y0 (optional): An initial point on the manifold St(d, r)^n at which to initialize the first Riemannian optimization problem.

    Returns:
    - SDPval: The optimal value of the semidefinite relaxation.
    - Yopt: A symmetric factor of an optimal solution Zopt = Yopt' * Yopt for the semidefinite relaxation.
    - xhat: A dictionary containing the estimate for the special Euclidean synchronization problem.
    - Fxhat: The objective value of the rounded solution xhat.
    - SE_Sync_info: A dictionary containing various possibly-interesting bits of information about the execution of the SE-Sync algorithm.
    - problem_data: A dictionary containing several auxiliary matrices constructed from the input measurements that are used internally throughout the SE-Sync algorithm.
    """
    print('\n\n========== SE-Sync ==========\n\n')
    timerVal = time.time()
    
    ## INPUT PARSING
    # SE-Sync settings:
    print('ALGORITHM SETTINGS:\n\n')
        
    if 'SE_Sync_opts' not in locals():
        print('Using default settings for SE-Sync:')
        SE_Sync_opts = {}  # Create empty dictionary
    else:
        print('SE-Sync settings:')

    if 'r0' in SE_Sync_opts:
        print(' Initial level of Riemannian Staircase:', SE_Sync_opts['r0'])
    else:
        SE_Sync_opts['r0'] = 5
        print(' Setting initial level of Riemannian Staircase to', SE_Sync_opts['r0'], '[default]')

    if 'rmax' in SE_Sync_opts:
        print(' Final level of Riemannian Staircase:', SE_Sync_opts['rmax'])
    else:
        SE_Sync_opts['rmax'] = 7
        print(' Setting final level of Riemannian Staircase to', SE_Sync_opts['rmax'], '[default]')

    if 'eig_comp_max_iters' in SE_Sync_opts:
        print(' Maximum number of iterations to perform for minimum eigenvalue computation in test for positive semidefiniteness:', SE_Sync_opts['eig_comp_max_iters'])
    else:
        SE_Sync_opts['eig_comp_max_iters'] = 2000
        print(' Maximum number of iterations to perform for minimum eigenvalue computation in test for positive semidefiniteness:', SE_Sync_opts['eig_comp_max_iters'], '[default]')

    if 'min_eig_num_tol' in SE_Sync_opts:
        print(' Tolerance for accepting an eigenvalue as numerically nonnegative in optimality verification:', SE_Sync_opts['min_eig_num_tol'])
    else:
        SE_Sync_opts['min_eig_num_tol'] = 1e-4
        print(' Tolerance for accepting an eigenvalue as numerically nonnegative in optimality verification:', SE_Sync_opts['min_eig_num_tol'], '[default]')

    if 'Cholesky' not in SE_Sync_opts:
        print(' Using QR decomposition to compute orthogonal projection [default]')
        SE_Sync_opts['Cholesky'] = False
    else:
        if SE_Sync_opts['Cholesky']:
            print(' Using Cholesky decomposition to compute orthogonal projection')
        else:
            print(' Using QR decomposition to compute orthogonal projection')

    if 'init' not in SE_Sync_opts:
        print(' Initialization method: chordal [default]')
        SE_Sync_opts['init'] = 'chordal'
    else:
        if SE_Sync_opts['init'] == 'chordal':
            print(' Initialization method: chordal')
        elif SE_Sync_opts['init'] == 'random':
            print(' Initialization method: random')
        else:
            raise ValueError('Initialization option "%s" not recognized! (Supported options are "chordal" or "random")' % SE_Sync_opts['init'])

    print('\n')

    ## Manopt settings:
    # Check Manopt options
    if 'Manopt_opts' not in locals():
        print('Using default settings for Manopt:')
        Manopt_opts = {}  # Create empty dictionary
    else:
        print('Manopt settings:')

    # Stopping tolerance for norm of Riemannian gradient
    if 'tolgradnorm' in Manopt_opts:
        print(' Stopping tolerance for norm of Riemannian gradient:', Manopt_opts['tolgradnorm'])
    else:
        Manopt_opts['tolgradnorm'] = 1e-2
        print(' Setting stopping tolerance for norm of Riemannian gradient to:', Manopt_opts['tolgradnorm'], '[default]')

    # Stopping tolerance for relative function decrease
    if 'rel_func_tol' in Manopt_opts:
        print(' Stopping tolerance for relative function decrease:', Manopt_opts['rel_func_tol'])
    else:
        Manopt_opts['rel_func_tol'] = 1e-5
        print(' Setting stopping tolerance for relative function decrease to:', Manopt_opts['rel_func_tol'], '[default]')

    # Maximum number of Hessian-vector products in each truncated Newton iteration
    if 'maxinner' in Manopt_opts:
        print(' Maximum number of Hessian-vector products to evaluate in each truncated Newton iteration:', Manopt_opts['maxinner'])
    else:
        Manopt_opts['maxinner'] = 1000
        print(' Setting maximum number of Hessian-vector products to evaluate in each truncated Newton iteration to:', Manopt_opts['maxinner'], '[default]')

    # Minimum number of trust-region iterations
    if 'miniter' in Manopt_opts:
        print(' Minimum number of trust-region iterations:', Manopt_opts['miniter'])
    else:
        Manopt_opts['miniter'] = 1
        print(' Setting minimum number of trust-region iterations to:', Manopt_opts['miniter'], '[default]')

    # Maximum number of trust-region iterations
    if 'maxiter' in Manopt_opts:
        print(' Maximum number of trust-region iterations:', Manopt_opts['maxiter'])
    else:
        Manopt_opts['maxiter'] = 500
        print(' Setting maximum number of trust-region iterations to:', Manopt_opts['maxiter'], '[default]')

    # Maximum permissible elapsed computation time
    if 'maxtime' in Manopt_opts:
        print(' Maximum permissible elapsed computation time [sec]:', Manopt_opts['maxtime'])

    # Preconditioner for truncated conjugate gradient inexact Newton step computations
    if 'preconditioner' not in Manopt_opts:
        print(' Using incomplete zero-fill Cholesky preconditioner for truncated conjugate gradient inexact Newton step computations [default]')
        Manopt_opts['preconditioner'] = 'ichol'
    else:
        if Manopt_opts['preconditioner'] == 'ichol':
            print(' Using incomplete zero-fill Cholesky preconditioner for truncated conjugate gradient inexact Newton step computations')
        elif Manopt_opts['preconditioner'] == 'Jacobi':
            print(' Using Jacobi preconditioner for truncated conjugate gradient inexact Newton step computations')
        elif Manopt_opts['preconditioner'] == 'none':
            print(' Using unpreconditioned truncated conjugate gradient for inexact Newton step computations')
        else:
            raise ValueError('Initialization option "%s" not recognized! (Supported options are "Jacobi" or "none")' % Manopt_opts['preconditioner'])
    
    
    ## Construct problem data matrices from input
    print('\n\nINITIALIZATION:\n\n')
    print('Constructing auxiliary data matrices from raw measurements...')
    aux_time_start = time.time()
    problem_data = construct_problem_data(measurements)
    auxiliary_matrix_construction_time = time.time() - aux_time_start
    print('Auxiliary data matrix construction finished. Elapsed computation time: {} seconds\n\n'.format(auxiliary_matrix_construction_time))
    
    ## Construct (Euclidean) preconditioning function handle, if desired
    if 'preconditioner' in Manopt_opts:
        precon_construction_start_time = time.time()
        if Manopt_opts['preconditioner'] == 'ichol':
            print('Constructing incomplete Cholesky preconditioner... ')

            LGrho = problem_data['ConLap']

            # Regularize this matrix by adding a very small positive
            # multiple of the identity to account for the fact that the
            # rotational connection Laplacian is singular

            # Compute incomplete zero-fill
            L = cholesky(LGrho.toarray())
            # diag_values = L.diagonal()
            # diagcomp = 1e-3
            # diag_values[np.abs(diag_values) < diagcomp] = diagcomp
            # L.setdiag(diag_values)
            # return(f'L: {L}')
            LT = L.T

            # def precon(u):
            #     return LT @ (L @ u)
            # print(f'L: {L.shape}')
            precon = lambda u: spsolve(LT, spsolve(L, u))
            # def precon(u):
            #     # Solve L * x = u
            #     x = solve(L, u)
            #     # Solve LT * y = x
            #     y = solve(LT, x)
            #     return y
            
        elif Manopt_opts['preconditioner'] == 'Jacobi':
            print('Constructing Jacobi preconditioner... ')
            J = problem_data['ConLap']

            # Extract diagonal elements
            D = spdiags(J.diagonal(), 0, J.shape[0], J.shape[1])

            # Invert these
            Dinv = 1.0 / D

            # Construct diagonal matrix with this size
            Pinv = spdiags(Dinv.diagonal(), 0, problem_data['d'] * problem_data['n'], problem_data['d'] * problem_data['n'])
            
            # Set preconditioning function
            # def precon(u):
            #     return Pinv @ u
            precon = lambda u: Pinv.dot(u)
            
        if Manopt_opts['preconditioner'] != 'none':
            precon_construction_end_time = time.time()
            print('Elapsed computation time: {} seconds\n\n'.format(precon_construction_end_time - precon_construction_start_time))

    # INITIALIZATION

    # The maximum number of levels in the Riemannian Staircase that we will
    # need to explore
    max_num_iters = SE_Sync_opts['rmax'] - SE_Sync_opts['r0'] + 1

    # Allocate storage for state traces
    optimization_times = np.zeros((1, max_num_iters))
    SDPLRvals = np.zeros((1, max_num_iters))
    min_eig_times = np.zeros((1, max_num_iters))
    min_eig_vals = np.zeros((1, max_num_iters))
    gradnorms = []
    Yvals = {}

    # Set up Manopt problem

    # We optimize over the manifold M := St(d, r)^N, the N-fold product of the
    # (Stiefel) manifold of orthonormal d-frames in R^r.
    manopt_data = {}
    print('n: ', problem_data['n'])
    print('d: ', problem_data['d'])
    print('r0: ', SE_Sync_opts['r0'])
    # manopt_data['M'] = pymanopt.manifolds.Stiefel(problem_data['n'], problem_data['d'], SE_Sync_opts['r0'])
    manopt_data['M'] = pymanopt.manifolds.Stiefel(problem_data['n'], problem_data['d'], k=SE_Sync_opts['r0'])

    # Check if an initial point was supplied
    if 'Y0' not in locals():
        if SE_Sync_opts['init'] == 'chordal':
            print('Computing chordal initialization...')
            init_time_start = time.time()
            Rchordal = chordal_initialization(measurements)
            Y0 = np.vstack((Rchordal, np.zeros((SE_Sync_opts['r0'] - problem_data['d'], problem_data['d'] * problem_data['n']))))
            init_time = time.time() - init_time_start
        else:  # Use randomly-sampled initialization
            print('Randomly sampling an initial point on St({}, {})^{} ...'.format(problem_data['d'], SE_Sync_opts['r0'], problem_data['n']))
            init_time_start = time.time()
            # Sample a random point on the Stiefel manifold as an initial guess
            Y0 = manopt_data['M'].random_point().T
            init_time = time.time() - init_time_start
        print('Elapsed computation time: {} seconds'.format(init_time))
    else:
        print('Using user-supplied initial point Y0 in Riemannian Staircase\n')
        init_time = 0

    # Check if a solver was explicitly supplied
    if 'solver' not in Manopt_opts:
        # Use the trust-region solver by default
        Manopt_opts['solver'] = pymanopt.optimizers.TrustRegions(verbosity=2)
    # solver_name = Manopt_opts['solver'].__name__
    # if solver_name not in ['trustregions', 'conjugategradient', 'steepestdescent']:
    #     raise ValueError('Unrecognized Manopt solver: {}'.format(solver_name))
    # print('\nSolving Riemannian optimization problems using Manopt''s "{}" solver\n'.format(solver_name))

    # Set cost function handles
    @pymanopt.function.numpy(manopt_data['M'])
    def cost(Y):
        print("===================== cost ========================")
        # Yt = Y.T
        print('Y: ', Y.shape)
        YQ = Qproduct(Y, problem_data, use_Cholesky=SE_Sync_opts['Cholesky']).T
        trQYtY = np.trace(YQ @ Y)
        print('cost: ', YQ.shape)
        print("===================== cost ========================")
        return trQYtY, YQ
    
    @pymanopt.function.numpy(manopt_data['M'])
    def euclidean_gradient(Y):
        print("===================== euclidean_gradient ========================")
        # Yt = Y.T
        print('Y: ', Y.shape)
        egrad = 2 * Qproduct(Y, problem_data, use_Cholesky=SE_Sync_opts['Cholesky'], verbose=True)
        print('euclidean gradient: ', egrad.shape)
        print("===================== euclidean_gradient ========================")
        return egrad
    
    # @pymanopt.function.numpy(manopt_data['M'])
    # def euclidean_gradient(Y):
    #     print("===================== euclidean_gradient ========================")
    #     Yt = Y.T
    #     egrad = 2 * Qproduct(Yt, problem_data, SE_Sync_opts['Cholesky']).T
    #     print("===================== euclidean_gradient ========================")
    #     return egrad
    
    @pymanopt.function.numpy(manopt_data['M'])
    def euclidean_hessian(Y, Ydot):
        print("===================== euclidean_hessian ========================")
        # Ydott = Ydot.T
        print(f'Ydot: {Ydot.shape}')
        Hvec = 2 * Qproduct(Ydot, problem_data, use_Cholesky=SE_Sync_opts['Cholesky'])
        print('euclidean hessian: ', Hvec.shape)
        print("===================== euclidean_hessian ========================")
        return Hvec

    # manopt_data['cost'] = lambda Y: evaluate_objective(Y.T, problem_data, SE_Sync_opts['Cholesky'])
    # manopt_data['egrad'] = lambda Y: Euclidean_gradient(Y.T, problem_data, SE_Sync_opts['Cholesky']).T
    # manopt_data['ehess'] = lambda Y, Ydot: Euclidean_Hessian_vector_product(Y.T, Ydot.T, problem_data, SE_Sync_opts['Cholesky']).T

    # Set preconditioning function, if desired
    # if 'precon' in globals():
    #     print(f'have precon!!!')
    #     manopt_data['precon'] = lambda x, u: manopt_data['M'].proj(x, precon(u))

    def preconditioner(x, u):
        print(f'x: {x.shape}')
        print(f'u: {u.shape}')
        # print(f'precon: {precon(u.T).shape}')
        return manopt_data['M'].projection(x, precon(u))

    # problem = pymanopt.Problem()

    # Set additional stopping criterion for Manopt: stop if the relative
    # decrease in function value between successive iterates drops below the
    # threshold specified in SE_Sync_opts.relative_func_decrease_tol
    # if solver_name == 'trustregions':
    Manopt_opts['stopfun'] = lambda manopt_problem, x, info, last: relative_func_decrease_stopfun(manopt_problem, x, info, last, Manopt_opts['rel_func_tol'])

    # Log the sequence of iterates visited by the Riemannian Staircase
    Manopt_opts['statsfun'] = log_iterates

    # Counter to keep track of how many iterations of the Riemannian Staircase
    # have been performed
    iter = 0

    for r in range(SE_Sync_opts['r0'], SE_Sync_opts['rmax'] + 1):
        iter += 1  # Increment iteration number
        
        # Starting at Y0, use Manopt's truncated-Newton trust-region method to
        # descend to a first-order critical point.
        
        print('\nRIEMANNIAN STAIRCASE (level r = {}):\n'.format(r))
        
        # YoptT, Fval, manopt_info, Manopt_opts = pymanopt.optimizers.optimize(manopt_data, Y0.T, Manopt_opts)
        problem = pymanopt.Problem(manopt_data['M'], cost,
                                   euclidean_gradient=euclidean_gradient, euclidean_hessian=euclidean_hessian)
                                #    preconditioner=preconditioner)
        
        # print(f'Y0: {Y0.shape}')
        # print('eta:', manopt_data['M'].zero_vector(Y0).shape)
        # eta = manopt_data['M'].zero_vector(Y0)
        # print('fgradx: ', problem.riemannian_gradient(Y0).shape)
        # fgradx = problem.riemannian_gradient(Y0)
        # print('z: ', problem.preconditioner(Y0,fgradx).shape)
        result = Manopt_opts['solver'].run(problem, initial_point=Y0.T)
        Yopt = result.point
        SDPLRval = result.cost
        
        # Store the optimal value and the elapsed computation time
        SDPLRvals[iter - 1] = SDPLRval
        optimization_times[iter - 1] = result.time
        
        # Store gradient norm and state traces
        gradnorms.append(result.gradient_norm)
        Yvals.append(result.iterations)
        
        # Augment Yopt by padding with an additional row of zeros; this
        # preserves Yopt's first-order criticality while ensuring that it is
        # rank-deficient
        
        Yplus = np.vstack((Yopt, np.zeros((1, problem_data['d'] * problem_data['n']))))
        
        
        print('\nChecking second-order optimality...\n')
        # At this point, Yplus is a rank-deficient critial point, so check
        # 2nd-order optimality conditions
        
        # Compute Lagrange multiplier matrix Lambda corresponding to Yplus
        Lambda = compute_Lambda(Yopt, problem_data, SE_Sync_opts['Cholesky'])
        
        # Compute minimum eigenvalue/eigenvector pair for Q - Lambda
        min_eig_comp_time_start = time.time()
        lambda_min, v = Q_minus_Lambda_min_eig(Lambda, problem_data, Yopt, SE_Sync_opts['min_eig_num_tol'], SE_Sync_opts['eig_comp_max_iters'], SE_Sync_opts['Cholesky'])
        min_eig_comp_time = time.time() - min_eig_comp_time_start
        
        # Store the minimum eigenvalue and elapsed computation times
        min_eig_vals[iter - 1] = lambda_min
        min_eig_times[iter - 1] = min_eig_comp_time
        
        if lambda_min > -SE_Sync_opts['min_eig_num_tol']:
            # Yopt is a second-order critical point
            print('Found second-order critical point! (minimum eigenvalue = {}, elapsed computation time {} seconds)\n'.format(lambda_min, min_eig_comp_time))
            break
        # else:
        #     print('Saddle point detected (minimum eigenvalue = {}, elapsed computation time {} seconds)\n'.format(lambda_min, min_eig_comp_time))
        #     # lambda_min is a negative eigenvalue of Q - Lambda, so the KKT
        #     # conditions for the semidefinite relaxation are not satisfied;
        #     # this implies that Yplus is a saddle point of the rank-restricted
        #     # semidefinite optimization.  Fortunately, the eigenvector v
        #     # corresponding to lambda_min can be used to provide a descent
        #     # direction from this saddle point, as described in Theorem 3.9 of
        #     # the paper "A Riemannian Low-Rank Method for Optimization over
        #     # Semidefinite Matrices with Block-Diagonal Constraints".
            
        #     # Define the vector Ydot := e_{r+1} * v'; this is tangent to the
        #     # manifold St(d, r+1)^n at Yplus and provides a direction of
        #     # negative curvature
        #     print('Computing escape direction...')
        #     Ydot = np.vstack((np.zeros((r, problem_data['d'] * problem_data['n'])), v))
            
        #     # Augment the dimensionality of the Stiefel manifolds in
        #     # preparation for the next iteration
            
        #     manopt_data['M'] = pymanopt.manifolds.Stiefel(problem_data['n'], problem_data['d'], k=r+1)

        #     problem = pymanopt.Problem(manopt_data['M'], cost,
        #                            euclidean_gradient=euclidean_gradient, euclidean_hessian=euclidean_hessian,
        #                            preconditioner=preconditioner)
            
        #     # Update preconditioning function, if it's used
        #     Manopt_opts['solver'] = pymanopt.optimizers.line_search.BackTrackingLineSearcher()
            
        #     # if 'precon' in globals():
        #     #     manopt_data['precon'] = lambda x, u: manopt_data['M'].proj(x, precon(u))
            
        #     # Perform line search along the escape direction Ydot to escape the
        #     # saddle point and obtain the initial iterate for the next level in
        #     # the Staircase
            
        #     # Compute a scaling factor alpha such that the scaled step
        #     # alpha*Ydot' should produce a trial point Ytest whose gradient has
        #     # a norm 100 times greater than the gradient tolerance stopping
        #     # criterion currently being used in the RTR optimization routine
        #     alpha = Manopt_opts['tolgradnorm'] / (np.linalg.norm(v) * abs(lambda_min))
            
        #     print('Line searching along escape direction to escape saddle point...')
        #     # line_search_time_start = time.time()
        #     result = Manopt_opts['solver'].search(problem, manopt_data['M'], Yplus, alpha*Ydot, )
        #     # stepsize, Y0T = pymanopt.optimizers.line_search.BackTrackingLineSearcher(manopt_data, Yplus.T, alpha * Ydot.T, SDPLRval)
        #     # line_search_time = time.time() - line_search_time_start 
        #     Y0 = result.point
        #     print('Line search completed (elapsed computation time {} seconds)\n'.format(result.time))

    print('\n\n===== END RIEMANNIAN STAIRCASE =====\n\n')

    ## POST-PROCESSING

    # Return optimal value of the SDP (in the case that a rank-deficient,
    # second-order critical point is obtained, this is equal to the optimum
    # value obtained from the Riemannian optimization

    SDPval = SDPLRval

    print('Rounding solution...')
    # Round the solution
    solution_rounding_time_start = time.time()
    Rhat = round_solution(Yopt, problem_data)
    solution_rounding_time = time.time() - solution_rounding_time_start
    print('Elapsed computation time: {} seconds\n'.format(solution_rounding_time))

    print('Recovering translational estimates...')
    # Recover the optimal translational estimates
    translation_recovery_time_start = time.time()
    that = recover_translations(Rhat, problem_data)
    translation_recovery_time = time.time() - translation_recovery_time_start
    print('Elapsed computation time: {} seconds\n'.format(translation_recovery_time))

    xhat = {}
    xhat['R'] = Rhat
    xhat['t'] = that

    Fxhat = evaluate_objective(Rhat, problem_data, SE_Sync_opts['Cholesky'])

    print('Value of SDP solution F(Y): {}'.format(SDPval))
    print('Norm of Riemannian gradient grad F(Y): {}'.format(result.gradient_norm))
    print('Value of rounded pose estimate xhat: {}'.format(Fxhat))
    print('Suboptimality bound of recovered pose estimate: {}\n'.format(Fxhat - SDPval))
    total_computation_time = time.time() - timerVal

    print('Total elapsed computation time: {} seconds\n\n'.format(total_computation_time))

    # Output info
    SE_Sync_info = {}
    SE_Sync_info['mat_construct_times'] = auxiliary_matrix_construction_time
    SE_Sync_info['init_time'] = init_time
    SE_Sync_info['SDPLRvals'] = SDPLRvals[:iter]
    SE_Sync_info['optimization_times'] = optimization_times[:iter]
    SE_Sync_info['min_eig_vals'] = min_eig_vals[:iter]
    SE_Sync_info['min_eig_times'] = min_eig_times[:iter]
    SE_Sync_info['manopt_info'] = result
    SE_Sync_info['total_computation_time'] = total_computation_time
    SE_Sync_info['Yvals'] = Yvals
    SE_Sync_info['gradnorms'] = gradnorms

    print('\n===== END SE-SYNC =====\n')