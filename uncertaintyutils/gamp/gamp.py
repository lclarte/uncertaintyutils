from typing import List
import numpy as np
import scipy.linalg as linalg

from . import prior
from . import likelihood

def iterate_gamp(X : List[List[float]], Y : List[float], w0 : List[float], likelihood, prior, max_iter : int = 200, tol : float =1e-7, 
                 damp : float =0.0, early_stopping : bool =False, verbose : bool = False, xhat_0 = None, vhat_0 = None, g_0 = None) -> dict:
    """
    MAIN FUNCTION : Runs G-AMP and returns the finals parameters. If we study
    the variance, we are interested in the vhat quantities. The 'variance' of the vector 
    w will (normally) be the sum of the vhat.

    parameters :
        - W : data matrix
        - y : funciton output
        - w0 : ground truth
    returns : 
        - retour : dictionnary with informations
    """

    # Preprocessing
    y_size, x_size = X.shape
    X2 = X * X
    
    # Initialisation
    if xhat_0 is None:
        xhat = np.zeros(x_size)
    else:
        xhat = np.copy(xhat_0)
    
    if vhat_0 is None:
        vhat = np.ones(x_size)
    else:
        vhat = np.copy(vhat_0)

    if g_0 is None:
        g = np.zeros(y_size)
    else:
        g = np.copy(g_0)

    for t in range(max_iter):
        # onsager term, why not take the divergence ? 
        V     = X2 @ vhat
        
        # here we see that V is the Onsager term
        omega = X @ xhat - V * g
        g, dg = likelihood.channel(Y, omega, V)

        # Second part
        A = - X2.T @ dg
        b = A * xhat + X.T @ g

        xhat_old = xhat.copy() # Keep a copy of xhat to compute diff

        xhat, vhat = prior.prior(b, A)

        diff = np.mean(np.abs(xhat-xhat_old))
        # Expression of MSE has been changed

        if (diff < tol):
            status = 'Done'
            break

        if verbose:
            # NOTE : Not necessarily the good q if the data matrix has not identity covariance
            q = np.mean(xhat * xhat)
            print(f'q = {q}')
            print(f'Variance = {np.mean(vhat)}')

    if verbose:
        print('t : ', t)
        print(f'diff : {diff}')

    retour = {}
    retour['estimator'] = xhat
    retour['variances'] = vhat
    retour['omega']     = omega
    retour['g_out']     = g
    
    return retour

def get_cavity_means_from_gamp(x_mat, y_vec, what, vhat, omega, likelihood_):
    n = len(y_vec)
    what_mat = np.tile(what, (n, 1)).T
    V = (x_mat * x_mat) @ vhat
    # For Gaussian likelihood, we can accelerate the computation by paralellization
    if isinstance(likelihood_, likelihood.gaussian_log_likelihood.GaussianLogLikelihood):
        return what_mat - x_mat.T * np.outer(vhat, likelihood_.fout(y_vec, omega, V))
    else:
        return what_mat - x_mat.T * np.outer(vhat, [ likelihood_.fout(y=y1, w=omega1, V=V1) for y1, omega1, V1 in zip(y_vec, omega, V) ])

def gamp_nonspherical_covariance(mat_x, vec_y, mat_prior_cov, likelihood, max_iter=200, tol=1e-7, damp=0.0):
    """
    Compute the Bayesian estimator when the prior covariance is not lambda * I_d. To do so we do a Cholesky decomposition of the covariance
    mat_prior_cov = LL^T and then transform the input mat_x -> mat_x @ L
    Note that the covariance must be positive definite, but adding epsilon * identity solves the issue of 0 eigenvalues
    """
    # transform the data
    L = linalg.cholesky(mat_prior_cov, lower=False)
    mat_x_L = mat_x @ L
    prior_ = prior.gaussian_prior.GaussianPrior(lambda_ = 1.0)

    result = iterate_gamp(mat_x_L, vec_y, None, likelihood, prior_, max_iter=max_iter, tol=tol, damp=damp)

    # In order to apply the estimator directly on test data we do a change of variable
    what = L @ result['estimator']
    # now we return not a vector but a matrix because of the change of variable I guess
    vhat = L @ np.diag('variances') @ L.T

    return what, vhat

def retrain_gamp(what_old, vhat_old, x_new, y_new, likelihood_, prior_, x_old = None, y_old = None, g_old = None, tol = 1e-5, max_iter = 200, verbose = False):
    """
    From an estimator (what, vhat) that has converged and a new sample (x_new, y_new), update and returns
    the new estimator.
    For now, we just run AMP from what, vhat as starting point
    argument:
    """
    assert (not x_old is None) and (not y_old is None), "Requires previous data to update"
    # stack x_old and x_new
    x_mat = np.vstack((x_old, x_new))
    y_vec = np.hstack((y_old, y_new))

    # for new samples we have no estimation of g so we just fill with zeros
    if g_old is None:
        g_vec = None
    else:
        g_vec = np.hstack((g_old, np.zeros(len(x_new))))

    return iterate_gamp(x_mat, y_vec, None, likelihood=likelihood_, prior=prior_, tol = tol, max_iter=max_iter, verbose=verbose, xhat_0=what_old, vhat_0=vhat_old, g_0=g_vec)