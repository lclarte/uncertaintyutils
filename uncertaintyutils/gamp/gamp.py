from typing import List
import numpy as np
import scipy.linalg as linalg

from . import prior

def iterate_gamp(X : List[List[float]], Y : List[float], w0 : List[float], likelihood, prior, max_iter : int = 200, tol : float =1e-7, 
                 damp : float =0.0, early_stopping : bool =False, verbose : bool = False) -> dict:
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
    xhat = np.zeros(x_size)
    vhat = np.ones(x_size)
    g = np.zeros(y_size)

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
    
    return retour

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
