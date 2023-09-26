from typing import List
import numpy as np

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
    d = len(w0)

    # Preprocessing
    y_size, x_size = X.shape
    X2 = X * X
    
    # Initialisation
    xhat = np.zeros(x_size)
    vhat = np.ones(x_size)
    g = np.zeros(y_size)

    count = 0

    status = None

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
    
    return retour