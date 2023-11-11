# laplace_prior.py
# used for Bayes optimal estimation 

import numpy as np
from scipy.special import erf, erfc

def dR_log_Z(Sigma, R, lambda_):
    tmp = np.sqrt(2.0 * Sigma)
    Rm, Rp = (R - lambda_ * Sigma), (R + lambda_ * Sigma)
    return - lambda_ * (1 + erf(Rm / tmp) - np.exp(2 * R * lambda_) * erfc(Rp / tmp) ) / \
                       (1 + erf(Rm / tmp) + np.exp(2 * R * lambda_) * erfc(Rp / tmp) )

def ddR_log_Z(Sigma, R, lambda_):
    tmp = np.sqrt(2.0 * Sigma)
    tmp_exp = np.exp(2 * R * lambda_)
    tmp_pi = np.sqrt(2.0 / np.pi)
    Rm, Rp = R - lambda_ * Sigma, R + lambda_ * Sigma
    return 2 * lambda_ * tmp_exp * ( - np.exp(-Rp**2 / tmp**2) * tmp_pi + ( 2 * lambda_ * np.sqrt(Sigma) - tmp_pi * np.exp(-Rm**2 / tmp**2)) * erfc(Rp / tmp) + \
            erf(Rm / tmp) * (2 * lambda_ * np.sqrt(Sigma) * erfc(Rp / tmp) - np.exp(-Rp**2 / tmp**2) * tmp_pi )) / \
            (np.sqrt(Sigma) * (1.0 + erf(Rm / tmp) + tmp_exp * erfc(Rp / tmp) )**2)
    
class LaplacePrior:
    def __init__(self, lambda_ = 1.0) -> None:
        # for ERM penalization
        self.lambda_ = lambda_
    
    def fa(self, Sigma : float, R : float) -> float:
        """
        Input function, independent of the variance of gaussian prior
        NOTE : Should not depend on the noise in label
        """
        return Sigma * (dR_log_Z(Sigma, R, self.lambda_) + R)

    def fv(self, Sigma : float, R : float) -> float:
        """
        Derivative of input function w.r.t. R, multiplied by Sigma
        """
        return Sigma**2 * (ddR_log_Z(Sigma, R, self.lambda_) + 1.0)
    
    def prior(self, b : float, A : float):
        '''
        Compute f and f' for Bernoulli-Gaussian prior
        
        Sigma = 1 / A
        R = b / A
        '''
        return self.fa(1. / A, b / A), self.fv(1. / A, b / A)