# lasso_prior.py
# see https://github.com/takashi-takahashi/approximate_message_passing/blob/master/ampy/AMPSolver.py

import numpy as np

class LassoPrior:
    def __init__(self, lambda_) -> None:
        self.lambda_ = lambda_

    def fa(self, Sigma, R):
        return (R - self.lambda_ * Sigma * np.sign(R)) * np.heaviside(np.abs(R) - self.lambda_ * Sigma, 0.5)

    def fv(self, Sigma, R):
        return Sigma * np.heaviside(np.abs(R) - self.lambda_ * Sigma, 0.5)
    
    def prior(self, b : float, A : float):
        '''
        Compute f and f' for Bernoulli-Gaussian prior
        
        Sigma = 1 / A
        R = b / A
        '''
        return self.fa(1. / A, b / A), self.fv(1. / A, b / A)