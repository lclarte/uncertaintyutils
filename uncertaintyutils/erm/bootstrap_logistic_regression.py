
# script to run bootstrap on logistic classification 

import numpy as np
import sklearn.linear_model as linear_model

from . import logistic_regression
from .. import utility

def bootstrap_logistic_classification(X, Y, lambda_, n_resamples = 1000):
    """
    NOTE : as n_resamples -> \infty, the average of the bootstrap weights should converge to the one of ERM
    The question is : what's the variance of the prediction for the confidence ? Does it contain the true one ? 
    """
    n, d = X.shape
    ws = np.zeros((n_resamples, d))

    for trial in range(n_resamples):
        indices = np.random.choice(d, size=n, replace=True)
        X_resample, Y_resample = X[indices], Y[indices]

        ws[trial] = logistic_regression.solve_logistic_regression(X_resample, Y_resample, lambda_)
    
    return ws

def poisson_logistic_classification(X, Y, lambda_, n_resamples = 1000):
    """
    Runs logistic regression with Poisson resampling, asymptotically equivalent to bootstrap
    """
    n, d = X.shape
    ws = np.zeros((n_resamples, d))
    for trial in range(n_resamples):
        weights = np.random.poisson(lam = 1.0, size = n)
        lr = linear_model.LogisticRegression(fit_intercept=False, max_iter=10000, tol=1e-16, C=1.0 / lambda_)
        lr.fit(X, Y, sample_weight=weights)
        ws[trial] = lr.coef_[0]

    return ws

# 

def subsample_logistic_classification(X, Y, lambda_, ratio, n_resamples = 100):
    n, d = X.shape
    ws = np.zeros((n_resamples, d))
    
    for trial in range(n_resamples):
        weights = np.random.binomial(n = 1, p = ratio, size = n)
        lr = linear_model.LogisticRegression(fit_intercept=False, max_iter=10000, tol=1e-16, C=1.0 / lambda_)
        lr.fit(X, Y, sample_weight=weights)
        ws[trial] = lr.coef_[0]
    return ws

# ===== 

def average_confidence(x, ws_bootstrap):
    # return the average of the confidence from each bootstrap
    return np.mean(utility.sigmoid(x @ ws_bootstrap.T), axis=-1)

#

def variance_confidence(x, ws_bootstrap):
    return np.var(utility.sigmoid(x @ ws_bootstrap.T), axis=-1)

def average_label(x, ws_bootstrap):
    return np.mean(np.sign(x @ ws_bootstrap.T), axis=-1)

def variance_label(x, ws_bootstrap):
    return np.var(np.sign(x @ ws_bootstrap.T), axis=-1)