"""
conformal_gamp.py
Main functions to conformalize predictions w/ GAMP
"""

import numpy as np

from ..gamp import gamp

def get_scores_gamp_erm(x, y, xtest, ytest, prior_, likelihood_, **gamp_args):
    """
    return:
        - tuple (scores of training data, score of test sample) if the hypothesis is that y_n+1 = ytest
    """
    xtrain = np.hstack((x, xtest.reshape((1, -1))))
    ytrain = np.hstack((y, ytest))
    
    result = gamp.iterate_gamp(xtrain, ytrain, None, likelihood_, prior_, **gamp_args)
    what = result['estimator']

    scores = np.abs(ytrain - xtrain @ what)
    return scores[:-1], scores[-1]

## 

def get_scores_gamp_loo_erm(x_stacked, y_stacked, prior_, likelihood_, **gamp_args):
    """
    Compared to the function get_scores_gamp_erm, here for each sample i, we remove it from the augmented training data
    and compute the estimator on the remaining data
    """   
    result = gamp.iterate_gamp(x_stacked, y_stacked, None, likelihood_, prior_, **gamp_args)
    what, vhat, omega = result['estimator'], result['variances'], result['omega']

    cavity_what = gamp.get_cavity_means_from_gamp(x_stacked, y_stacked, what, vhat, omega, likelihood_)

    scores = np.abs(y_stacked - np.diag(x_stacked @ cavity_what))
    return scores[:-1], scores[-1]

def get_gamp_loo_conformal_set(x_train, y_train, x_test, y_grid, prior_, likelihood_, coverage, **gamp_args):
    """
    coverage = 1 - alpha

    In gamp_args : use xhat_0 and vhat_0 as the result of AMP on x_train, y_train to initialize AMP near the convergence and reduce
    the number of iterations needed.
    """
    score_function = lambda x, y : get_scores_gamp_loo_erm(x, y, prior_, likelihood_, **gamp_args)

    n = len(x_train)
    x_stacked = np.vstack((x_train, x_test.reshape((1, -1))))
    y_stacked = np.hstack((y_train, 0.0))

    ys = []
    for y in y_grid:
        y_stacked[-1] = y
        scores_train, score_x = score_function(x_stacked, y_stacked)
        if score_x <= np.quantile(scores_train, q = np.ceil(coverage * (n + 1)) / n, interpolation='higher'):
            ys.append(y)

    return ys


## 

def get_jacknife_confidence_interval(x_train, y_train, x_test, likelihood_, prior_, coverage):
    """
    Here, coverage = 1 - 2 * alpha when we take the alpha and 1-alpha quantiles of the predictions + abs(residuals)
    """
    alpha = (1.0 - coverage) / 2.0
    n = len(x_train)
    result = gamp.iterate_gamp(x_train, y_train, None, likelihood_, prior_, tol=1e-4)
    what, vhat = result['estimator'], result['variances']
    what_cavity = gamp.get_cavity_means_from_gamp(x_train, y_train, what, vhat, result['omega'], likelihood_)

    train_residuals = np.diag(x_train @ what_cavity) - y_train
    lowers, uppers = what_cavity.T @ x_test - np.abs(train_residuals), what_cavity.T @ x_test + np.abs(train_residuals)

    return  np.quantile(lowers,  q=np.floor(alpha * (n+1))/n), np.quantile(uppers, q=np.ceil((1.-alpha) * (n+1))/n)