import numpy as np
from time import time

import uncertaintyutils as uu
import uncertaintyutils.data as data
import uncertaintyutils.gamp.gamp as gamp
import uncertaintyutils.gamp.likelihood as likelihood
import uncertaintyutils.gamp.prior as prior

def test_gamp_retrain():
    d = 128
    n = 256
    n_add = 32

    wstar, x, y = data.sample_data(n=n, d=d, likelihood="logit", noise_std=0.0)
    x_add, y_add= data.sample_data_from_teacher(n=n_add, teacher=wstar, likelihood="logit", noise_std=0.0)

    likelihood_ = likelihood.erm_logit_likelihood.ERMLogitLikelihood()
    prior_      = prior.gaussian_prior.GaussianPrior(lambda_ = 1.0)

    # compute the estimator by retraining
    begin_iterate = time()
    result = gamp.iterate_gamp(x, y, None, likelihood_, prior_, max_iter=200, tol=1e-4)
    end_iterate = time()
    what, vhat, g = result['estimator'], result['variances'], result['g_out']
    begin_retrain = time()
    result_retrain = gamp.retrain_gamp(what, vhat, x_add, y_add, likelihood_, prior_, x_old = x, y_old=y, g_old=g, tol=1e-4)
    end_retrain = time()

    what_retrain, vhat_retrain = result_retrain['estimator'], result_retrain['variances']

    # compute the estimator directly on the whole dataset
    x_whole = np.concatenate((x, x_add))
    y_whole = np.concatenate((y, y_add))
    begin_whole = time()
    result_whole_dataset = gamp.iterate_gamp(x_whole, y_whole, None, likelihood_, prior_, max_iter=200, tol=1e-4)
    end_whole = time()
    what_whole_dataset, vhat_whole_dataset = result_whole_dataset['estimator'], result_whole_dataset['variances']

    print(f'Time for first train and retrain are {end_iterate - begin_iterate} and {end_retrain - begin_retrain}')
    print(f'Time to train on whole dataset is {end_whole - begin_whole}')

    # check that the two estimators are close
    print(f'Distance between retrain and whole is {np.linalg.norm(what_retrain - what_whole_dataset)}, norms are {np.linalg.norm(what_retrain)}, {np.linalg.norm(what_whole_dataset)}')
    print(np.linalg.norm(vhat_retrain - vhat_whole_dataset))

if __name__ == "__main__":
    test_gamp_retrain()