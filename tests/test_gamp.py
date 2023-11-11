import numpy as np
from time import time

import matplotlib.pyplot as plt
import sklearn.linear_model as linear_model

import uncertaintyutils as uu
import uncertaintyutils.data as data
import uncertaintyutils.gamp.gamp as gamp
import uncertaintyutils.gamp.likelihood as likelihood
import uncertaintyutils.gamp.prior as prior
import uncertaintyutils.gamp.prior.laplace_prior as laplace_prior
import uncertaintyutils.gamp.prior.lasso_prior as lasso_prior

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

def test_gamp_lasso():
    """
    Test the lasso prior
    """
    d = 50
    n = 100
    wstar = data.sample_teacher(np.eye(d))
    x = np.random.normal(0.0, 1.0, size=(n, d)) / np.sqrt(d)
    y = x @ wstar

    lambda_lasso = 1.0 / n

    lr_lasso = linear_model.Lasso(alpha=lambda_lasso, fit_intercept=False)
    lr_lasso.fit(x, y)

    prior_ = lasso_prior.LassoPrior(lambda_ = 1.0)
    likelihood_ = likelihood.gaussian_log_likelihood.GaussianLogLikelihood(noise_variance=1.0)

    result = gamp.iterate_gamp(x, y, None, likelihood_, prior_, max_iter=10, tol=1e-4, verbose=True)
    what, vhat = result['estimator'], result['variances']
    print(f'Correlation between AMP and Lasso is {lr_lasso.coef_ @ what / np.linalg.norm(lr_lasso.coef_) / np.linalg.norm(what)}')
    plt.scatter(lr_lasso.coef_, what)
    plt.xlabel('Lasso')
    plt.ylabel('AMP')
    plt.show()

def test_gamp_laplace():
    d = 100
    noise_std = 1.0
    wstar = np.random.laplace(0.0, 1.0, size=d)
    rho = np.linalg.norm(wstar)**2 / d

    n_list = list(map(int, np.linspace(50, 1000, 30)))
    q_list, m_list, v_list = [], [], []
    
    for n in n_list:
        x = np.random.normal(0.0, 1.0, size=(n, d)) / np.sqrt(d)
        y = x @ wstar + noise_std * np.random.normal(0.0, 1.0, size=n)

        # prior_ = laplace_prior.LaplacePrior(lambda_ = 1.0)
        prior_ = lasso_prior.LassoPrior(lambda_ = 1.0)
        likelihood_ = likelihood.gaussian_log_likelihood.GaussianLogLikelihood(noise_variance=noise_std**2)

        result = gamp.iterate_gamp(x, y, None, likelihood_, prior_, max_iter=50, tol=1e-4, verbose=False)
        what, vhat = result['estimator'], result['variances']

        q_list.append(what @ what / d)
        m_list.append(what @ wstar / d)
        v_list.append(np.mean(vhat))

    plt.plot(n_list, q_list, label='q')
    plt.plot(n_list, m_list, label='m')
    plt.plot(n_list, v_list, label='v')
    plt.legend()
    plt.show()

    plt.plot(n_list, np.array(m_list) / np.sqrt(q_list) / np.sqrt(rho))
    plt.show()


if __name__ == "__main__":
    test_gamp_laplace()