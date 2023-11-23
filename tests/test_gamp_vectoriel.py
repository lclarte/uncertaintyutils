import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as linear_model
from sklearn.utils import resample
from tqdm import tqdm

import uncertaintyutils.data as data
import uncertaintyutils.gamp.gamp_weighted_loss as gamp_weighted_loss
from uncertaintyutils.gamp.gamp_weighted_loss_se import state_evolution

def basic_test():
    d = 128
    k = 5
    sampling_ratio = 3.0
    noise_std = 1.0
    lambda_ = 1e-2
    n = int(sampling_ratio * d)

    wstar = data.sample_teacher(np.eye(d))
    x_train = np.random.normal(0.0, 1.0, size=(n, d)) / np.sqrt(d)
    y_train = x_train @ wstar + noise_std * np.random.normal(0.0, 1.0, size=(n,))
    weights  = np.random.poisson(lam=1.0, size=(n, k))

    channel = gamp_weighted_loss.WeightedRidgeChannel(noise_variance = 1.0, k = k)
    denoiser = gamp_weighted_loss.GaussianDenoiser(lambda_ = lambda_, k = k)

    result = gamp_weighted_loss.iterate_gamp(x_train, y_train, weights, channel, denoiser, max_iter = 100, verbose=True, tol=1e-3)

    what, vhat = result['estimator'], result['variances']

    print('With GAMP :')
    print('m = ', what.T @ wstar / d)
    print('q = ', what.T @ what / d)

    # do the same stuff with sklearn
    lr_coefs = np.zeros((d, k))

    for i in range(k):
        x_train_resample, y_train_resample = resample(x_train, y_train)

        lr = linear_model.Ridge(alpha = lambda_, fit_intercept=False)
        lr.fit(x_train_resample, y_train_resample)
        lr_coefs[:, i] = lr.coef_

    print('With sklearn :')
    print('m = ', lr_coefs.T @ wstar / d)
    print('q = ', lr_coefs.T @ lr_coefs / d)

def plot_overalps_amp_vs_true_bootstrap(random_seed = 0):
    np.random.seed(random_seed)
    d = 256
    noise_std = 1.0
    k = 5
    lambda_ = 0.5

    wstar = data.sample_teacher(np.eye(d))
    wstar = np.sqrt(d) * wstar / np.linalg.norm(wstar)

    # q0 is diagonal term (norm of one estimator), q1 is off-diagonal term (correlation between two estimators)
    q0_list, q1_list, m_list = [], [], []
    sklearn_q0_list, sklearn_q1_list, sklearn_m_list = [], [], []
    se_q0_list, se_q1_list, se_m_list = [], [], []

    sampling_ratio_range = np.arange(1., 10.0, 1.0)
    se_sampling_ratio_range = np.arange(1.0, 10.0, 0.1)

    for sampling_ratio in tqdm(sampling_ratio_range):
        n = int(sampling_ratio * d)
        x_train = np.random.normal(0.0, 1.0, size=(n, d)) / np.sqrt(d)
        y_train = x_train @ wstar + noise_std * np.random.normal(0.0, 1.0, size=(n,))

        # weights for GAMP
        weights  = np.random.poisson(lam=1.0, size=(n, k))

        # use noise_variance = 1.0 to match sklearn implementation
        channel = gamp_weighted_loss.WeightedRidgeChannel(noise_variance = 1.0, k = k)
        denoiser = gamp_weighted_loss.GaussianDenoiser(lambda_ = lambda_, k = k)

        result = gamp_weighted_loss.iterate_gamp(x_train, y_train, weights, channel, denoiser, max_iter = 100, verbose=False)
        what, vhat = result['estimator'], result['variances']

        Q_matrix = what.T @ what / d
        q0_list.append(np.mean(np.diag(Q_matrix)))
        q1_list.append(
            (np.sum(Q_matrix) - np.trace(Q_matrix)) / (k * (k-1))
        )
        m_list.append(np.mean(what.T @ wstar / d))

        # do the same stuff with sklearn
        lr_coefs = np.zeros((d, k))

        for i in range(k):
            x_train_resample, y_train_resample = resample(x_train, y_train)

            lr = linear_model.Ridge(alpha =lambda_, fit_intercept=False)
            lr.fit(x_train_resample, y_train_resample)
            lr_coefs[:, i] = lr.coef_

        sklearn_Q_matrix = lr_coefs.T @ lr_coefs / d
        sklearn_q0_list.append(np.mean(np.diag(sklearn_Q_matrix)))
        sklearn_q1_list.append(
            (np.sum(sklearn_Q_matrix) - np.trace(sklearn_Q_matrix)) / (k * (k-1))
        )
        sklearn_m_list.append(np.mean(lr_coefs.T @ wstar / d))

    for sampling_ratio in tqdm(se_sampling_ratio_range):
        # compute the overlaps with state evolution
        se_m, se_q, se_v, _, _, _ = state_evolution(sampling_ratio, lambda_, noise_std**2)
        se_m_list.append(np.mean(se_m))
        se_q0_list.append(np.mean(np.diag(se_q)))
        se_q1_list.append((np.sum(se_q) - np.trace(se_q)) / 2)
        
    
    plt.scatter(sampling_ratio_range, q0_list,marker='^', c='b', label='q0 (GAMP)')
    plt.scatter(sampling_ratio_range, q1_list,marker='^', c='r', label='q1 (GAMP)')
    plt.scatter(sampling_ratio_range, m_list, marker='^', c='g', label='m (GAMP)')

    plt.scatter(sampling_ratio_range, sklearn_q0_list,marker='x',c='b', label='q0 (sklearn)')
    plt.scatter(sampling_ratio_range, sklearn_q1_list,marker='x',c='r', label='q1 (sklearn)')
    plt.scatter(sampling_ratio_range, sklearn_m_list, marker='x',c='g',label='m (sklearn)')

    plt.plot(se_sampling_ratio_range, se_q0_list, c='b', label='q0 (SE)')
    plt.plot(se_sampling_ratio_range, se_q1_list, c='r', label='q1 (SE)')
    plt.plot(se_sampling_ratio_range, se_m_list, c='g',label='m (SE)')

    plt.xlabel('Sampling ratio')
    plt.ylabel('Overlap')

    plt.legend()
    plt.grid()
    plt.title(f'GAMP vs true bootstrap, {d=}, {lambda_=}')
    plt.show()

def plot_overlaps_amp_vs_true_full_resample(random_seed=0):
    np.random.seed(random_seed)
    d = 256
    noise_std = 1.0
    # need k = 2 because we resample the full training data once
    k = 5
    lambda_ = 1e-2

    wstar = data.sample_teacher(np.eye(d))

    # q0 is diagonal term (norm of one estimator), q1 is off-diagonal term (correlation between two estimators)
    q0_list, q1_list, m_list = [], [], []
    sklearn_q0_list, sklearn_q1_list, sklearn_m_list = [], [], []

    sampling_ratio_range = np.arange(1.0, 15.0, 1.0)
    for sampling_ratio in tqdm(sampling_ratio_range):
        n = int(sampling_ratio * d)
        # sample 2 training sets for the "full resample"
        x_train = np.random.normal(0.0, 1.0, size=(k*n, d)) / np.sqrt(d)
        y_train = x_train @ wstar + noise_std * np.random.normal(0.0, 1.0, size=(k*n,))

        # weights for GAMP of size (2*n, k)
        weights = np.zeros((k*n, k))
        for i in range(k):
            weights[i*n:(i+1)*n, i] = 1.0

        # use noise_variance = 1.0 to match sklearn implementation
        channel = gamp_weighted_loss.WeightedRidgeChannel(noise_variance = 1.0, k = k)
        denoiser = gamp_weighted_loss.GaussianDenoiser(lambda_ = lambda_, k = k)

        result = gamp_weighted_loss.iterate_gamp(x_train,  y_train, weights, channel, denoiser, max_iter = 100, verbose=False)
        what, vhat = result['estimator'], result['variances']

        Q_matrix = what.T @ what / d
        q0_list.append(np.mean(np.diag(Q_matrix)))
        q1_list.append( (np.sum(Q_matrix) - np.trace(Q_matrix)) / (k * (k-1)) ) 
        m_list.append(np.mean(what.T @ wstar / d))

        # do the same stuff with sklearn
        lr_coefs = np.zeros((d, k))

        for i in range(k):
            x_train = np.random.normal(0.0, 1.0, size=(n, d)) / np.sqrt(d)
            y_train = x_train @ wstar + noise_std * np.random.normal(0.0, 1.0, size=(n,))

            lr = linear_model.Ridge(alpha =lambda_, fit_intercept=False)
            lr.fit(x_train, y_train)
            lr_coefs[:, i] = lr.coef_

        sklearn_Q_matrix = lr_coefs.T @ lr_coefs / d
        sklearn_q0_list.append(np.mean(np.diag(sklearn_Q_matrix)))
        sklearn_q1_list.append( (np.sum(sklearn_Q_matrix) - np.trace(sklearn_Q_matrix)) / (k * (k-1)) ) 
        sklearn_m_list.append(np.mean(lr_coefs.T @ wstar / d))
        
    plt.scatter(sampling_ratio_range, q0_list,marker='^', c='b', label='q0 (GAMP)')
    plt.scatter(sampling_ratio_range, q1_list,marker='^', c='r', label='q1 (GAMP)')
    plt.scatter(sampling_ratio_range, m_list, marker='^', c='g', label='m (GAMP)')

    plt.scatter(sampling_ratio_range, sklearn_q0_list,marker='x',c='b', label='q0 (sklearn)')
    plt.scatter(sampling_ratio_range, sklearn_q1_list,marker='x',c='r', label='q1 (sklearn)')
    plt.scatter(sampling_ratio_range, sklearn_m_list, marker='x',c='g',label='m (sklearn)')

    plt.xlabel('Sampling ratio')
    plt.ylabel('Overlap')

    plt.legend()
    plt.grid()
    plt.title(f'GAMP vs full resample, {d=}, {lambda_=}, {k=}')
    plt.show()

if __name__ == '__main__':
    # basic_test()
    plot_overalps_amp_vs_true_bootstrap()
    # plot_overlaps_amp_vs_true_full_resample()