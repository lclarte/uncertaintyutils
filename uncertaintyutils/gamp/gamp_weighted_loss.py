"""
gamp_weighted_loss.py
Implementatin of GAMP as done in https://arxiv.org/pdf/1806.05451.pdf 
where we have an additional quenched disorder for each of the K estimators, and our channel / denoiser are 
K-dimensional
"""

import numpy as np

def multiply_3d_by_2d(x, y):
    return np.squeeze(np.matmul(x, y[:, :, np.newaxis]), axis=-1)

def iterate_gamp(x_mat, y_vec, weights_mat, vec_channel, vec_denoiser, tol = 1e-5, max_iter = 10, verbose = False):
    """
    x_mat : size n x d
    y_vec : size n
    weights_mat : size n x K
    
    vec_channel : returns a vector of size K, uses weights_mat in addition to the usual arguments (y, omega, v)
    vec_denoiser : returns a vector of size K
    """
    n, d        = x_mat.shape
    _, k        = weights_mat.shape
    
    x_squared_mat = x_mat * x_mat
    g             = np.zeros((n, k))
    sigma         = np.array([np.eye(k) for _ in range(d)])
    # sigma_inv is the same as A
    sigma_inv     = np.copy(sigma)

    what          = np.zeros((d, k))
    # each row of vhat is the covariance between the K estimators for index 1 <= i <= d
    vhat          = np.array([np.eye(k) for _ in range(d)])

    for t in range(max_iter):
        what_old = np.copy(what)
        # Update omega and v
        # np.matmul(sigma_inv, np.matmul(vhat, sigma)) is of size (d, k, k) so that onsager : (d, k)
        onsager_mat = np.matmul(sigma_inv, np.matmul(vhat, sigma))
        omega_vec = x_mat @ what - np.einsum('mi,ikl,ml->mk', x_squared_mat, onsager_mat, g) # omega : (n, k)
        v_mat     = np.einsum('mi,ikl->mkl', x_squared_mat, vhat) # v : (n, k, k)

        # Compute g, dg (of size n, k and size n, k, k)
        g, dg = vec_channel.channel(y_vec, weights_mat, omega_vec, v_mat)

        # update a, b (of shape d, k, k and shape d, k)
        sigma_inv = - np.einsum('mi,mkl->ikl', x_squared_mat, dg) # sigma_inv is the same as A in the paper and has shape d, k, k
        sigma     = np.linalg.inv(sigma_inv)
        # np.squeeze is used to remvoe the last dimension of size 1 that we added to use np.matmul
        # other ways b_vec         = np.squeeze(np.matmul(sigma_inv, what[:, :, np.newaxis]), axis=-1) + x_mat.T @ g
        b_vec         = multiply_3d_by_2d(sigma_inv, what) + x_mat.T @ g

        # update the marginals what, vhat that are now matrices of size K x d
        what, vhat = vec_denoiser.denoiser(b_vec, sigma_inv)

        diff = np.max(np.linalg.norm(what - what_old, axis=1))
        if diff < tol:
            break

        if verbose:
            print(f'Iteration {t}, difference is {diff}')

    return {
        'estimator' : what,
        'variances' : vhat
    }

class WeightedRidgeChannel:
    """
    Channel corresponding to the loss
    \sum_{k = 1}^K alpha_k (y - w_k @ x)^2 / noise_variance

    the argument weight_mats will be a n X d matrix
    """
    def __init__(self, noise_variance, k):
        # NOTE : To take into account the student noise we can just rescale the 
        # weights by noise_variance
        # TODO : Redo the computation and check this rigorously
        self.noise_variance = noise_variance    
        self.k = k

    def channel(self, y_vec, weights_mat, omega_vec, v_mat):
        """
        Since I think that in the vector case  gout = V^{-1} (prox - omega), I get in the end the expression
        gout = V^{-1} @ \Sigma^{-1} @ A ( y - omega )
        where : 
        A = diag(alpha_1, ..., alpha_K) with alpha the weights for each estimator
        \Sigma = V^{-1} + A
        """
        # y_mat has shape (n, k)
        y_mat = np.tile(y_vec, (self.k, 1)).T
        # weights_mat_diag is a (n, k, k) matrix where weights_mat_diag[i] = diag(weights_mat[i])
        A = np.array([np.diag(w) for w in weights_mat]) / self.noise_variance
        v_mat_inv = np.linalg.inv(v_mat)
        Sigma_inv = np.linalg.inv(v_mat_inv + A)
        tmp = np.matmul(np.matmul(v_mat_inv, Sigma_inv), A)
        # print(tmp.shape, y_vec.shape, omega_vec.shape)
        return multiply_3d_by_2d(tmp, y_mat - omega_vec), - tmp

class GaussianDenoiser:
    def __init__(self, lambda_, k : int):
        self.lambda_ = lambda_
        self.k = k
            
    def denoiser(self, b_vec, a_mat):
        inverse = np.linalg.inv(self.lambda_ * np.eye(self.k) + a_mat)
        return multiply_3d_by_2d(inverse, b_vec), inverse
    