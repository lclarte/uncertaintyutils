"""
Here we will do all the integrals directly because we are in the ridge case, so 
we can't accelerate the computation
"""
import numpy as np
import scipy.integrate as integrate
import scipy.linalg as linalg
import scipy.optimize as optimize
import scipy.stats as stats
from tqdm import tqdm

import gcmpyo3

BOUND = 5.0

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def hessian_logistic_loss(y, z_vec, weights_vec):
    return np.diag(weights_vec * (1.0 / np.cosh(y * z_vec)**2) / 4.0)

def gradient_logistic_loss(y, z_vec, weights_vec):
    return y * weights_vec * (sigmoid(y * z_vec) - 1.0)

def prox_logistic_multivariate(y, omega_vec, v_inv_mat, weights_vec):
    """
    Provide the inverse of V to accelerate the computations
    """
    def aux(z_vec):
        return weights_vec @ np.log(1.0 + np.exp(- y * z_vec)) + 0.5 * (z_vec - omega_vec) @ v_inv_mat @ (z_vec - omega_vec)
    
    def jac_aux(z_vec):
        return gradient_logistic_loss(y, z_vec, weights_vec) + v_inv_mat @ (z_vec - omega_vec)
    
    # NOTE : What's the best method to use ? 
    res = optimize.minimize(aux, x0=omega_vec, method='Newton-CG', jac=jac_aux, tol=1e-3)
    if res.success:
        return res.x
    else:
        return omega_vec
    
def gout_logistic_multivariate(y, omega_vec, v_inv_mat, weights_vec):
    return v_inv_mat @ (prox_logistic_multivariate(y, omega_vec, v_inv_mat, weights_vec) - omega_vec)

def dwgout_logistic_multivariate(y, omega_vec, v_inv_mat, weights_vec, v_mat):
    """
    NOTE : We ask for v_mat because we don't want to inverse v_inv_mat to save time 
    The expression of the derivative of the prox is (I_d + V H(loss)))^{-1}
    """
    k = len(omega_vec)
    prox = prox_logistic_multivariate(y, omega_vec, v_inv_mat, weights_vec)
    derivative_prox = np.linalg.inv( np.eye(k) + v_mat @ hessian_logistic_loss(y, prox, weights_vec) )
    return v_inv_mat @ (derivative_prox - np.eye(k))

# Function to model the different kind of resamples

weights_proba_function_bootstrap = lambda w1, w2 : stats.poisson.pmf(w1, mu=1.0) * stats.poisson.pmf(w2, mu=1.0)
weights_proba_function_full_resample = lambda w1, w2 : 0.5 if w1 != w2 else 0.0
def get_weights_proba_function_cross_validation(k):
    assert k >= 2, "k must be >= 2"
    def weights_proba_function(w1, w2):
        if w1 != w2:
            return 1.0 / k
        # the case w1 == w2 == 0.0 is because we model k estimators but run the state evolution for 2 of them 
        elif w1 == w2 == 0.0:
            return 1.0 - 2.0 / k
        else:
            return 0.0
    return weights_proba_function

## 

"""
Here, we have an integral over a 2-dimensional vector of function that are vector or matrix-valued function,
so what we do is we do the integral over the first dimension and then we do the integral over the second dimension
so functions called e.g. integrand_m_hat_vec_fixed_x1 is when the first dimension is fixed and we integrate over the second dimension
then in the function integrand_m_hat_vec we integrate over the 1st dimension
"""

def integrand_m_hat_vec(y, omega_vec, v_inv_mat, conditional_mean, conditional_variance, weights_vec):
    gout = gout_logistic_multivariate(y, omega_vec, v_inv_mat, weights_vec)
    return gout * gcmpyo3.Logit(noise_variance=0).call_dz0(y, conditional_mean, conditional_variance)

def integrand_m_hat_vec_fixed_x1(x1, y, m_vec, q_inv_mat, q_sqrt_mat, v_inv_mat, conditional_variance, weights_vec, limit_quad_vec):
    return np.exp(- x1**2 / 2.0) * integrate.quad_vec(lambda x2 : np.exp(-x2**2 / 2.0) * integrand_m_hat_vec(y, q_sqrt_mat @ [x1, x2], v_inv_mat, m_vec @ q_inv_mat @ q_sqrt_mat @ [x1, x2], conditional_variance, weights_vec),
                                                     -BOUND, BOUND, limit=limit_quad_vec)[0] / (2 * np.pi)

"""
NOTE : Since quad_vec cannot integrate matrices, we return the first line of the matrix and we rebuild the matrix when we return 
q_hat_mat and v_hat_mat
"""

def integrand_q_hat_vec(y, omega_vec, v_inv_mat, conditional_mean, conditional_variance, weights_vec):
    """
        As explained, we return the 1st row of the matrix
    """
    gout = gout_logistic_multivariate(y, omega_vec, v_inv_mat, weights_vec).reshape((-1, 1))
    return (gout @ gout.T)[0] * gcmpyo3.Logit(noise_variance=0).call_z0(y, conditional_mean, conditional_variance)

def integrand_q_hat_vec_fixed_x1(x1, y, m_vec, q_inv_mat, q_sqrt_mat, v_inv_mat, conditional_variance, weights_vec, limit_quad_vec):
    return np.exp(-x1**2 / 2.0 ) * integrate.quad_vec(lambda x2 : np.exp(- x2**2 / 2.0) * integrand_q_hat_vec(y, q_sqrt_mat @ [x1, x2], v_inv_mat, m_vec @ q_inv_mat @ q_sqrt_mat @ [x1, x2], conditional_variance, weights_vec),
                                                      -BOUND, BOUND, limit=limit_quad_vec)[0] / (2 * np.pi)

# 

def integrand_v_hat_vec(y, omega_vec, v_mat, v_inv_mat, conditional_mean, conditional_variance, weights_vec):
    dgout = dwgout_logistic_multivariate(y, omega_vec, v_inv_mat, weights_vec, v_mat)[0]
    return dgout * gcmpyo3.Logit(noise_variance=0).call_z0(y, conditional_mean, conditional_variance)

def integrand_v_hat_vec_fixed_x1(x1, y, m_vec, q_inv_mat, q_sqrt_mat, v_mat, v_inv_mat, conditional_variance, weights_vec, limit_quad_vec):
    return np.exp(-x1**2 / 2.0 ) * integrate.quad_vec(lambda x2 : np.exp(- x2**2 / 2.0) * integrand_v_hat_vec(y, q_sqrt_mat @ [x1, x2], v_mat, v_inv_mat, m_vec @ q_inv_mat @ q_sqrt_mat @ [x1, x2], conditional_variance, weights_vec),
                                                      -BOUND, BOUND, limit=limit_quad_vec)[0] / (2 * np.pi)

def update_hatoverlaps_fixed_weights(m_vec, q_mat, v_mat, rho_float, weights_vec, limit_quad_vec):
    """
    # NOTE : This is the part that changes compared to the Ridge case
    # NOTE 2 : https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix to compute the square root explicitely

    we update without the alpha so we'll need to multiply later
    # NOTE : Here we assume the student noise variance is 1,
    otherwise we can divide the weights_vec -> weights_vec / noise_variance
    # NOTE 2 : Having a student_noise variance different from 1 is useless because
    we can just rescale the regularization by the student noise variance
    """
    k = len(m_vec)
    q_sqrt_mat = linalg.sqrtm(q_mat)
    q_inv_mat = np.linalg.inv(q_mat)

    m_hat_vec = np.zeros_like(m_vec)
    q_hat_vec = np.zeros_like(m_vec)
    v_hat_vec = np.zeros_like(m_vec)

    v_inv_mat = np.linalg.inv(v_mat)
    
    conditional_variance = rho_float - m_vec @ q_inv_mat @ m_vec

    ys = [-1, 1]

    for y in ys:
        print(f'    {y=}')
        m_hat_vec += integrate.quad_vec(lambda x1 : integrand_m_hat_vec_fixed_x1(x1, y, m_vec, q_inv_mat, q_sqrt_mat, v_inv_mat, conditional_variance, weights_vec, limit_quad_vec=limit_quad_vec), -BOUND, BOUND, limit=limit_quad_vec)[0]
        q_hat_vec += integrate.quad_vec(lambda x1 : integrand_q_hat_vec_fixed_x1(x1, y, m_vec, q_inv_mat, q_sqrt_mat, v_inv_mat, conditional_variance, weights_vec, limit_quad_vec=limit_quad_vec), -BOUND, BOUND, limit=limit_quad_vec)[0]
        v_hat_vec += integrate.quad_vec(lambda x1 : integrand_v_hat_vec_fixed_x1(x1, y, m_vec, q_inv_mat, q_sqrt_mat, v_mat, v_inv_mat, conditional_variance, weights_vec, limit_quad_vec=limit_quad_vec), -BOUND, BOUND, limit=limit_quad_vec)[0]

    return m_hat_vec, np.array([[q_hat_vec[0], q_hat_vec[1]], [q_hat_vec[1], q_hat_vec[0]]]), np.array([[v_hat_vec[0], v_hat_vec[1]], [v_hat_vec[1], v_hat_vec[0]]])

def update_hatoverlaps(m_vec, q_mat, v_mat, rho_float, alpha,  
                        weights_bound, weights_proba_function, limit_quad_vec):
    """
    Here, gout -> vector, dgout -> matrix and we need to integrate over the 
    sampling weights
    NOTE : For now we only to the Poisson case, k = 2

    weights_proba_function : takes two integers as an argument and returns a proba.
    For bootstrap w/ independent resamples : 
        weights_proba_function = lambda w1, w2 : stats.poisson.pmf(w1, mu=1.0) * stats.poisson.pmf(w2, mu=1.0)
    For full resample of X, y : 
        weights_proba_function = lambda w1, w2 : 0.5 if w1 != w2 else 0.0
    """
    assert len(m_vec) == 2, "Only implemented for k = 2"

    mhat_vec = np.zeros(2)
    qhat_mat = np.zeros((2, 2))
    vhat_mat = np.zeros((2, 2))

    for w1 in range(weights_bound):
        for w2 in range(weights_bound):
            print(f'{w1=}, {w2=}')
            weights_vec = np.array([w1, w2])
            proba = weights_proba_function(w1, w2)
            tmp_mhat_vec, tmp_qhat_mat, tmp_vhat_mat = update_hatoverlaps_fixed_weights(m_vec, q_mat, v_mat, rho_float, weights_vec, limit_quad_vec=limit_quad_vec)
            mhat_vec += proba * tmp_mhat_vec
            qhat_mat += proba * tmp_qhat_mat
            vhat_mat += proba * tmp_vhat_mat
            
    return alpha * mhat_vec, alpha * qhat_mat, alpha * vhat_mat

# 

def update_overlaps(hat_m_vec, hat_q_mat, hat_v_mat, lambda_ridge):
    assert len(hat_m_vec) == 2, "Only implemented for k = 2"
    k = len(hat_m_vec)
    hat_m_vec_reshape = np.reshape(hat_m_vec, (k, 1))
    ridge_mat         = np.linalg.inv(lambda_ridge * np.eye(k) + hat_v_mat)
    m_vec             = ridge_mat @ hat_m_vec
    q_mat             = ridge_mat @ (hat_m_vec_reshape @ hat_m_vec_reshape.T + hat_q_mat) @ ridge_mat.T
    v_mat             = ridge_mat

    return m_vec, q_mat, v_mat

def state_evolution(alpha, lambda_ridge, max_iter=100, tol=1e-5, weights_bound=5, weights_proba_function=weights_proba_function_bootstrap,
                    verbose=False, limit_quad_vec = 1000):
    """
    Note : weight_bound is NOT included in the weights, e.g. if weight_bound = 5, the max weight is 4
    so for full resample we need to set weight_bound = 2
    """
    q_mat = 0.99 * np.eye(2)
    m_vec = np.array([0.1, 0.1])
    v_mat = np.eye(2) + 0.01 * np.ones((2, 2))
    rho_float = 1.0

    qhat_old = np.zeros((2, 2))

    for t in range(max_iter):
        print(f'{t=}')
        print('Updating hat overlaps')
        hat_m_vec, hat_q_mat, hat_v_mat = update_hatoverlaps(m_vec, q_mat, v_mat, rho_float, alpha, 
                                                            weights_bound=weights_bound, weights_proba_function=weights_proba_function,
                                                            limit_quad_vec=limit_quad_vec)
        
        print('Updating overlaps')
        m_vec, q_mat, v_mat = update_overlaps(hat_m_vec, hat_q_mat, hat_v_mat, lambda_ridge)

        diff = np.max(np.abs(qhat_old - hat_q_mat))
        if diff < tol:
            break

        if verbose:
            print(f'{t=}, {diff=}')
            print('m_vec = ', m_vec)
            print('q_mat = ', q_mat)
            print('v_mat = ', v_mat)

        qhat_old = hat_q_mat.copy()

    return m_vec, q_mat, v_mat, hat_m_vec, hat_q_mat, hat_v_mat