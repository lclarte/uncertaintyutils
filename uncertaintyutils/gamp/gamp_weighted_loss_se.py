"""
Here we will do all the integrals directly because we are in the ridge case, so 
we can't accelerate the computation
"""
import numpy as np
import scipy.stats as stats
from tqdm import tqdm

"""
#NOTE : the functions are useless but I keep them for reference 
def z0(y, omega, v, noise_variance):
    # Here, omega and v are scalar because it's the teacher channel that's scalar valued
    return np.exp(- 0.5 * (y - omega)**2 / (noise_variance + v)) / np.sqrt(2.0 * np.pi * (noise_variance + v))

def dz0(y, omega, v, noise_variance):
        return (y - omega) / (v + noise_variance) * np.exp(- 0.5 * (y - omega).powi(2) / (noise_variance + v)) / np.sqrt(2.0 * np.pi * (noise_variance + v))
"""

# Function to model the different kind of resamples

weights_proba_function_bootstrap = lambda w1, w2 : stats.poisson.pmf(w1, mu=1.0) * stats.poisson.pmf(w2, mu=1.0)


weights_proba_function_full_resample = lambda w1, w2 : 0.5 if (w1 == 1.0 and w2 == 0.0) or (w1 == 0.0 and w2 == 1.0) else 0.0
# with this proba functio we'll look at the correlation between the average of bootstrap and ERM
# the second estimator will be ERM so the proba is 0 if w2 != 1
weights_proba_function_bootstrap_erm = lambda w1, w2 : stats.poisson.pmf(w1, mu=1.0) if w2 == 1.0 else 0.0

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


def update_hatoverlaps_fixed_weights(m_vec, q_mat, v_mat, rho_float, weights_vec, teacher_noise_variance):
    """
    we update without the alpha so we'll need to multiply later
    # NOTE : Here we assume the student noise variance is 1,
    otherwise we can divide the weights_vec -> weights_vec / noise_variance
    # NOTE 2 : Having a student_noise variance different from 1 is useless because
    we can just rescale the regularization by the student noise variance
    """
    k = len(weights_vec)
    inv_q_mat   = np.linalg.inv(q_mat)
    # as in the scalar case, the teacher noise is just added to rho (its square norm)
    vstar_float = rho_float + teacher_noise_variance - m_vec @ inv_q_mat @ m_vec
    bias_mat    = np.vstack([m_vec.reshape((1, 2)), m_vec.reshape((1, 2))]) @ inv_q_mat - np.eye(k)
    # gout_mat is the matrix used in gout, e.g. gout = gout_mat @ (y - omega), dgout = - gout_mat
    gout_mat = np.linalg.inv(np.eye(k) + np.diag(weights_vec) @ v_mat) @ np.diag(weights_vec)
    
    mhat_vec = gout_mat @ np.ones(2)
    qhat_mat = gout_mat @ (vstar_float * np.ones((2, 2)) + bias_mat @ q_mat @ bias_mat.T) @ gout_mat.T
    vhat_mat = gout_mat

    return mhat_vec, qhat_mat, vhat_mat

def update_hatoverlaps(m_vec, q_mat, v_mat, rho_float, alpha, teacher_noise_variance, 
                        weights_bound_1, weights_bound_2, weights_proba_function):
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

    for w1 in range(weights_bound_1):
        for w2 in range(weights_bound_2):
            weights_vec = np.array([w1, w2])
            proba = weights_proba_function(w1, w2)
            tmp_mhat_vec, tmp_qhat_mat, tmp_vhat_mat = update_hatoverlaps_fixed_weights(m_vec, q_mat, v_mat, rho_float, weights_vec, teacher_noise_variance)
            mhat_vec += proba * tmp_mhat_vec
            qhat_mat += proba * tmp_qhat_mat
            vhat_mat += proba * tmp_vhat_mat
            
    return alpha * mhat_vec, alpha * qhat_mat, alpha * vhat_mat

# 

def update_overlaps(hat_m_vec, hat_q_mat, hat_v_mat, lambda_ridge):
    assert len(hat_m_vec) == 2, "Only implemented for k = 2"
    k = len(hat_m_vec)
    hat_m_vec_reshape = np.reshape(hat_m_vec, (k, 1))
    ridge_mat = np.linalg.inv(lambda_ridge * np.eye(k) + hat_v_mat)
    m_vec = ridge_mat @ hat_m_vec
    q_mat = ridge_mat @ (hat_m_vec_reshape @ hat_m_vec_reshape.T + hat_q_mat) @ ridge_mat.T
    v_mat = ridge_mat

    return m_vec, q_mat, v_mat

def state_evolution(alpha, lambda_ridge, teacher_noise_variance, max_iter=100, tol=1e-5, weights_bound_1=5, weights_bound_2=5, weights_proba_function=weights_proba_function_bootstrap):
    """
    Note : weight_bound is NOT included in the weights, e.g. if weight_bound = 5, the max weight is 4
    so for full resample we need to set weight_bound = 2
    """
    q_mat = 0.99 * np.eye(2)
    m_vec = np.array([0.1, 0.1])
    v_mat = np.eye(2) + 0.01 * np.ones((2, 2))
    rho_float = 1.0

    q_old = np.zeros((2, 2))

    for t in range(max_iter):
        hat_m_vec, hat_q_mat, hat_v_mat = update_hatoverlaps(m_vec, q_mat, v_mat, rho_float, alpha, teacher_noise_variance, 
                                                            weights_bound_1=weights_bound_1, weights_bound_2=weights_bound_2, weights_proba_function=weights_proba_function)
        m_vec, q_mat, v_mat = update_overlaps(hat_m_vec, hat_q_mat, hat_v_mat, lambda_ridge)

        diff = np.max(np.abs(q_old - q_mat))
        if diff < tol:
            break

        q_old = q_mat.copy()

    if t == max_iter - 1:
        print(f"Warning : state evolution did not converge, reached {max_iter} iterations, {diff=}")

    return m_vec, q_mat, v_mat, hat_m_vec, hat_q_mat, hat_v_mat

