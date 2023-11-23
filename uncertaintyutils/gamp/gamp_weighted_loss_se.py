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

def update_hatoverlaps_fixed_weights(m_vec, q_mat, v_mat, rho_float, weights_vec, teacher_noise_variance):
    """
    we update without the alpha so we'll need to multiply later
    # NOTE : Here we assume the student noise variance is 1,
    otherwise we can divide the weights_vec -> weights_vec / noise_variance
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

def update_hatoverlaps(m_vec, q_mat, v_mat, rho_float, alpha, teacher_noise_variance):
    """
    Here, gout -> vector, dgout -> matrix and we need to integrate over the 
    sampling weights
    NOTE : For now we only to the Poisson case, k = 2
    """
    assert len(m_vec) == 2, "Only implemented for k = 2"

    mhat_vec = np.zeros(2)
    qhat_mat = np.zeros((2, 2))
    vhat_mat = np.zeros((2, 2))

    bound = 5
    for w1 in range(bound):
        for w2 in range(bound):
            weights_vec = np.array([w1, w2])
            proba = stats.poisson.pmf(w1, mu=1.0) * stats.poisson.pmf(w2, mu=1.0)
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

def state_evolution(alpha, lambda_ridge, teacher_noise_variance, max_iter=100, tol=1e-5):
    q_mat = 0.99 * np.eye(2)
    m_vec = np.array([0.1, 0.1])
    v_mat = np.eye(2) + 0.01 * np.ones((2, 2))
    rho_float = 1.0

    qhat_old = np.zeros((2, 2))

    for t in range(max_iter):
        hat_m_vec, hat_q_mat, hat_v_mat = update_hatoverlaps(m_vec, q_mat, v_mat, rho_float, alpha, teacher_noise_variance)
        m_vec, q_mat, v_mat = update_overlaps(hat_m_vec, hat_q_mat, hat_v_mat, lambda_ridge)

        diff = np.max(np.abs(qhat_old - hat_q_mat))
        if diff < tol:
            break

        qhat_old = hat_q_mat.copy()

    return m_vec, q_mat, v_mat, hat_m_vec, hat_q_mat, hat_v_mat