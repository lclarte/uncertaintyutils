"""
Code to to relaxted belief propagation, has a higher complexity than AMP but is more accurate
"""

import numpy as np

def iterate_bp(x, y, prior, likelihood, n_iter, tol = 1e-3, verbose=False, damp=0.8):
    n, d = x.shape
    cavity_means, cavity_variances = np.zeros((d, n)), np.ones((d, n))
    V, omega = np.zeros((n, d)), np.zeros((n, d))
    diff = 0.0

    x2 = x * x

    for t in range(n_iter):
        old_cavity_means = cavity_means.copy()
        old_cavity_variances = cavity_variances.copy()
        
        for mu in range(n):
            for i in range(d):
                V[mu, i] = sum(x2[mu, j] * cavity_variances[j, mu] for j in range(d) if j != i)
                omega[mu, i] = sum(x[mu, j] * cavity_means[j, mu] for j in range(d) if j != i)
        
        A, B = compute_message_node_to_variable_from_V_omega(x, x2, y, V, omega, likelihood)

        R, Sigma = compute_message_variable_to_node_from_A_B(A, B)

        for mu in range(n):
            for i in range(d):
                # a_{i \to \mu}
                cavity_means[i, mu] = prior.fa(Sigma=Sigma[i, mu], R=R[i, mu])
                # v_{i \to \mu}
                cavity_variances[i, mu] = prior.fv(Sigma=Sigma[i, mu], R=R[i, mu])
    
        cavity_variances = (1.0 - damp) * cavity_variances + damp * old_cavity_variances
        cavity_means     = (1.0 - damp) * cavity_means     + damp * old_cavity_means
        
        diff = np.mean(np.abs(old_cavity_means - cavity_means))

        if verbose:
            print(f'iteration = {t}, diff = {diff}')
        if diff < tol:
            break

    what, vhat = compute_mean_variance_from_A_B(A, B, prior)

    return {
        'cavity_means' : cavity_means,
        'cavity_variances' : cavity_variances,
        'estimator' : what,
        'variances' : vhat
    }

def compute_message_node_to_variable_from_V_omega(x, x_squared, y, V, omega, likelihood):
    """
    Compute the parmeters A and B in the relaxed BP (equation 144 in 1511.02476), which parametrize
    the message m_{\mu \to i}, from V and omega
    """
    n, d = x.shape
    A, B = np.zeros((n, d)), np.zeros((n, d))
    for mu in range(n):
        for i in range(d):
            A[mu, i] = -likelihood.dwfout(y=y[mu], w=omega[mu, i], V=V[mu, i]) * x_squared[mu, i]
            B[mu, i] = likelihood.fout(y=y[mu], w=omega[mu, i], V=V[mu, i]) * x[mu, i]

    return A, B
                
def compute_message_variable_to_node_from_A_B(A, B):
    """
    Compute the "mean" and "variance" parameters R, Sigma  of the message m_{i \to \mu} from
    the parameters A and B of the message m_{\mu \to i}
    """
    # A_{\mu \to i} has shape (n, d)
    n, d = A.shape
    Sigma, R = np.zeros((d, n)), np.zeros((d, n))

    for mu in range(n):
        for i in range(d):
            Sigma[i, mu] = 1.0 / (np.sum(A[:, i]) - A[mu, i])
            R[i, mu]   = Sigma[i, mu] * (np.sum(B[:, i]) - B[mu, i])

    # Sigma = 1.0 / (np.tile(np.sum(A, axis = 0), (n, 1)) - A).T
    # R     = Sigma * (np.tile(np.sum(B, axis = 0), (n, 1)) - B).T

    return R, Sigma

def compute_mean_variance_from_A_B(A, B, prior):
    n, d = A.shape
    Sigma = 1.0 / np.sum(A, axis = 0)
    R = Sigma * np.sum(B, axis = 0)
    mean = [prior.fa(Sigma=s, R=r) for s, r in zip(Sigma, R)]
    variance = [prior.fv(Sigma=s, R=r) for s, r in zip(Sigma, R)]
    return mean, variance