import numpy as np
import scipy.special as special
from scipy.integrate import quad
import scipy.stats as stats

import gcmpyo3

# defined as "scale" such that logit(x) \simeq probit(scale * x)
LOGIT_PROBIT_SCALING = 0.5875651988237005

def sigmoid(x):
    return 1.0 / (1. + np.exp(-x))

def sigmoid_inv(p):
    return np.log(p / (1.0 - p))

def sigmoid_gaussian_integral(mean, variance):
    """
    Integrates the sigmoid likelihood vs a Gaussian of mean "mean" and variance "variance"
    uses the approximation sigmoid(x) = probit(LOGIT_PROBIT_SCALING * x)
    """
    return sigmoid( mean / np.sqrt(1.0 + (LOGIT_PROBIT_SCALING**2 * variance)))

def probit(x):
    return 1.0 - special.erfc(-x / np.sqrt(2.)) / 2.

# 

def generalisation_error_probit_teacher(rho, m, q, sigma, error_tolerance = None):
    return 1. / np.pi * np.arccos(m / (np.sqrt(q * (rho + sigma**2))))

def generalisation_loss_probit_teacher(rho, m, q, student_variance, teacher_variance, error_tolerance = float('inf')):
    """
        It's still going to be with a logistic student
    """
    bound = 10.0
    teacher    = gcmpyo3.Probit(noise_variance = 0.0)

    if student_variance != 0.0:
        # NOTE : Make sure that the Z0 is normalized ! 
        # LogisticDataModel because we average over logistic activation
        student = gcmpyo3.Logit(noise_variance = 0.0)
        student_Z0 = lambda y, omega : - np.log( student.call_z0(y, omega, student_variance))
    if student_variance == 0.0:
        student_Z0 = lambda y, omega : np.log(1 + np.exp(- y * omega))
    
    loss  = 0.0
    error = 0.0

    for y in [-1.0, 1.0]:
        tmp = quad(lambda xi : teacher.call_z0(y, m / np.sqrt(q) * xi, rho - m**2 / q + teacher_variance) * student_Z0(y, np.sqrt(q) * xi) * stats.norm.pdf(xi, loc = 0.0, scale = 1.0), 
                               -bound, bound)
        loss  += tmp[0]
        error += tmp[1]

    if error > error_tolerance:
        print(f'Warning : error {error} is bigger than tolerance {error_tolerance}')
    
    return loss