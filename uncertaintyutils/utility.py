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
    return special.erfc(-x / np.sqrt(2.)) / 2.

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

## 

def generalisation_error_logit_teacher(rho, m, q, sigma = 0.0, error_tolerance = float('inf')):
    bound = 20.0
    # NOTE : The noise variance sigma**2 of the teacher is put in the conditional variance for numerical precision concerns
    model = gcmpyo3.Logit ( noise_variance = 0.0 )

    def integrand_plus(xi):
        # Probability that we make a mistake when the xabel y = -1 (and not 1 !) because it's the proba that 
        # the teacher has the label 1 given that the student has a -1 i.e a local field xi < 0
        # so if xi > 0, there is not error
        if xi > 0.0:
            return 0.0
        # Caution ! Needs to be normalized !!! 
        # If xi < 0 i.e predicted label is -1, we return the proba tht the true label is 1 
        return model.call_z0(1.0, m / np.sqrt(q) * xi, rho - m * m / q + sigma**2)

    def integrand_minus(xi):
        if xi < 0.0:
            return 0.0
        # Caution ! Needs to be normalized !!! 
        return model.call_z0(-1.0, m / np.sqrt(q) * xi, rho - m * m / q + sigma**2)
        # return model.call_z0(-1, m / np.sqrt(q) * xi, rho - m**2 / q)
    
    I_plus, I_plus_error  = quad(lambda xi : integrand_plus(xi) * stats.norm.pdf(xi, loc = 0.0, scale = 1.0), -bound, bound)
    I_minus, I_minus_error = quad(lambda xi : integrand_minus(xi) * stats.norm.pdf(xi, loc = 0.0, scale = 1.0), -bound, bound)

    error = I_minus_error + I_plus_error

    if error > error_tolerance:
        print(f'Warning : error {error} is bigger than tolerance {error_tolerance}')

    return I_minus + I_plus

def generalisation_loss_logit_teacher(rho, m, q, student_variance, teacher_variance, error_tolerance = float('inf')):
    bound = 20.0
    teacher    = gcmpyo3.Logit(noise_variance = 0.0)

    if student_variance != 0.0:
        # NOTE : Make sure that the Z0 is normalized ! 
        # LogisticDataModel because we average over logistic activation
        student = gcmpyo3.Logit(noise_variance = 0.0)
        student_Z0 = lambda y, omega : - np.log( student.call_z0(y, omega, student_variance))
    if student_variance == 0.0:
        student_Z0 = lambda y, omega : np.log(1 + np.exp(- y * omega))
    
    loss = 0.0
    error= 0.0
    for y in [-1.0, 1.0]:
        tmp = quad(lambda xi : teacher.call_z0(y, m / np.sqrt(q) * xi, rho - m**2 / q + teacher_variance) * student_Z0(y, np.sqrt(q) * xi) * stats.norm.pdf(xi, loc = 0.0, scale = 1.0), 
                               -bound, bound)
        loss  += tmp[0]
        error += tmp[1]

    if error > error_tolerance:
        print(f'Warning : error {error} is bigger than tolerance {error_tolerance}')

    return loss