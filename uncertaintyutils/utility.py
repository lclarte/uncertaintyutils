import numpy as np
import scipy.special as special

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