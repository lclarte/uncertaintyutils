# temperature_scaling.py
# functions to do TS as well as expectation consistency

import numpy as np
import scipy.optimize as optimize
from .. import utility

def find_optimal_temperature_temperature_scaling(logits, labels, T_min = 0.01, T_max = 10.0):
    """
    labels must be in {-1 ,1}
    """
    def objective(T):
        probas = logits / T
        return np.mean(np.log(1.0 + np.exp(- labels * probas)))
    
    res = optimize.minimize_scalar(objective, bounds = [T_min, T_max], method = 'bounded')
    return float(res.x)

def find_optimal_temperature_expectation_consistency(logits, labels, T_min = 0.01, T_max = 10.0):
    """
    labels must be in {-1 ,1}
    """
    accuracy = np.mean(np.sign(logits) == labels)

    def objective(T):
        probas = utility.sigmoid(np.abs(logits) / T)
        return np.mean(probas) - accuracy

    res = optimize.root_scalar(objective, bracket = [T_min, T_max])
    return float(res.root)
