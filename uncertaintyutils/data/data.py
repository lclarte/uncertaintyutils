'''
data.py
'''

from typing import Tuple

import numpy as np
import scipy.linalg as linalg
import scipy.stats as stats
import sys
from time import time

from .. import utility

### KEEPING THE TWO FUNCTIONS BELOW FOR BACKWARDS COMP.

"""
Structure 
"""

def iid_teacher(d : float) -> np.ndarray:
    return np.random.normal(0., 1., size=(d, ))

def iid_input(n : int, d : int) -> np.ndarray:
    return np.random.normal(0., 1., size=(n, d)) / np.sqrt(d)

# ==========

def sample_teacher(covariance : np.ndarray) -> np.ndarray:
    """
    Sample teacher from Gaussian distribution with covariance
    """
    d = len(covariance)
    mean = np.zeros(d)
    return np.random.multivariate_normal(mean, covariance)

def sample_gaussian_input(n : int, covariance : np.ndarray) -> np.ndarray:
    d = len(covariance)
    mean = np.zeros(d)
    return stats.multivariate_normal.rvs(mean=mean, cov=covariance, size=n)

## Matched setting

def sample_labels_from_local_fields(local_fields, likelihood = "logit", noise_std = 0.0):
    """
    returns the labels between -1 and 1
    """
    if likelihood == "probit":
        if noise_std > 0.0:
            likelihood_function = lambda x : utility.probit(x / noise_std)
        elif noise_std == 0.0:
            likelihood_function = lambda x : np.heaviside(x, 0.5)
    elif likelihood == "logit":
            likelihood_function = lambda x : utility.sigmoid_gaussian_integral(mean = x, variance = noise_std**2)
    else:
        raise NotImplementedError
    y = np.random.binomial(n=1, p = likelihood_function(local_fields))
    return 2 * y - 1

def sample_data_from_teacher(n, teacher, likelihood = "logit", noise_std = 0.0, input_covariance = None):
    """
    Sample data where the teacher and student model match. We rescale the input by sqrt(dimension)
    because the teacher's components are O(1)
    The labels are either -1, 1
    arguments:
        - likelihood is either logit or probit
    """
    d = len(teacher)
    if input_covariance is None:
        input_covariance = np.eye(d)
    if len(input_covariance) != d:
        raise Exception
    
    x = sample_gaussian_input(n, input_covariance)

    y = sample_labels_from_local_fields(x @ teacher, likelihood = likelihood, noise_std = noise_std)
    return x, y

def sample_data(n, d, likelihood = "logit", noise_std = 0.0, teacher_covariance = None, input_covariance = None):
    if teacher_covariance is None:
        teacher_covariance = np.eye(d)
    if d != len(teacher_covariance):
        raise Exception
    teacher = sample_teacher(teacher_covariance)
    x, y = sample_data_from_teacher(n, teacher, likelihood=likelihood, noise_std=noise_std, input_covariance = input_covariance)
    return teacher, x, y

## Gcm setting

def sample_gcm_data_from_teacher(n, teacher, teacher_teacher_covariance, teacher_student_covariance, student_student_covariance, likelihood = "logit", noise_std = 0.0):
    """
    Returns the teachers'data, student's data and the labels
    arguments:
        - noise_std : noise of the likelihood in the teacher space
        - teacher_teacher_covariance : covariance of the teacher's data
        - teacher_student_covariance : covariance of the teacher's data and the student's data
        - student_student_covariance : covariance of the student's data
    """
    # stack the covariance matrices to get the covariance of the joint distribution
    covariance = np.block([[teacher_teacher_covariance, teacher_student_covariance], [teacher_student_covariance.T, student_student_covariance]])
    # sample from the joint distribution
    joint_data = sample_gaussian_input(n, covariance)
    # split the data
    teacher_input, student_input = joint_data[:, :len(teacher)], joint_data[:, len(teacher):]
    y = sample_labels_from_local_fields(teacher_input @ teacher, likelihood = likelihood, noise_std = noise_std)
    return teacher_input, student_input, y

def sample_gcm_data(n, d,  teacher_teacher_covariance, teacher_student_covariance, student_student_covariance, likelihood = "logit", noise_std = 0.0, teacher_covariance = None):
    teacher_covariance = teacher_covariance or np.eye(d)
    teacher = sample_teacher(teacher_covariance)
    teacher_input, student_input, y = sample_gcm_data_from_teacher(n, teacher, teacher_teacher_covariance, teacher_student_covariance, student_student_covariance, likelihood = likelihood, noise_std = noise_std)
    return teacher, teacher_input, student_input, y