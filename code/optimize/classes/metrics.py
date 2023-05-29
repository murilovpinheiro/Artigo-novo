from sklearn.metrics import confusion_matrix
import numpy as np
"""
This module contains functions for computing different loss and constraint functions
"""

def evaluate_metric(metric_name, C):
    """
    Generic function to evaluate a metric

    Attributes:
        metric_name (string): Name of metric
        C (array-like, dtype=float, shape=(n,n)): Confusion matrix

    Returns:
        metric (float): Metric value
    """
    if metric_name not in globals():
        raise KeyError('Metric name not found')
    metric_fun = globals()[metric_name]
    return metric_fun(C)

def fmeasure(C):
    """
    Attributes:
        C (array-like, dtype=float, shape=(n,n)): Confusion matrix

    Returns:
        loss (float): F-measure loss
    """
    # no secret just calculating the 1 - f1 measure from a confusion matrix
    if C[0, 1] + C[1, 1] > 0:
       prec = C[1, 1] * 1.0 / (C[0, 1] + C[1, 1])
    else:
       prec = 1.0
    rec = C[1, 1] * 1.0 / (C[1, 0] + C[1, 1])
    if prec + rec == 0:
       return 0.0
    else:
       return 1.0 - 2 * prec * rec / (prec + rec)
    
def dp(CC):
    """
    Attributes:
        C (array-like, dtype=float, shape=(n,n)): Confusion matrix

    Returns:
        cons (float): Demographic parity constraint function value
    """

    M = CC.shape[0] # number of prottected attr.
    C_mean = np.zeros((2, 2)) # value of the mean between all positions for each prottected attr. conf. matrix 
    dparity = np.zeros((M, 1)) # array of all demographic disparity

    for j in range(M):
       C_mean += CC[j, :, :].reshape((2, 2)) * 1.0 / M 
       # calc of C_mean

    for j in range(M):
       dparity[j] = CC[j, 0, 1] + CC[j, 1, 1] - C_mean[0, 1] - C_mean[1, 1]
       # calc of dp
    
    return np.abs(dparity).max() # return the max absolute value