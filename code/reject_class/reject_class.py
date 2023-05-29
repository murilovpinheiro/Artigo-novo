# Load all necessary packages
import sys
sys.path.append("../")
import numpy as np
import pandas as pd
from warnings import warn

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric, BinaryLabelDatasetMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector
from aif360.algorithms.postprocessing.reject_option_classification\
        import RejectOptionClassification

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from IPython.display import Markdown, display
import matplotlib.pyplot as plt
import os

diretorio_atual = os.getcwd()
#diretorio_pai = os.path.dirname(diretorio_atual)
path = os.path.abspath(diretorio_atual)

def compute_metrics(dataset_true, dataset_pred,
                    unprivileged_groups, privileged_groups, file_name,
                    disp=True):
    """Compute the key metrics"""
    classified_metric_pred = ClassificationMetric(dataset_true,
                                                  dataset_pred,
                                                  unprivileged_groups=unprivileged_groups,
                                                  privileged_groups=privileged_groups)
    metrics = {}

    # Calculate F1-score
    y_true = dataset_true.labels
    y_pred = dataset_pred.labels
    f1 = f1_score(y_true, y_pred)
    metrics["F1-score"] = f1

    # Calculate Statistical Parity Difference
    spd = classified_metric_pred.statistical_parity_difference()
    metrics["Statistical parity difference"] = spd
    
    if disp:
        with open(file_name, 'a') as files:
            for k, v in metrics.items():
                print("%s = %.8f" % (k, v), file = files)

    return metrics

def run(dataset_orig, df_name, privileged_groups, unprivileged_groups):
    file_name = path + "//results//results_reject_class//"+df_name+"_results_roc.csv"

    # Metric used (should be one of allowed_metrics)
    metric_name = "Statistical parity difference"

    # Upper and lower bound on the fairness metric used
    metric_ub = 0.05
    metric_lb = -0.05
            
    #random seed for calibrated equal odds prediction
    np.random.seed(1)

    # Verify metric name
    allowed_metrics = ["Statistical parity difference",
                    "Average odds difference",
                    "Equal opportunity difference"]
    if metric_name not in allowed_metrics:
        raise ValueError("Metric name should be one of allowed metrics")

    # Get the dataset and split into train and test
    dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.7], shuffle=True)
    dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)

    # Logistic regression classifier and predictions
    scale_orig = StandardScaler()
    X_train = scale_orig.fit_transform(dataset_orig_train.features)
    y_train = dataset_orig_train.labels.ravel()

    lmod = LogisticRegression()
    lmod.fit(X_train, y_train)
    y_train_pred = lmod.predict(X_train)

    # positive class index
    pos_ind = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]

    dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
    dataset_orig_train_pred.labels = y_train_pred

    dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
    X_valid = scale_orig.transform(dataset_orig_valid_pred.features)
    y_valid = dataset_orig_valid_pred.labels
    dataset_orig_valid_pred.scores = lmod.predict_proba(X_valid)[:,pos_ind].reshape(-1,1)

    dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
    X_test = scale_orig.transform(dataset_orig_test_pred.features)
    y_test = dataset_orig_test_pred.labels
    dataset_orig_test_pred.scores = lmod.predict_proba(X_test)[:,pos_ind].reshape(-1,1)

    num_thresh = 100
    ba_arr = np.zeros(num_thresh)
    class_thresh_arr = np.linspace(0.01, 0.99, num_thresh)
    for idx, class_thresh in enumerate(class_thresh_arr):
        
        fav_inds = dataset_orig_valid_pred.scores > class_thresh
        dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
        dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label
        
        classified_metric_orig_valid = ClassificationMetric(dataset_orig_valid,
                                                dataset_orig_valid_pred, 
                                                unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups)
        
        ba_arr[idx] = 0.5*(classified_metric_orig_valid.true_positive_rate()\
                        +classified_metric_orig_valid.true_negative_rate())

    best_ind = np.where(ba_arr == np.max(ba_arr))[0][0]
    best_class_thresh = class_thresh_arr[best_ind]

    ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups, 
                                    privileged_groups=privileged_groups, 
                                    low_class_thresh=0.01, high_class_thresh=0.99,
                                    num_class_thresh=100, num_ROC_margin=50,
                                    metric_name=metric_name,
                                    metric_ub=metric_ub, metric_lb=metric_lb)

    ROC = ROC.fit(dataset_orig_valid, dataset_orig_valid_pred)

    # Metrics for the test set
    fav_inds = dataset_orig_valid_pred.scores > best_class_thresh
    dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
    dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label

    # with open(file_name, 'a') as files:
    #     print('Valid No Change', file = files)

    metric_valid_bef = compute_metrics(dataset_orig_valid, dataset_orig_valid_pred, 
                    unprivileged_groups, privileged_groups, file_name) # RL normal

    # Transform the validation set
    dataset_transf_valid_pred = ROC.predict(dataset_orig_valid_pred)
    # with open(file_name, 'a') as files:
    #     print('Valid with Change', file = files)
    metric_valid_aft = compute_metrics(dataset_orig_valid, dataset_transf_valid_pred, 
                    unprivileged_groups, privileged_groups, file_name) # RL com ROC

    # Metrics for the test set
    fav_inds = dataset_orig_test_pred.scores > best_class_thresh
    dataset_orig_test_pred.labels[fav_inds] = dataset_orig_test_pred.favorable_label
    dataset_orig_test_pred.labels[~fav_inds] = dataset_orig_test_pred.unfavorable_label
    with open(file_name, 'a') as files:
        print('Test No Change', file = files)
    metric_test_bef = compute_metrics(dataset_orig_test, dataset_orig_test_pred, 
                    unprivileged_groups, privileged_groups, file_name) # RL teste

    dataset_transf_test_pred = ROC.predict(dataset_orig_test_pred)
    with open(file_name, 'a') as files:
        print('Test with Change', file = files)
    metric_test_aft = compute_metrics(dataset_orig_test, dataset_transf_test_pred, 
                    unprivileged_groups, privileged_groups, file_name) # RL teste com ROC