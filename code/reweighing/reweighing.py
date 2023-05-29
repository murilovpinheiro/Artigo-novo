# Load all necessary packages
import sys
sys.path.append("../")
import numpy as np
from tqdm import tqdm

import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing.reweighing import Reweighing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from sklearn.metrics import f1_score

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
    file_name = path + "//results//results_reweighing//"+ df_name + "_resultsRW.csv"
    all_metrics =  ["Statistical parity difference",
                    "Average odds difference",
                    "Equal opportunity difference"]

    # Get the dataset and split into train and test
    dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.7], shuffle=True)
    dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)

    # Metric for the original dataset
    metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                                                unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups)
    print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())
    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
    RW.fit(dataset_orig_train)
    dataset_transf_train = RW.transform(dataset_orig_train)
    metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train, 
                                            unprivileged_groups=unprivileged_groups,
                                            privileged_groups=privileged_groups)

    print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf_train.mean_difference())

    # Logistic regression classifier and predictions
    scale_orig = StandardScaler()
    X_train = scale_orig.fit_transform(dataset_orig_train.features)
    y_train = dataset_orig_train.labels.ravel()
    w_train = dataset_orig_train.instance_weights.ravel()

    lmod = LogisticRegression()
    lmod.fit(X_train, y_train, 
            sample_weight=dataset_orig_train.instance_weights)
    y_train_pred = lmod.predict(X_train)

    # positive class index
    pos_ind = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]

    dataset_orig_train_pred = dataset_orig_train.copy()
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


    print("Best balanced accuracy (no reweighing) = %.4f" % np.max(ba_arr))
    print("Optimal classification threshold (no reweighing) = %.4f" % best_class_thresh)

    bal_acc_arr_orig = []
    disp_imp_arr_orig = []
    avg_odds_diff_arr_orig = []

    print("Classification threshold used = %.4f" % best_class_thresh)
    with open(file_name, 'a') as files:
        print('No Reweighing', file = files)
    for thresh in tqdm(class_thresh_arr):
        
        if thresh == best_class_thresh:
            disp = True
        else:
            disp = False
        
        fav_inds = dataset_orig_test_pred.scores > thresh
        dataset_orig_test_pred.labels[fav_inds] = dataset_orig_test_pred.favorable_label
        dataset_orig_test_pred.labels[~fav_inds] = dataset_orig_test_pred.unfavorable_label
        
        metric_test_bef = compute_metrics(dataset_orig_test, dataset_orig_test_pred, 
                                        unprivileged_groups, privileged_groups, file_name,
                                        disp = disp)
        
    scale_transf = StandardScaler()
    X_train = scale_transf.fit_transform(dataset_transf_train.features)
    y_train = dataset_transf_train.labels.ravel()

    lmod = LogisticRegression()
    lmod.fit(X_train, y_train,
            sample_weight=dataset_transf_train.instance_weights)
    y_train_pred = lmod.predict(X_train)

    dataset_transf_test_pred = dataset_orig_test.copy(deepcopy=True)
    X_test = scale_transf.fit_transform(dataset_transf_test_pred.features)
    y_test = dataset_transf_test_pred.labels
    dataset_transf_test_pred.scores = lmod.predict_proba(X_test)[:,pos_ind].reshape(-1,1)

    bal_acc_arr_transf = []
    disp_imp_arr_transf = []
    avg_odds_diff_arr_transf = []

    print("Classification threshold used = %.4f" % best_class_thresh)
    with open(file_name, 'a') as files:
        print('With Reweighing', file = files)
    for thresh in tqdm(class_thresh_arr):
        
        if thresh == best_class_thresh:
            disp = True
        else:
            disp = False
        
        fav_inds = dataset_transf_test_pred.scores > thresh
        dataset_transf_test_pred.labels[fav_inds] = dataset_transf_test_pred.favorable_label
        dataset_transf_test_pred.labels[~fav_inds] = dataset_transf_test_pred.unfavorable_label
        
        metric_test_aft = compute_metrics(dataset_orig_test, dataset_transf_test_pred, 
                                        unprivileged_groups, privileged_groups, file_name,
                                        disp = disp)