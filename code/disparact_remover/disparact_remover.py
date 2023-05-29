from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from matplotlib import pyplot as plt

import sys
sys.path.append("../")
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score


from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets import BinaryLabelDataset
import os

diretorio_atual = os.getcwd()
#diretorio_pai = os.path.dirname(diretorio_atual)
path = os.path.abspath(diretorio_atual)

# print(path)

def run(df, df_name, privileged_groups, unprivileged_groups, size, label_name, protected, features_to_keep):
    df = df[features_to_keep]
    ad = BinaryLabelDataset(favorable_label = 1., unfavorable_label = 0., df = df, label_names = [label_name], 
                            protected_attribute_names = [protected], unprivileged_protected_attributes = 0., privileged_protected_attributes = 1.,)

    scaler = MinMaxScaler(copy=False)

    # teste e treino vão ser de 70 e 30
    split = int(size * 0.3)
    test, train = ad.split([split]) # ? -> não sei
    train.features = scaler.fit_transform(train.features)
    test.features = scaler.fit_transform(test.features)

    index = train.feature_names.index(protected)

    DIs = []
    SPD = []
    F1s = []
    
    for level in (np.linspace(0., 1., 11)):
        di = DisparateImpactRemover(repair_level=level)
        train_repd = di.fit_transform(train)
        test_repd = di.fit_transform(test)
        
        X_tr = np.delete(train_repd.features, index, axis=1)
        X_te = np.delete(test_repd.features, index, axis=1)
        y_tr = train_repd.labels.ravel()
        
        lmod = LogisticRegression(class_weight='balanced', solver='liblinear')
        lmod.fit(X_tr, y_tr)
        
        test_repd_pred = test_repd.copy()
        test_repd_pred.labels = lmod.predict(X_te)

        p = [{protected: 1}]
        u = [{protected: 0}]
        cm = BinaryLabelDatasetMetric(test_repd_pred, privileged_groups=p, unprivileged_groups=u)
        y_true = test_repd.labels.ravel()
        y_pred = test_repd_pred.labels.ravel()
        DIs.append(cm.disparate_impact())
        SPD.append(cm.statistical_parity_difference())
        F1s.append(f1_score(y_true, y_pred))

    i = 0
    file_name = path + "//results//results_DI_remover//" + df_name +"_results_DI_remover.csv"
    with open(file_name, 'a') as files:
        for level in (np.linspace(0., 1., 11)):
            print("Nível de Reparo: ", level, file = files)
            print("F1-Score: ", F1s[i],  file = files)
            print("Statistical Parity: ", SPD[i],  file = files)
            i+=1