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
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score


from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.datasets import AdultDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets import BinaryLabelDataset

protected = 'sex'
privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]
df = pd.read_csv('.//data//german.csv')
df = df[['credit',"sex","month","credit_amount", "investment_as_income_percentage",
         "age"]]

ad = BinaryLabelDataset(favorable_label = 1., unfavorable_label = 0., df = df, label_names = ['credit'], protected_attribute_names = ['sex'], unprivileged_protected_attributes = 0., privileged_protected_attributes = 1.)

'''
protected = 'sex'
ad = AdultDataset(protected_attribute_names=[protected],
    privileged_classes=[['Male']], categorical_features=[],
    features_to_keep=['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']) # OKEY
    # just numerical'''

scaler = MinMaxScaler(copy=False)

test, train = ad.split([300]) # ? -> não sei
train.features = scaler.fit_transform(train.features)
test.features = scaler.fit_transform(test.features)

index = train.feature_names.index(protected)
print(index)

DIs = []
SPD = []
F1s = []
for level in tqdm(np.linspace(0., 1., 11)):
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
file_name = ".//results_DI_remover//german_results_DI_remover.csv"
with open(file_name, 'a') as files:
    for level in (np.linspace(0., 1., 11)):
        print("Nível de Reparo: ", level, file = files)
        print("Statistical Parity: ", SPD[i],  file = files)
        print("F1-Score: ", F1s[i],  file = files)
        i+=1