import sys
sys.path.insert(0, '../')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Datasets
# Fairness metrics
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric# Explainers
# Scalers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Bias mitigation techniques
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import PrejudiceRemover
from aif360.datasets import BinaryLabelDataset

from collections import defaultdict
import os

diretorio_atual = os.getcwd()
#diretorio_pai = os.path.dirname(diretorio_atual)
path = os.path.abspath(diretorio_atual)

def test(dataset, model, thresh_arr, privileged_groups, unprivileged_groups ):
    # aif360 inprocessing algorithm
    y_val_pred_prob = model.predict(dataset).scores
    pos_ind = 0
    
    metric_arrs = defaultdict(list)
    for thresh in thresh_arr:
        y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_val_pred
        metric = ClassificationMetric(
                dataset, dataset_pred,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
        
        metric_arrs['stat_par_diff'].append(metric.statistical_parity_difference())

        y_true = dataset.labels.ravel()
        y_pred = dataset_pred.labels.ravel()
        f1 = f1_score(y_true, y_pred)
        metric_arrs['f1_score'].append(f1)
    
    return metric_arrs

# privileged_groups = [{'gender': 1}]
# unprivileged_groups = [{'gender': 0}]
# df = pd.read_csv('.//data//adult.csv')
# dataset_orig = BinaryLabelDataset(favorable_label = 1., unfavorable_label = 0., df = df, label_names = ['income'], protected_attribute_names = ['gender'], unprivileged_protected_attributes = 0., privileged_protected_attributes = 1.)
def run(dataset_orig, df_name, privileged_groups, unprivileged_groups):
    file_name = path + "//results//results_prejudice_rem//"+ df_name +"_resultsPR.csv"

    sens_ind = 0

    sens_attr = dataset_orig.protected_attribute_names[sens_ind]

    # Divisão em treinamento, teste e validação
    dataset_train, dataset_orig_vt = dataset_orig.split([0.7], shuffle=True)
    dataset_val, dataset_test = dataset_orig_vt.split([0.5], shuffle=True)

    # Pré-processamento dos dados
    scaler = StandardScaler()

    dataset_train.features = scaler.fit_transform(dataset_train.features)
    dataset_test.features = scaler.transform(dataset_test.features)
    dataset_val.features = scaler.transform(dataset_val.features)
    etalist = [0.1, 1, 25]
    # Modelo PrejudiceRemover
    for eta in etalist:
        model = PrejudiceRemover(sensitive_attr=sens_attr, eta=eta)
        pr_orig = model.fit(dataset_train)

        # Validação e seleção de limiar
        thresh_arr = np.linspace(0.01, 0.50, 50)
        val_metrics = test(dataset=dataset_val, model=pr_orig, thresh_arr=thresh_arr, 
                           privileged_groups = privileged_groups, unprivileged_groups= unprivileged_groups)
        best_ind = np.argmax(val_metrics['f1_score'])

        # Teste final com o melhor limiar selecionado
        metrics = test(dataset=dataset_test, model=pr_orig, thresh_arr=[thresh_arr[best_ind]], 
                       privileged_groups = privileged_groups, unprivileged_groups= unprivileged_groups)

        # Salvar métricas em arquivo
        with open(file_name, 'a') as file:
            print("eta: ", eta, file = file)
            print("F1-Score:", metrics['f1_score'][0], file=file)
            print("Statistical Parity:", metrics['stat_par_diff'][0], file=file)
