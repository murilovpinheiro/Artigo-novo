import sys
sys.path.insert(0, './code/disparact_remover')
sys.path.insert(0, './code/reject_class')
sys.path.insert(0, './code/reweighing')
sys.path.insert(0, './code/prejudice_remover')

import numpy as np
from tqdm import tqdm

import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric

import disparact_remover as dr
import reject_class as rc
import reweighing as rw
import prejudice_remover as pr

## import dataset
print("Digite o Dataset que deseja: ")
df_name = input() # "adult", "german", "compas"


if df_name == "adult":
#   dataset_orig = AdultDataset()
    df = pd.read_csv('.//data//adult.csv')

    protected = "gender"
    label_name = "income"
    
    privileged_groups = [{'gender': 1}]
    unprivileged_groups = [{'gender': 0}]
    
    features_to_keep=['income', 'gender', 'age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    
elif df_name == "german":
#   dataset_orig = GermanDataset()
    df = pd.read_csv('.//data//german.csv')

    protected = "sex"
    label_name = "credit"

    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]
    
    features_to_keep=['credit',"sex","month","credit_amount", "investment_as_income_percentage", "age"]
    
elif df_name == "compas":
#   dataset_orig = CompasDataset()
    df = pd.read_csv('.//data//compas.csv')

    label_name = "Two_yr_Recidivism"
    protected = "Race"

    privileged_groups = [{'Race': 1}]
    unprivileged_groups = [{'Race': 0}]

    features_to_keep=['Two_yr_Recidivism',"Race","Number_of_Priors","score_factor"]

print("COMEÇOU: ")

for i in tqdm([0, 1, 2, 3, 4]):
    df = df.sample(frac=1)

    dataset_orig = BinaryLabelDataset(favorable_label = 1., unfavorable_label = 0., df = df, label_names = [label_name],
                                      protected_attribute_names= [protected], privileged_protected_attributes = privileged_groups[0][protected], 
                                      unprivileged_protected_attributes = unprivileged_groups[0][protected])
    
    dr.run(df, df_name, privileged_groups, unprivileged_groups, df.shape[0], label_name, protected, features_to_keep)

    rc.run(dataset_orig, df_name, privileged_groups, unprivileged_groups)

    rw.run(dataset_orig, df_name, privileged_groups, unprivileged_groups)

    pr.run(dataset_orig, df_name, privileged_groups, unprivileged_groups)

print("TERMINOU.")