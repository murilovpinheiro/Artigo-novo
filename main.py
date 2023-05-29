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
    protected = "gender"
    privileged_groups = [{'gender': 1}]
    unprivileged_groups = [{'gender': 0}]
    df = pd.read_csv('.//data//adult.csv')
    label_name = "income"
    features_to_keep=['income', 'gender', 'age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    dataset_orig = BinaryLabelDataset(favorable_label = 1., unfavorable_label = 0., df = df, label_names = ['income'],
                                       protected_attribute_names = ['gender'], unprivileged_protected_attributes = 0., privileged_protected_attributes = 1.)
    
elif df_name == "german":
#   dataset_orig = GermanDataset()
    protected = "sex"
    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]
    df = pd.read_csv('.//data//german.csv')
    label_name = "credit"
    features_to_keep=['credit',"sex","month","credit_amount", "investment_as_income_percentage", "age"]
    dataset_orig = BinaryLabelDataset(favorable_label = 1., unfavorable_label = 0., df = df, label_names = ['credit'],
                                       protected_attribute_names = ['sex'], unprivileged_protected_attributes = 0., privileged_protected_attributes = 1.)
    
elif df_name == "compas":
#   dataset_orig = CompasDataset()
    protected = "Race"
    privileged_groups = [{'Race': 1}]
    unprivileged_groups = [{'Race': 0}]
    df = pd.read_csv('.//data//compas.csv')
    label_name = "Two_yr_Recidivism"
    features_to_keep=['Two_yr_Recidivism',"Race","Number_of_Priors","score_factor"]
    dataset_orig = BinaryLabelDataset(favorable_label = 1., unfavorable_label = 0., df = df, label_names = ["Two_yr_Recidivism"],
                                           protected_attribute_names= ["Race"], unprivileged_protected_attributes = 0., privileged_protected_attributes = 1.)

print("COMEÇOU: ")
dr.run(df, df_name, privileged_groups, unprivileged_groups, df.shape[0], label_name, protected, features_to_keep)

rc.run(dataset_orig, df_name, privileged_groups, unprivileged_groups)

rw.run(dataset_orig, df_name, privileged_groups, unprivileged_groups)

pr.run(dataset_orig, df_name, privileged_groups, unprivileged_groups)

print("TERMINOU.")