import pandas as pd
import numpy as np

adult_rem = pd.read_csv("./results/results_DI_remover/adult_results_DI_remover.csv", delimiter = ":", header=None)
german_rem = pd.read_csv("./results/results_DI_remover/german_results_DI_remover.csv", delimiter = ":", header=None)
compas_rem = pd.read_csv("./results/results_DI_remover/compas_results_DI_remover.csv", delimiter = ":", header=None)

adult_op = pd.read_csv("./results/results_optimizer/adult_resultsOP.csv", delimiter = "/", header=None) #
german_op = pd.read_csv("./results/results_optimizer/german_resultsOP.csv", delimiter = "/", header=None) #
compas_op = pd.read_csv("./results/results_optimizer/compas_resultsOP.csv", delimiter = "/", header=None) # 

#OP BEM FORMATADO

adult_pr = pd.read_csv("./results/results_prejudice_rem/adult_resultsPR.csv", delimiter = ":", header=None)
german_pr = pd.read_csv("./results/results_prejudice_rem/german_resultsPR.csv", delimiter = ":", header=None)
compas_pr = pd.read_csv("./results/results_prejudice_rem/compas_resultsPR.csv", delimiter = ":", header=None)

adult_roc = pd.read_csv("./results/results_reject_class/adult_results_roc.csv", delimiter = ":", header=None)
german_roc = pd.read_csv("./results/results_reject_class/german_results_roc.csv", delimiter = ":", header=None)
compas_roc = pd.read_csv("./results/results_reject_class/compas_results_roc.csv", delimiter = ":", header=None)

adult_rw= pd.read_csv("./results/results_reweighing/adult_resultsRW.csv", delimiter = ":", header=None)
german_rw = pd.read_csv("./results/results_reweighing/german_resultsRW.csv", delimiter = ":", header=None)
compas_rw = pd.read_csv("./results/results_reweighing/compas_resultsRW.csv", delimiter = ":", header=None)

def formatar_rem(dataset, dataset_name):
    repair_levels = dataset.loc[dataset[0] == "Nivel de Reparo"].reset_index()
    fscores = dataset.loc[dataset[0] == "F1-Score"].reset_index()
    spd = dataset.loc[dataset[0] == "Statistical Parity"].reset_index()
    new_dataset = pd.DataFrame(columns = [ "F1-Score", "Statistical Parity", "Another_Info"])
    new_dataset["F1-Score"] = fscores[1]
    new_dataset["Statistical Parity"] = spd[1]
    new_dataset["Another_Info"] = repair_levels[1]
    new_dataset["Method"] = "DisparactImpactRemover"
    new_dataset["Data_Name"] = dataset_name

    return new_dataset

def formatar_op(dataset, dataset_name):
    dataset = dataset.drop(3, axis = 1)
    new_dataset = pd.DataFrame(columns = [ "F1-Score", "Statistical Parity", "Another_Info"])
    new_dataset["F1-Score"] = 1 - (dataset[1]).astype(float)
    new_dataset["Statistical Parity"] = dataset[2]
    new_dataset["Another_Info"] = dataset[0]
    new_dataset["Method"] = "Optimizer"
    new_dataset["Data_Name"] = dataset_name

    return new_dataset

def formatar_pr(dataset, dataset_name):
    eta = dataset.loc[dataset[0] == "eta"].reset_index()
    fscores = dataset.loc[dataset[0] == "F1-Score"].reset_index()
    spd = dataset.loc[dataset[0] == "Statistical Parity"].reset_index()
    new_dataset = pd.DataFrame(columns = [ "F1-Score", "Statistical Parity", "Another_Info"])
    new_dataset["F1-Score"] = fscores[1]
    new_dataset["Statistical Parity"] = spd[1]
    new_dataset["Another_Info"] = eta[1]
    new_dataset["Method"] = "PrejudiceRemover"
    new_dataset["Data_Name"] = dataset_name


    return new_dataset

def formatar_roc(dataset, dataset_name):
    test_type = dataset.loc[(dataset[0] == "Test No Change ") | (dataset[0] == "Test with Change ")].reset_index()
    test_type[0] = test_type[0].str.strip()
    fscores = dataset.loc[dataset[0] == "F1-score "].reset_index()
    spd = dataset.loc[dataset[0] == "Statistical parity difference "].reset_index()
    new_dataset = pd.DataFrame(columns = [ "F1-Score", "Statistical Parity", "Another_Info"])

    new_dataset["F1-Score"] = fscores[1]
    new_dataset["Statistical Parity"] = spd[1]
    new_dataset["Another_Info"] = test_type[0]
    new_dataset["Method"] = "RejectClass"
    new_dataset["Data_Name"] = dataset_name


    return new_dataset

def formatar_rw(dataset, dataset_name):
    test_type = dataset.loc[(dataset[0] == "With Reweighing ") | (dataset[0] == "No Reweighing ")].reset_index()
    test_type[0] = test_type[0].str.strip()
    fscores = dataset.loc[dataset[0] == "F1-score "].reset_index()
    spd = dataset.loc[dataset[0] == "Statistical parity difference "].reset_index()
    new_dataset = pd.DataFrame(columns = [ "F1-Score", "Statistical Parity", "Another_Info"])

    new_dataset["F1-Score"] = fscores[1].astype(float)
    new_dataset["Statistical Parity"] = spd[1].astype(float)
    new_dataset["Another_Info"] = test_type[0]
    new_dataset["Method"] = "Reweighing"
    new_dataset["Data_Name"] = dataset_name

    return new_dataset


adult_rem = formatar_rem(adult_rem, "Adult") # OK
german_rem = formatar_rem(german_rem, "German")
compas_rem = formatar_rem(compas_rem, "Compas")

adult_op = formatar_op(adult_op, "Adult") # OK
german_op = formatar_op(german_op, "German")
compas_op = formatar_op(compas_op, "Compas")

adult_pr = formatar_pr(adult_pr, "Adult") # OK
german_pr = formatar_pr(german_pr, "German")
compas_pr = formatar_pr(compas_pr, "Compas")

adult_roc = formatar_roc(adult_roc, "Adult") # OK
german_roc = formatar_roc(german_roc, "German")
compas_roc = formatar_roc(compas_roc, "Compas")

adult_rw = formatar_rw(adult_rw, "Adult")
german_rw = formatar_rw(german_rw, "German")
compas_rw = formatar_rw(compas_rw, "Compas")

final_dataset = pd.concat([adult_rem, german_rem, compas_rem,
                           adult_op, german_op, compas_op,
                           adult_pr, german_pr, compas_pr,
                           adult_roc, german_roc, compas_roc,
                           adult_rw, german_rw, compas_rw], axis = 0)

final_dataset.to_csv("final_results.csv", index = False)