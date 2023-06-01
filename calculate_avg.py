import pandas as pd

# Função para normalizar uma métrica para uma escala de 0 a 1
def normalize_metric(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

# Função para calcular a média ponderada considerando igualmente as métricas normalizadas
def weighted_average(metric_A, metric_B, min_A, max_A, min_B, max_B):
    normalized_A = normalize_metric(metric_A, min_A, max_A)
    normalized_B = normalize_metric(1 - metric_B, min_B, max_B)
    return 0.5 * normalized_A + 0.5 * normalized_B
# Ler o dataset
dataset = pd.read_csv('final_results.csv')

# Extrair as colunas das métricas A e B do dataset

metric_A_values = dataset['F1-Score']
metric_B_values = dataset['Statistical Parity'].apply(abs)
min_A = metric_A_values.min()
max_A = metric_A_values.max()
min_B = metric_B_values.min()
max_B = metric_B_values.max()

# Calcular a média ponderada para cada linha do dataset
weighted_avg_values = []
for metric_A, metric_B in zip(metric_A_values, metric_B_values):
    weighted_avg = weighted_average(metric_A, metric_B, min_A, max_A, min_B, max_B)
    weighted_avg_values.append(weighted_avg)

# Adicionar os valores da média ponderada como uma nova coluna no dataset
dataset['weighted_avg'] = weighted_avg_values
dataset.to_csv("final_results.csv", index = False)

# Imprimir o resultado
print(dataset)
