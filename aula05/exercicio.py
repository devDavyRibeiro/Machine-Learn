"""
Dataset de estudantes - Classificação

Objetivo:
Classificar alunos com base na nota final (G3)

Classes:
0 = Reprovado (0–9)
1 = Médio (10–14)
2 = Alto (15–20)
"""

#%% BIBLIOTECAS

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import pandas as pd
import numpy as np

#%% CARGA DOS DADOS

df = pd.read_csv('student-mat.csv', sep=';')

print("\nColunas:\n", df.columns)
input('Aperte uma tecla para continuar:')

#%% CRIAR CLASSES A PARTIR DO G3

def classificar_nota(nota):
    if nota <= 9:
        return 0   # Reprovado
    elif nota <= 14:
        return 1   # Médio
    else:
        return 2   # Alto

df['G3_class'] = df['G3'].apply(classificar_nota)

#%% DEFINIÇÃO DE X e Y

X = df[[
    'sex', 'age', 'address', 'Pstatus',
    'Medu', 'Fedu',
    'Mjob', 'Fjob',
    'guardian',
    'famrel',
    'studytime',
    'failures',
    'schoolsup', 'famsup',
    'activities',
    'paid',
    'internet',
    'higher',
    'romantic',
    'freetime',
    'goout',
    'Walc', 'Dalc',
    'health'
]]

y = df['G3_class']

print('\nClasses:\n', y.value_counts())
input('Aperte uma tecla para continuar:')

#%% TRATAMENTO (CATEGÓRICOS → NUMÉRICO)

X = pd.get_dummies(X)

#%% DIVISÃO TREINO / TESTE

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nFormato treino:", X_train.shape)
print("Formato teste:", X_test.shape)
input('Aperte uma tecla para continuar:')

#%% NORMALIZAÇÃO (IMPORTANTE)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%% CONFIGURAÇÃO DA REDE

mlp = MLPClassifier(
    verbose=True,
    hidden_layer_sizes=(100, 25),
    max_iter=1000,
    tol=1e-6,
    activation='relu'
)

#%% TREINAMENTO

mlp.fit(X_train, y_train)

#%% TESTE COM UM EXEMPLO

exemplo = X_test[0:1]

print('\nClasse real:', y_test.iloc[0])
print('Classe prevista:', mlp.predict(exemplo))
input('Aperte uma tecla para continuar:')

#%% AVALIAÇÃO

y_pred = mlp.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nAcurácia:", accuracy)

#%% MATRIZ DE CONFUSÃO

cm = confusion_matrix(y_test, y_pred)

print("\nMatriz de Confusão:")
print(cm)

# Gráfico
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#%% TESTE COM DIFERENTES ARQUITETURAS

for camadas in [(10,), (20,), (50,), (100,), (200,), (10,10), (20,20), (50,50)]:
    mlp = MLPClassifier(
        hidden_layer_sizes=camadas,
        max_iter=1000,
        tol=1e-6
    )
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("\nAcurácia com", camadas, ":", acc)
    

#%% PARÂMETROS

    print("\nClasses:", mlp.classes_)
    print("Loss:", mlp.loss_)
    print("Iterações:", mlp.n_iter_)
    print("Camadas:", mlp.n_layers_)
    print("Saídas:", mlp.n_outputs_)

#%% ALGUNS PARÂMETROS DA REDE

print("\nClasses = ", mlp.classes_)     # lista de classes
print("Erro = ", mlp.loss_)    # fator de perda (erro)
print("Amostras visitadas = ", mlp.t_)     # número de amostras de treinamento visitadas 
print("Atributos de entrada = ", mlp.n_features_in_)   # número de atributos de entrada (campos de X)
print("N ciclos = ", mlp.n_iter_)      # númerode iterações no treinamento
print("N de camadas = ", mlp.n_layers_)    # número de camadas da rede
print("Tamanhos das camadas ocultas: ", mlp.hidden_layer_sizes)
print("N de neurons saida = ", mlp.n_outputs_)   # número de neurons de saida
print("F de ativação = ", mlp.out_activation_)  # função de ativação utilizada

## TEORIA: 
# Verdadeiros e falsos positivos / negativos
# Acurácia, precisão, recall, score F, 