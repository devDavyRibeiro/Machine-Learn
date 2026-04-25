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
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split #para não viesar os resultados, o ideal é separar os dados em treino e teste
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


#%% CARGA DOS DADOS

df = pd.read_csv('student-mat.csv', sep=';')

print("\nColunas:\n", df.columns)
input('Aperte uma tecla para continuar:')


#%% DEFINIÇÃO DE X e Y

numerics = df[['age','Medu','Fedu','traveltime','studytime','failures','famrel','freetime','goout','Dalc','Walc','health','absences']]
textos = df[['sex','address','famsize','Pstatus','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic']]
y = df['G3']

print('\nClasses:\n', y.value_counts())
input('Aperte uma tecla para continuar:')

#%% TRATAMENTO (CATEGÓRICOS → NUMÉRICO)

encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(textos)
X = np.hstack((numerics.values, X_encoded))

num_cols = numerics.columns
cat_cols = encoder.get_feature_names_out(textos.columns)
all_cols = list(num_cols) + list(cat_cols)
X_df = pd.DataFrame(X, columns=all_cols)

print("\nColunas após tratamento:\n", X)
input('Aperte uma tecla para continuar:')
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

#%% GRAFICO
import seaborn as sns

sns.pairplot(X_df)

#%% CONFIGURAÇÃO DA REDE

mlp = MLPRegressor(
    verbose=True,
    hidden_layer_sizes=(100),
    max_iter=1000,
    tol=1e-6,
)

#%% TREINAMENTO

mlp.fit(X_train, y_train)


#%% AVALIAÇÃO

y_pred = mlp.predict(X_test)
from sklearn.metrics import mean_squared_error, mean_absolute_error

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'\nResultados da Avaliação:')
print(f'MSE (Mean Squared Error): {mse:.2f}')
print(f'MAE (Mean Absolute Error): {mae:.2f}')

#%% REGRESSÃO LINEAR
# Iniciando o modelo
from sklearn.linear_model import LinearRegression
lm = LinearRegression()

#%% 
# TREINAR O MODELO
lm.fit(X_train,y_train)
print('Coeficientes: \n', lm.coef_)

#%% 
# Previsão de dados de teste
prev = lm.predict( X_test)

#%% 
# GRÁFICO DOS DADOS X Y
import matplotlib.pyplot as plt
plt.scatter(y_test, prev)
plt.xlabel('Y Test')
plt.ylabel('Y previsto')

#%% 
# Avaliando o Modelo
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, prev))
print('MSE:', metrics.mean_squared_error(y_test, prev))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prev)))

#%% ALGUNS PARÂMETROS DA REDE
#
#print("\nClasses = ", mlp.classes_)     # lista de classes
#print("Erro = ", mlp.loss_)    # fator de perda (erro)
#print("Amostras visitadas = ", mlp.t_)     # número de amostras de treinamento visitadas 
#print("Atributos de entrada = ", mlp.n_features_in_)   # número de atributos de entrada (campos de X)
#print("N ciclos = ", mlp.n_iter_)      # númerode iterações no treinamento
#print("N de camadas = ", mlp.n_layers_)    # número de camadas da rede
#print("Tamanhos das camadas ocultas: ", mlp.hidden_layer_sizes)
#print("N de neurons saida = ", mlp.n_outputs_)   # número de neurons de saida
#print("F de ativação = ", mlp.out_activation_)  # função de ativação utilizada
