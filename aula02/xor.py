#%% BIBLIOTECAS

from sklearn.neural_network import MLPClassifier

# Cargas de dados da Tabela XOR
X = [[0,0],[0,1],[1,0],[1,1]]
y = [0,1,1,0]

#%% cONFIGURAÇÃO Rede Neural

mlp = MLPClassifier(verbose=True,
                    max_iter=2000,
                    tol=1e-3,
                    activation='relu')

#%% Treinamento de Rede
mlp.fit(X,y) # executa treinamento - ver console

#%% teste
for caso in X:
    print('caso : ', caso, ' previsto: ', mlp.predict([caso]))

'''
    print( mlp.predict ( [ 0 , 0] ) )
    print( mlp.predict ( [ 0 , 1] ) )
    print( mlp.predict ( [ 1 , 0] ) )
    print( mlp.predict ( [ 1 , 1] ) )
'''

#%% Alguns parametros da rede
print("classes = ",mlp.classes_) # lista de classes
print('Erro = ',mlp.loss_) # fator de perda (erro)
print('Amostrars visitadas = ',mlp.t_) # número de amostras de treinamento
print('Atributos de entrada = ', mlp.n_features_in_) # número de atributos
print('N Ciclos = ', mlp.n_iter_) #numero de iterações no treinamento
print('N de camadas = ',mlp.n_layers_) #numero de layers
print('Tamanhos das camadas ocultas: ', mlp.hidden_layer_sizes)# tamanho da camada oculta
print('N de neurons saida = ',mlp.n_outputs_) # numero de neurons de s
print('F de ativação = ', mlp.out_activation_)  # função de ativação util
