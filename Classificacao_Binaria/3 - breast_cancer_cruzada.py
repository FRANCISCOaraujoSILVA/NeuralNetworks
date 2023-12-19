import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier  # pip install scikeras
from sklearn.model_selection import cross_val_score  # Função que faz a divisão da base de dados (Cross Validation)


"""
- VALIDAÇÃO CRUZADA. Uma técnica mais eficiente para fazer avaliações de algoritmos de aprendizagem de máquinas. Em
trabalhos científicos e de pesquisa essa técnica é a mais utilizada. Com essa prática, todos os modelos são usados para 
fazer o treinamento e teste alternativamente

- O percentual de acerto é a média da precisão obtida em cada K (cv).

- K = 10, muito aceito na comunidade científica.

- underfitting: Seria como tentar eliminar um tiranossaura-rex (problema complexo) com uma raquete
    - Terá resultados ruins na base de treinamento.
- overfitting: Seria com tentar matar um mosquito (problema simples) com uma bazuca (muitos recursos)
    - Terá resultados bons na base de dados de treinamento
    - Terá resultados ruins na base de dados de teste
    - Muito específico
    - Memorização
    - Erros na variação de novas instâncias
    
Dropout: para corrigir ou atenuar o problema de overfitting. Irá zerá alguns valores da entrada (na camada de entrada ou 
camada oculta), para que esses valores (aleatórios) não tenham influencia no resultado final.

"""

entradas_breast = "\\Users\\franc\\PycharmProjects\\NeuralNetworks\\Classificacao_Binaria\\entradas_breast.csv"
saidas_breast = "\\Users\\franc\\PycharmProjects\\NeuralNetworks\\Classificacao_Binaria\\saidas_breast.csv"
previsores = pd.read_csv(entradas_breast)
classe = pd.read_csv(saidas_breast)

# Para a validação cruzada precisamos criar uma função


def criarRede():
    classificador = Sequential()
    """
    Nota: dropout: sempre após a criação de uma camada. Serve para zerar alguns neurônios de forma que estes não tenham 
    nenhuma influência no resultado final. É recomendado ter um dropout entre 20 e 30% 
    """
    classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
    classificador.add(Dropout(0.2))  # 20% dos neurônios da camada de entrada serão zerados

    classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))
    classificador.add(Dropout(0.2))  # 20% dos neurônios da camada oculta serão zerados

    classificador.add(Dense(units=1, activation='sigmoid'))
    otimizador = keras.optimizers.Adam(learning_rate=0.001, weight_decay=0.0001, clipvalue=0.5)
    classificador.compile(optimizer=otimizador, loss='binary_crossentropy', metrics=['binary_accuracy'])
    return classificador


# Variável classificador
classificador = KerasClassifier(build_fn=criarRede, epochs=100, batch_size=10)
# build_fn: função que faz a criação da rede neural

# Realiza os testes. Teremos 10 resultados
resultados = cross_val_score(estimator=classificador, X=previsores, y=classe, cv=10, scoring='accuracy')
# X: indica quais são os atributos previsores
# y: recebe a classe
# cv: faz a divisão da base de dados para fazer a validação cruzada. É o K, nesse caso teremos 10 fatias, sendo que uma
# delas será usada para teste enquanto o restante (90%) será usado para treinamento, vamos alternar entre todas as
# fatias para fazer a validação cruzada.
# scoring: a forma como queremos retornar o resultado

# Daqui pra cima ele já realiza a o treinamento com a validação cruzada

media = resultados.mean()  # média, para saber o percentual de acerto da base de dados
print(media)

desvio = resultados.std()  # Desvio padrão
print(desvio)
# desvio padrão, quanto maior o desvio, maior a chance de ter overfitting na base de dados.
# overfitting: Quando o algoritmo (rede neural) se adapta demais a base de dados de treinamento. Isso implica que
# quando vamos passar dados novos para essa rede, ela não vai nos fornecer bons resultados
