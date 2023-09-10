import pandas as pd
import keras
from sklearn.model_selection import train_test_split  # pip install scikit-learn
from keras.models import Sequential  # Classe para a criação da rede neural
from keras.layers import Dense  # Vamos usar camadas densas na rede neural
from sklearn.metrics import confusion_matrix, accuracy_score

previsores = pd.read_csv('entradas_breast.csv')  # Atributos previsores
classe = pd.read_csv('saidas_breast.csv')  # Atributo classe (meta)

"""
- Essa é a forma correta de trabalhar com o Keras. Ou seja, em uma variável ficam os atributos previsores e em outra as
classes (onde queremos fazer a previsão).

- sklearn faz uma divisão automática no banco de dados entre treinamento e teste. Essa biblioteca é muito usada em
 machine learn. Vamos fazer uma combinação entre keras e sklearn.

- Dense indica que cada um dos neurônios será ligado com cada neurônio da camada subsequente. Também chamada de Full
Connection.
"""

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = \
    train_test_split(previsores, classe, test_size=0.25)
# - test_size = 0.25, indica que vamos usar 25% dos registros para teste e 75% para treinamento.
# - Veja que na saída, temos apenas duas classes, isso indica que é um problema de classificação binária (simples).


# ----------------- Criação da rede neural:
classificador = Sequential()  # classificador é o nome da rede neural

"---------------------------Primeira camada oculta + camada de entrada"
classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
# Instanciando um objeto da classe dense (full connection)
# units: quantidade de neurônios da primeira camada oculta.
# units = (n° entradas + n° na camada de sáida)/ = (30+1)/12=15.5=16
# activation: 'relu', é recomendável começar por ela. fornece melhores resultados melhores para deep learn
# kernel_initializer: 'random_uniform', indica como vamos fazer a inicialização dos pesos. Apenas p/ a 1° camada oculta
# input_dim: indica quantos elementos existem nas camadas de entrada (são os 30 elementos previsores)

"""
Importante notar que começamos pela primeira camada oculta porque o parâmetro input_dim cria os neurônios para
a camada de entrada.
Para saber mais detalhes sobre os parâmetros: keras documentation.
"""


# ----------------- Nova camada oculta (nem sempre melhora os resultados)
classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))
# Observe que o único detalhe é que tiramos o parâmetro input_dim=30, já que os neurônios de entrada já foram criados.
# Também usamos 16 neurônios nessa segunda camada oculta (no geral é assim), mas precismos avaliar a melhor configuração


# ----------------- Criando a camada de saída
classificador.add(Dense(units=1, activation='sigmoid'))  # Apenas 1 neurônio na camada de saída. Rede neural criada
# activation='sigmoid', já que a classificação é binária e precisamos retornar valores entre 0 e 1 (probabildade)


# ----------------- Compilar a rede neural
# Também podemos configurar o otimizador ADAM
otimizador = keras.optimizers.Adam(learning_rate=0.001, weight_decay=0.0001, clipvalue=0.5)  # Instanciando o ADAM
# learning_rate: taxa de aprendizado, é o tamanho do passo
# weight_decay: taxa decaimento do learning rate a cada atualização de pesos. Acelera o processo de descida do gradiente
# clipvalue: evita o efeito ping-pong próximo do mínimo, faz o congelamento de um valor quanto atingir determinado range

# Configuração e execução da rede neural
classificador.compile(optimizer=otimizador, loss='binary_crossentropy', metrics=['binary_accuracy'])
# optimizer: realiza o ajuste dos pesos, como a descida do gradiente por exemplo. O ADAM é uma otimização da descida
# do gradiente estocástico. No geral, é o que fornece melhores resultados
# loss: nossa função de perda. Onde vamos fazer o cálculo do erro. Função para quando temos apenas duas classes
# (classificação binária)
# metrics=['binary_accuracy']: registros classificados certos / registro classficados errados. acurácia


# ----------------- Treinamento da rede neural
classificador.fit(previsores_treinamento, classe_treinamento, batch_size=10, epochs=100)  # Apenas a precisão do train.
# A precisão do treinamento não é a real
# fit: encaixar previsores treinamento com classe treinamento
# batch_size: calcula o erro para dez registro antes de ajustar os pesos. Faz bastante diferença no resultado final
# epochs: é número de vezes que queremos fazer o ajuste dos pesos (o treinamento)


# ----------------- Visualizando pesos que a rede neural conseguiu aprender
peso0 = classificador.layers[0].get_weights()  # A visualização dos pesos vem depois do método fit
# layers[0]: pega a primeira camada com o método get_weights(). São os pesos entre camada de entrada e oculta
# Os pesos são definidos a cada execução, o que pode fazer com se obtenha valores diferentes

# print(peso0)  [[30, 16],[16,]]
# [[30, 16],[16,]] 30 entradas ligando a primeira camada oculta e um bias ligando outros 16 neurônios
# print(len(peso0)) tamanho 2 devido a unidade de bias. Para disativar: Parâmetro use_bias=False em Dense

pesos1 = classificador.layers[1].get_weights()
print(pesos1)  # [[16, 16],[16,]]
# [[16, 16],[16,]] 16 neurônios da primeira camada ligando a outros 16 nerônios da segunda camada oculta, mais uma
# unidade de bias como a última cada oculta

pesos2 = classificador.layers[1].get_weights()
print(pesos2)
# [[16, 1],[1,]]  16 neurônios da segunda camada oculta ligando com a saída e uma bias ligando com o neurônio da saída


# ----------------- Previsão real usando a base de dado de teste (forma correta)
previsoes = classificador.predict(previsores_teste)  # previsores_teste são os 146 registros
# Ou seja, vamos passar cada um dos 143 registros para a rede neural, para que ela possa fazer todos os cálculos
# dos pesos, multiplicação, somatório, relu, sigmoid e retornar um valor de probablildade

previsoes = (previsoes > 0.5)  # True para probabilidades maiores que 0.5 (para visualizar melhor)

# Medindo o acerto. Vamos fazer um comparativo entre os 2 vetores. Avaliando na base de dados de teste (forma correta)
precisao = accuracy_score(classe_teste, previsoes)  # classe_teste (0 e 1), privisoes (True ou False)
print(precisao)  # Valor da precisão na base de dados de teste (forma correta)
print(' ')

# Criando uma matriz de confusão
matriz = confusion_matrix(classe_teste, previsoes)
print(matriz)   # Indica quais classes temos mais acerto. y: classe, x: previsão

# Até aqui (de previsoes = classificador.predict(previsores_teste) até print(matriz), fizemos avaliação do algortimo na
# base de dados de teste de forma manual usando o sklearn. Mas podemos usar o keras, abaixo
# ----------------- Fazendo os mesmos cálculos usando o Keras
resultado = classificador.evaluate(previsores_teste, classe_teste)
print(resultado)  # retorna o valor da função de erro, e o valor da precisão
# Nota, observe que o valor da precisão é o mesmo do resultado
