import pandas as pd
import keras
from sklearn.model_selection import train_test_split  # pip install scikit-learn
from keras.models import Sequential  # classe para a criação da rede neural. Usa o modo sequencial
from keras.layers import Dense  # vamos usar camadas densas na rede neural. Todos os neurônios são ligados com todos
from sklearn.metrics import confusion_matrix, accuracy_score

entradas_breast = "\\Users\\franc\\PycharmProjects\\NeuralNetworks\\Classificacao_Binaria\\entradas_breast.csv"
saidas_breast = "\\Users\\franc\\PycharmProjects\\NeuralNetworks\\Classificacao_Binaria\\saidas_breast.csv"
previsores = pd.read_csv(entradas_breast)  # atributos previsores
classe = pd.read_csv(saidas_breast)  # classe (meta/resposta). 0 -> Benigno, 1 -> Maligno


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
# - veja que na saída, temos apenas duas classes, isso indica que é um problema de classificação binária (simples).


# ----------------- Criação da rede neural:
classificador = Sequential()  # classificador é o nome da rede neural

"---------------------------Primeira camada oculta + camada de entrada"
classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
# instanciando um objeto da classe dense (full connection)
# units: quantidade de neurônios da primeira camada oculta
# units = (n° entradas + n° na camada de sáida)/ = (30+1)/12=15.5=16
# activation: 'relu', é recomendável começar por ela. Fornece melhores resultados melhores para deep learn
# kernel_initializer: 'random_uniform', indica como vamos fazer a inicialização dos pesos.
# input_dim: indica quantos elementos existem na camada de entrada (são os 30 elementos previsores). Apenas p/ a 1°
# camada oculta

"""
Importante notar que começamos pela primeira camada oculta porque o parâmetro input_dim cria os neurônios para
a camada de entrada.
Para saber mais detalhes sobre os parâmetros: visete o site keras documentation.
"""


# ----------------- Nova camada oculta (nem sempre melhora os resultados)
classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))
# Observe que o único detalhe é que tiramos o parâmetro input_dim=30, já que os neurônios de entrada já foram criados.
# Também usamos 16 neurônios nessa segunda camada oculta (no geral é assim), mas precismos avaliar a melhor configuração


# ----------------- Criando a camada de saída
classificador.add(Dense(units=1, activation='sigmoid'))  # Apenas 1 neurônio na camada de saída. Rede neural criada
# activation='sigmoid', já que a classificação é binária e precisamos retornar valores entre 0 e 1 (probabildade)


"Até este momento, a rede neural já foi criada, na qual esta rede neural possui uma estrutura"

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
# loss: nossa função de perda. Onde vamos fazer o cálculo do erro.
# binary_crossentropy: função para quando temos apenas duas classes (classificação binária)
# metrics=['binary_accuracy']: registros classificados certos / registro classficados errados. Será nossa acurácia


# ----------------- Treinamento da rede neural
classificador.fit(previsores_treinamento, classe_treinamento, batch_size=10, epochs=100)  # Apenas a precisão do train.
# A precisão do treinamento não é a real, apenas aproximada
# fit: encaixar previsores treinamento com classe treinamento
# batch_size: calcula o erro para dez registro antes de ajustar os pesos. Faz bastante diferença no resultado final
# epochs: é número de vezes que queremos fazer o ajuste dos pesos (o treinamento)


# ----------------- Visualizando os pesos que a rede neural conseguiu aprender
pesos0 = classificador.layers[0].get_weights()  # A visualização dos pesos vem depois do método fit
# layers[0]: pega a primeira camada com o método get_weights(). São os pesos entre a camada de entrada e oculta
# Os pesos são definidos a cada execução, o que pode fazer com se obtenha valores diferentes

# print(pesos0)  # [[30, 16],[16,]] 30 neurônios camada de entrada ligando a 16 neurônios da primeira camada oculta
# 30: número de entradas
# 16: quantidades de neurônios na camada oculta
# [16,] uma unidade de bias ligando outros 16 neurônios
# print(len(peso0)) tamanho 2 devido a unidade de bias. Para disativar: Em Dense -> arâmetro use_bias=False

pesos1 = classificador.layers[1].get_weights()
# print(pesos1)  # [[16, 16],[16,]]
# [[16, 16],[16,]] 16 neurônios da primeira camada oculta ligando a outros 16 nerônios da segunda camada oculta,
# e mais uma unidade de bias ligando com a última cada oculta

pesos2 = classificador.layers[1].get_weights()
# print(pesos2)
# [[16, 1],[1,]]  16 neurônios da segunda camada oculta ligando com a a camada de saída e uma bias ligando com o
# neurônio da saída


# ----------------- Previsão real usando a base de dado de teste (forma correta)
previsoes = classificador.predict(previsores_teste)  # previsores_teste são os 146 registros
# Ou seja, vamos passar cada um dos 143 registros para a rede neural, para que ela possa fazer todos os cálculos
# dos pesos, multiplicação, somatório, relu, sigmoid e retornar um valor de probablildade

previsoes = (previsoes > 0.5)  # True para probabilidades maiores que 0.5 (para visualizar melhor)

# Medindo o acerto. Vamos fazer um comparativo entre os 2 vetores. Avaliando na base de dados de teste (forma correta)
precisao = accuracy_score(classe_teste, previsoes)  # classe_teste (0 e 1), privisoes (True ou False)
print(f'Precisão real com a base de dados de teste: {precisao}')  # Valor da precisão na base de dados de teste
# (forma correta)
print(' ')

# Criando uma matriz de confusão
matriz = confusion_matrix(classe_teste, previsoes)
print(f'Matriz de confusão:\n {matriz}')   # Indica quais classes temos mais acerto. y: classe, x: previsão

# Até aqui (de previsoes = classificador.predict(previsores_teste) até print(f'Matriz de confusão: {matriz}'),
# fizemos avaliação do algortimo na base de dados de teste de forma manual usando o sklearn. Mas podemos usar o keras,
# abaixo

# ----------------- Fazendo os mesmos cálculos usando o Keras
resultado = classificador.evaluate(previsores_teste, classe_teste)
# Esse comando submete previsores_teste para a nossa rede neural para que ela faça a previsão e a avaliação com a
# classe_teste
print(f'Valor da função erro e valor da precisão: {resultado}')
# Nota, observe que o valor da precisão aqui é o mesmo da variável precisao
