import numpy as np
# site: keras.io, para aprender mais sobre outras funções de ativação


def stepFunction(soma):  # transfer function
    if soma >= 1:
        return 1
    return 0


def sigmoidFunction(soma):
    return 1 / (1 + np.exp(-soma))


def tanhFunction(soma):
    return (np.exp(soma) - np.exp(-soma)) / (np.exp(soma) + np.exp(-soma))


def reluFunction(soma):
    if soma >= 0:
        return soma
    return 0


def linearFunction(soma):
    return soma


def softmaxFunction(x):  # x = [7.0, 2.0, 1.3], quanto maior o valor, maior a probabilidade
    ex = np.exp(x)
    return ex / ex.sum()
