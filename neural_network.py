import numpy as np
import math

def pairs(seq):
    lst = []
    for i in range(len(seq) - 1):
        lst.append((seq[i], seq[i + 1]))
    return lst

def create_weights_and_biases(neurons_per_layer):
    weights_matrices = []
    biases_vectors = []
    for num_in, num_out in pairs(neurons_per_layer):
        weights_matrices.append(np.random.random(size=[num_out, num_in]))
        biases_vectors.append(np.random.random(size=[num_out]))
    return weights_matrices, biases_vectors

def forward(x, weights_matrices, biases_vectors):
    for w, b in zip(weights_matrices, biases_vectors):
        x = eitan(w @ x + b)
    return x


@np.vectorize
def relu(x):
    return max(x, 0)


@np.vectorize
def eitan(x):
    return math.atan(x) / (math.pi / 2)

