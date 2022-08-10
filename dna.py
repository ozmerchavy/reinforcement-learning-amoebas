import numpy as np
from neural_network import create_weights_and_biases
import random
import base64
from typing import Union


def random_mutation_for_arr(arr: np.array, max_abs_change=0.05):
    arr = arr.copy()
    for i in range(len(arr)):
        addition =  np.random.random(size=arr[i].shape) * 2 - 1
        arr[i] += addition * max_abs_change
    return arr


class DNA:
    def __init__(self, genes: Union[str, tuple]):
        if type(genes) is str:
            array = np.array  # this hack is for 'eval'
            __dict = eval(str(base64.b64decode(genes), encoding='utf-8'))
            for k, v in __dict.items():
                setattr(self, k, v)
        else:
            self.weights_and_biases, self.sens_diff, self.sens_len, self.hue, self.sight_angle = genes

     
    @staticmethod
    def new_random(neurons_per_layer=[4,2,1]):
        weights_and_biases = create_weights_and_biases(neurons_per_layer)
        sens_diff = random.random() * 1.3 + 0.9
        sens_len = random.randrange(45, 80)
        hue = random.random()
        sight_angle = random.random()
        return DNA((weights_and_biases, sens_diff, sens_len, hue, sight_angle))

    def mutation(self):
        w, b = self.weights_and_biases
        weights_and_biases = random_mutation_for_arr(w), random_mutation_for_arr(b)
        sens_diff = self.sens_diff + (random.random() - 0.5) / 5
        sens_len = self.sens_len  + random.randint(-1, 1) / 5
        hue = (self.hue + random.random() / 27) % 1
        sight_angle = self.sight_angle + (random.random() - 0.5) / 5
        return DNA((weights_and_biases, sens_diff, sens_len, hue, sight_angle))

    def __str__(self) -> str:
        return str(base64.b64encode(bytes(str(self.__dict__), encoding='utf-8')), encoding='utf-8')



