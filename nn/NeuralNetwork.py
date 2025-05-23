import numpy as np
from typing import List

class NeuralNetwork:
    def __init__(self, input_layer_size: int):
        '''
        Creates new NeuralNetwork that can take `input_layer_size` sized vector as its input.
        Call `add_hidden_layer` and finaly `add_output_layer` to make the NeuralNetwork usable
        '''
        self.__shape = [input_layer_size]
        self.__weights: List[np.ndarray] = []
        self.__biases: List[np.ndarray] = []
        self.__activation_functions: List[np.ufunc] = []

        self.__output_layer_added = False


    def add_hidden_layer(self, hidden_layer_size: int, activation_function: np.ufunc):
        pass

    def add_output_layer(self, output_layer_size: int, activation_function: np.ufunc):
        self.__output_layer_added = False
        pass

    def feed_forward(self, input: np.ndarray) -> np.ndarray:
        pass

    def save_weights_and_biases(self, path: str):
        pass

    def load_weights_and_biases(self, path: str):
        pass