import numpy as np
from typing import List
from copy import deepcopy


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


    @property
    def shape(self):
        """
        returns tuple of numbers of neurons in each layer
        """
        return tuple(self.__shape)
    

    def copy(self):
        return deepcopy(self)


    def add_hidden_layer(self, hidden_layer_size: int, activation_function: np.ufunc):
        """
        adds hidden layer,
        raises ValueError if output layer was already added
        """
        if self.__output_layer_added:
            raise ValueError("Can't add hidden layer, since output layer has already been added")
        self.__shape += [hidden_layer_size]
        self.__activation_functions += [activation_function]
        self.__biases += [np.zeros(hidden_layer_size)]

        weights_matrix_shape = self.shape[-1], self.shape[-2]
        self.__weights += [np.ones(weights_matrix_shape)]

    def add_output_layer(self, output_layer_size: int, activation_function: np.ufunc):
        """
        adds output layer,
        raises ValueError if output layer was already added
        """
        if self.__output_layer_added:
            raise ValueError("Can't add output layer, since output layer is already present")
        
        self.__shape += [output_layer_size]
        self.__activation_functions += [activation_function]
        self.__biases += [np.zeros(output_layer_size)]

        weights_matrix_shape = self.shape[-1], self.shape[-2]
        self.__weights += [np.ones(weights_matrix_shape)]

        self.__output_layer_added = True
        

    def feed_forward(self, input: np.ndarray) -> np.ndarray:
        """
        feeds forwad given input vector thru the neural network
        """
        if not self.__output_layer_added:
            raise ValueError("Output layer not present")
        if input.shape != (self.shape[0],):
            raise ValueError(f"input vector has to have the same size as input_layer, expected {(self.shape[0],)}; got {input.shape}")
        
        y = np.array(input)
        for w, af, b in zip(self.__weights, self.__activation_functions, self.__biases):
            y = af(w @ y - b)
        
        return y

    def save_weights_and_biases(self, path: str):
        """
        saves NeuralNetwork's weights and biases to a file, by using numpy 'npz' format
        """
        np.savez(path, *self.__weights, *self.__biases)
    

    def load_weights_and_biases(self, path: str):
        """
        loads weights and biases from a npz file,
        The NeuralNetwork needs to already have its architecture defined e.i. `add_hidden_layer` and `add_output_layer` have been called already
        """
        if not self.__output_layer_added:
            raise ValueError("Output layer not present")
        # TODO: Add file/network compatibilty checks
        npzfile = np.load(path)
        N = len(npzfile.files)

        for i, arr in enumerate(npzfile.files):
            if i < N // 2:
                self.__weights[i] = npzfile[arr]
            else:
                self.__biases[i - N // 2] = npzfile[arr]