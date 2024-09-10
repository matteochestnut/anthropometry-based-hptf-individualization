from keras.layers import Layer, Dense, Activation
from keras import backend as K
from keras.initializers import RandomUniform, Constant, Initializer
from keras.models import Sequential
from keras.losses import MeanSquaredError
import numpy as np

class InitCentersRandom(Initializer):
    '''
    Initializer function to initialize centers of the RBF layer
    A random examples from the dataset passed is picked
    From: https://github.com/raaaouf/RBF_neural_network_python/blob/master/RBF_neuralNetwork%20.py
    '''
    def __init__(self, X):
        '''
        # Arguments
        X (np.array): a dataset to which pick the examples to initialized the RBF centers with
        '''
        self.X = X
    
    def __call__(self, shape, dtype = None):
        assert shape[1] == self.X.shape[1]
        idx = np.random.randint(self.X.shape[0], size = shape[0])
        return self.X[idx, :]

class RBFLayer(Layer):
    '''
    Class that inherits from keras.layers.Layer class to build a layer with
    a Radial Basis Funntion as activation function
    Code based on: https://github.com/raaaouf/RBF_neural_network_python/blob/master/RBF_neuralNetwork%20.py
    '''
    def __init__(self, units, gammas = 1.0, initializer = None, **kwargs):
        '''
        Initialize a RBF lyer
        
        # Arguments
        units (int): number of neurons in the RBF layer
        gammas (float): default values for the gammas parameter when computing the radial function
        initializer: a function to initiliaze the centers of each radial function assign to a unit
        
        By inheriting from Layer, build, call and compute_output_shape need to ber implemented
        '''
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self._init_gammas = gammas
        if not initializer:
            self.initializer = RandomUniform(0.0, 1.0)
        else:
            self.initializer = initializer
        super(RBFLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.centers = self.add_weight(
            name = 'centers',
            shape = (self.units, input_shape[1]),
            initializer = self.initializer,
            trainable = True
        )
        self.gammas = self.add_weight(
            name = 'gammas',
            shape = (self.units),
            initializer = Constant(value = self._init_gammas),
            trainable = True
        )
        
    def call(self, inputs):
        C = K.expand_dims(self.centers)
        H = K.transpose(C - K.transpose(inputs))
        act = K.exp( -self.gammas * K.sum(H**2, axis = 1) )
        return act
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)
    
    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'output_dim': self.units
        }
        base_config = super(RBFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
def my_loss(y, y_hat):
    return K.sqrt( K.sum( K.square(y - y_hat) ) / (K.int_shape(y)[0] * K.int_shape(y)[1]) )

def RBFNN(number_units, centers, output_dim, loss = 'custom', gammas = 1.0):
    '''
    '''
    model = Sequential()
    model.add(RBFLayer(
        units = number_units,
        gammas = gammas,
        initializer = InitCentersRandom( centers ),
        input_shape = (centers.shape[1],)
    ))
    model.add(Dense(output_dim))
    model.add(Activation('linear'))
    
    # compiling
    if loss == 'custom':
        model.compile(optimizer = 'adam', loss = my_loss)
    else:
        model.compile(optimizer = 'adam', loss = MeanSquaredError())
    return model