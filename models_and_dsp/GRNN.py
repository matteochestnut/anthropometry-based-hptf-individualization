import numpy as np

class GRNN:
    '''
    Simple Generalized Regression Neural Network
    '''
    def __init__(self, sigma = 1.0):
        '''
        Initialize the only parameter of the network
        
        # Arguments
        sigma (float): standard deviation for the Gaussian Kernel
        '''
        self._sigma = sigma
    
    def fit(self, x_train, y_train):
        '''
        Loading training dataset to compute the predictions
        # Arguments
        x_train (np.array): data points as a N x M matrix, N is number of observations, M is the number of features
        y_train (n.array): labels corresponding to the datapoints ad a N x D matrix, N is the number of observations
        '''
        self._x_train = x_train
        self._y_train = y_train
        
    def predict(self, x_test):
        '''
        Predicting training set labels
        Computing a weighted sum og the Gaussian Kernels between each point and the dataset using trainig labels as weights
        The weighted sum is normalized by the sum of the Gassian Kernels
        
        # Arguments
        x_test: data points as a N x M matrix, N is number of observations, M is the number of features
        '''
        prediction = []
        for i  in range(x_test.shape[0]):
            distance = self._distance(x_test[i])
            act = self._gaussianKernel(distance)
            numerator = np.matmul( self._y_train.transpose(), act )
            denominator = np.sum(act)
            #prediction.append( np.array(numerator / denominator) )
            prediction.append( np.divide(numerator, denominator) )
        return np.array(prediction)
    
    def _gaussianKernel(self, distance):
        '''
        Computing Gaussian Kernel
        
        # Arguments
        distance (np.array): array of distances between a datapoint and the points in the training set
        '''
        act = np.exp( -(distance**2) / (2 * (self._sigma)**2) )
        return act
    
    def _distance(self, dataPoint):
        '''
        Computing distances between a data point and examples in the training set
        '''
        distance = np.sqrt( np.sum( (dataPoint - self._x_train)**2 , axis = 1 ) )
        return distance