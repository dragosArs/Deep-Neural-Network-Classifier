import numpy as np
from Layer import Layer
from Hyperparameters import Hyperparameters
from numba import jit, cuda

class Network:
    def __init__(self, hyper_parameters, input_size, number_of_labels):
        self.hyper_parameters = hyper_parameters
        self.layers = []
        self.hyper_parameters.layer_breadth.insert(0, input_size)
        self.hyper_parameters.layer_breadth.append(number_of_labels)
        for i in range(1, len(hyper_parameters.layer_breadth)):
            self.layers.append(Layer(hyper_parameters.layer_breadth[i - 1], hyper_parameters.layer_breadth[i]))

    def train(self, trainX, trainY):
        for i in range(0, self.hyper_parameters.max_iter):
            j = 0
            while j < len(trainX):
                end =  min(j + self.hyper_parameters.batch_size, len(trainX))
                self.apply_error_correction(trainX[j:end], trainY[j:end])
                j = end + 1
            ##to add convergence error stop
    
    ##This method is used for the TRAIN action
    def predict(self, X, Y):
        output = self.forward_propagation(X)
        predicted = np.argmax(output)
        if predicted == Y - 1:
            return True
        else:
            return False

    ##This method is used for the PREDICT action
    def make_prediction_for_dataset(self, input):
        predictions = np.zeros(len(input), dtype=int)
        for i, object in enumerate(input):
            predictions[i] = np.argmax(self.forward_propagation(object)) + 1
        return predictions

    ##calculate average of loss over a mini-batch and update weights using this average          
    def apply_error_correction(self, batchX, batchY):
        real_output = np.zeros((self.hyper_parameters.number_of_labels))
        real_output[batchY[0] - 1] = 1
        W_error_sum, b_error_sum = self.calculate_gradient(batchX[0], real_output)
        #This loop iterates through each sample and then avearges errors
        for sample in range(1, len(batchX)):
            real_output = np.zeros((self.hyper_parameters.number_of_labels))
            real_output[batchY[sample] - 1] = 1
            W_errors, b_errors = self.calculate_gradient(batchX[sample], real_output)
            for j in range(0, len(self.layers)):
                W_error_sum[j] += W_errors[j]
                b_error_sum[j] += b_errors[j]
        for i, layer in enumerate(self.layers):
            layer.W = layer.W - self.hyper_parameters.step_size * W_error_sum[i] / len(batchX)
            layer.b = layer.b - self.hyper_parameters.step_size * b_error_sum[i] / len(batchX)

    ##Returns a vector with the gradient for each layer
    def calculate_gradient(self, input, output):
        self.forward_propagation(input)
        self.backward_propagation(input, output)
        W_errors = [] 
        b_errors = []
        for layer in self.layers:
            W_errors.append(layer.W_error)
            b_errors.append(layer.b_error)
        return list(W_errors), list(b_errors)

    ##Feed an input to the network and return the values of the output layer
    def forward_propagation(self, input):
        output = np.ndarray.copy(input)
        for layer in self.layers:
            output = layer.calculate_a(output)
        return output

    ##Perform back-propagation using desired output and calculate errors for every layer linearly using a recurrence equation
    def backward_propagation(self, input, output):
        n = len(self.layers)
        last_layer = self.layers[n - 1]
        last_layer.calculate_delta_last(output)
        last_layer.calculate_error(self.layers[n - 2].a)
        for i in range(n - 2, -1, -1):
            cur_layer = self.layers[i]
            next_layer = self.layers[i + 1]
            cur_layer.calculate_delta_normal(next_layer.W, next_layer.delta)
            if i == 0:
                cur_layer.calculate_error(input)
            else:
                cur_layer.calculate_error(self.layers[i - 1].a)
        





    

        

