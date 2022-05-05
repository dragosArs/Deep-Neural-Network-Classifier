import pickle

import numpy as np

from configparser import ConfigParser

from Hyperparameters import Hyperparameters
from Network import Network
from Util import (read_features, read_labels, split_dataset,
                  unison_shuffled_copies)

##I will use the following pseudocode to split my dataset in training, validation and test sets
#   Split dataset into training set(80%) and test set(20%)
#   FOR i = 1 TO ... 
#       Choose valid hyperparameters in the given range for network i
#       Initialize network with random weights and biases
#       FOR j = 1 TO 5 
#           Shuffle training set
#           Split training set into validation set(first 5%) and training set(last 95%)
#           Train network using mini-batches
#           Calculate accuracy over validation set j
#       Calculate average accuracy. If better than what has been seen update best network



config_object = ConfigParser()
config_object.read("./config.ini")
train_config = config_object["TRAIN"]
predict_config = config_object["PREDICT"]
common_config = config_object["COMMON"]

input_size = int(common_config["input_size"])
number_of_labels = int(common_config["number_of_labels"])
min_layer_size = int(common_config["min_layer_size"])
max_layer_size = int(common_config["max_layer_size"])
convergence_error = float(common_config["convergence_error"])
max_iter = int(common_config["max_iter"])
batch_size = int(common_config["batch_size"])
step_size = float(common_config["step_size"])
hyperparameter_search_iterations = int(common_config["hyperparameter_search_iterations"])
network_to_write_filename = train_config["network_filename"]
features_filename = train_config["features_filename"]
targets_filename = train_config["targets_filename"]
predictions_input_filename = predict_config["input_filename"]
predictions_output_filename = predict_config["output_filename"]
network_to_read_filename = predict_config["network_filename"]


action = input("Please make your selection(train/predict):")
if action == 'train':
    #dataX = read_features("./data/features.txt")
    dataX = read_features(features_filename)
    #dataY = read_labels("./data/targets.txt")
    dataY = read_labels(targets_filename)
    trainX, testX = split_dataset(dataX, input_size, 0.8)
    trainY, testY = split_dataset(dataY, input_size, 0.8)
    max_accuracy = 0
    rng = np.random.default_rng()
    for i in range(0, hyperparameter_search_iterations):
        number_of_hidden_layers = rng.integers(1, 3)
        layer_breadth = []
        for k in range(0, number_of_hidden_layers):
            layer_breadth.append(rng.integers(min_layer_size, max_layer_size))
        hyperparameters = Hyperparameters(step_size, layer_breadth, max_iter, convergence_error, batch_size, number_of_labels) 
        network = Network(hyperparameters, input_size, number_of_labels)
        avg_accuracy = 0
        for j in range(0, 5):
            trainX, trainY = unison_shuffled_copies(trainX, trainY)
            actual_trainX, validationX = split_dataset(trainX, input_size, 0.95)
            actual_trainY, validationY = split_dataset(trainY, input_size, 0.95) 
            network.train(actual_trainX, actual_trainY) 
            sum = 0
            ##calculate accuracy of model on validation set
            for val_input, val_output in zip(validationX, validationY):
                if network.predict(val_input, val_output):
                    sum += 1
            avg_accuracy += sum / len(validationX)
        avg_accuracy /= 5 
        if avg_accuracy > max_accuracy:
            max_accuracy = avg_accuracy
            best_network = network

    test_accuracy = 0
    for input_test, output_test in zip(testX, testY):
        if best_network.predict(input_test, output_test):
            test_accuracy += 1
    test_accuracy /= len(testX)
    with open(network_to_write_filename, "wb") as outfile:
        pickle.dump(best_network, outfile)
    print("A neural network was created, with a test accuracy of ", test_accuracy)

elif action == "predict":
    with open(network_to_read_filename, "rb") as infile:
        network = pickle.load(infile)
    input_to_predict = read_features(predictions_input_filename)
    np.savetxt(predictions_output_filename, network.make_prediction_for_dataset(input_to_predict), fmt="%i")

else:
    print("Action is not recognised")


           


        

