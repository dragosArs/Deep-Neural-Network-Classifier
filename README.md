## Project: Deep Neural Network Multi-Class Classifier

This classifier is a tool for training a network and saving it for later use or making predictions using a pre-existing network.
The program can train multiple networks for different use cases(different datasets with a variable number of attributes and labels). The only requirement is that the features should be float values and the labels should be integers.
The algorithm is able to search different hyperparameter configurations that provide a network with best accuracy over the validation set. The number of hidden layers is randomly chosen between 1 and 3, while the number of neurons in a hidden layer is randomly chosen from a range chosen by the user.

## How to use it for training
1. Place a features CSV file and a targets CSV file in the data folder
2. Edit the TRAIN section of the configuration file with the filename of the features file, the filename of the targets file and a desired name for the network that will be trained and then saved in the networks folder
3. Run the Main.py program and choose "train" when prompted
4. If everything provided was correct the network will take from a couple of minutes(for medium sized datasets) to a couple of hours(for big datasets) to train and when finished, you will get a message specifying the accuracy of the trained model. Your network has completed its training and is available in the networks folder in serialized form and ready for future use.

## How to use it for predicting
1. Place a features CSV file in the data folder
2. Edit the PREDICT section of the configuration file with the filename of the features file, filename of a valid network(one that has been trained on the same type of data) from the networks folder and the filename of the output file that will contain the predictions
3. Run the Main.py program and choose "predict" when prompted
4. If everything provided was correct the network will almost instantaneously create or edit a file in the data folder with the desired predictions

### Editing (hyper)parameters in the config file
    number_of_labels = integer representing the number of different classes an object can be assigned to
    input_size = integer value representing the number of attributes of an object
    min_layer_size = integer representing the lower bound used for fine tuning the network -> minimum number of neurons in a hidden layer
    max_layer_size = integer representing the upper bound used for fine tuning the network -> maximum number of neurons in a hidden layer
    convergence_error = float used as a stopping criterion for training; right now it is not used
    max_iter = integer representing the maximum number of epochs in the training stage; there is no stopping criterion, so this number is always reached
    batch_size = integer representing the size of the mini-batches(mini-batch gradient descent is used)
    step_size = float representing how fast the weights should be updated
    hyperparameter_search_iterations = number of searches performed for best configuration of network; should be at least one

## Authors
* Dragos-Cristian Arsene (dragosarsene7@gmail.com)

