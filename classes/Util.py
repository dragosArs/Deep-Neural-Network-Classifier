import numpy as np


# Reads the features from a text file.
def read_features(filename: str) -> list:
    input = np.loadtxt(filename, dtype='f', delimiter=',')
    return input


# Reads the labels from a text file.
def read_labels(filename: str) -> list:
    input = np.loadtxt(filename, dtype='i', delimiter=',')
    return input

# Splits the dataset in two.
def split_dataset(data: list, input_size: int, percentage: float):
    split_point = int(percentage * len(data))
    return data[:split_point], data[split_point:]

#Shuffles two data structures in the same way
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
