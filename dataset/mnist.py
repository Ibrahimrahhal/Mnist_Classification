import numpy as np
from scipy.ndimage.interpolation import shift
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from dataset.cleaning import sort_by_target, create_shifted_copies

def load_ready_mnist():
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    mnist.target = mnist.target.astype(np.int8)

    # Sorting the dataset in order to split it correctly to train, test subsets
    sort_by_target(mnist)
    X, y = mnist['data'], mnist['target']

    # Splitting the dataset
    X_train, X_test, y_train, y_test = X[:10], X[60000:], y[:10], y[60000:]

    # Creating two extra copies of each image, each one shifted one pixel to one direction (training set expansion) I am
    # leaving it commented because it triples the size of the dataset. And it takes much more time to process it,
    # it increases the accuracy though

    # X_train, y_train = create_shifted_copies(X_train, y_train)


    # Shuffling the dataset
    shuffle_index = np.random.permutation(10)
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

    # X_train_scale is the final fully prepared variable for the training data
    # y_train is the final fully prepared variable for the training data labels

    return X_train_scaled, y_train
