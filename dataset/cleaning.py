import numpy as np
from scipy.ndimage.interpolation import shift
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml

def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]


def create_shifted_copies(X, y):
    for i, image in enumerate(X):
        shifted1 = shift(image, 1, cval=0)
        shifted2 = shift(image, -1, cval=0)
        X = np.append(X, [shifted1, shifted2], axis=0)
        y = np.append(y, y[i])
    return X, y