import random

import numpy as np
from scipy.spatial import distance


def kmeans(X, k, t):
    """
    :param X: numpy array of size (m, d) containing the test samples
    :param k: the number of clusters
    :param t: the number of iterations to run
    :return: a column vector of length m, where C(i) ∈ {1, . . . , k} is the identity of the cluster in which x_i has been assigned.
    """
    m, d = X.shape
    centroids_indexes = random.choices(range(m), k=k)
    centroids = X[centroids_indexes]
    C = None
    for _ in range(t):
        distances = distance.cdist(X, centroids)  # shape (m,k)
        C = np.argmin(distances, axis=1).reshape(m)  # shape (m,)
        new_centroids = [(X[C == i]).mean(axis=0) for i in range(k)]
        new_centroids = np.array(new_centroids)
        if np.array_equal(new_centroids, centroids):
            break
        centroids = new_centroids
    return C.reshape(m, 1) if C is not None else None


def simple_test():
    # load sample data (this is just an example code, don't forget the other part)
    data = np.load('mnist_all.npz')
    X = np.concatenate((data['train0'], data['train1']))
    m, d = X.shape

    # run K-means
    c = kmeans(X, k=10, t=10)

    assert isinstance(c, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert c.shape[0] == m and c.shape[1] == 1, f"The shape of the output should be ({m}, 1)"


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()

    # here you may add any code that uses the above functions to solve question 2
