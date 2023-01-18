import numpy as np
import scipy.io as sio
from scipy.spatial import distance


def singlelinkage(X, k):
    """
    :param X: numpy array of size (m, d) containing the test samples
    :param k: the number of clusters
    :return: a column vector of length m, where C(i) ∈ {1, . . . , k} is the identity of the cluster in which x_i has been assigned.
    """
    m, d = X.shape
    clusters = [[i] for i in range(m)]  # trivial singletons clustering
    distance_mat = distance.squareform(distance.pdist(X))

    while len(clusters) > k:
        # Find the closest pair of clusters
        i, j = np.unravel_index(np.argmin(distance_mat), distance_mat.shape)

        # Merge the two closest clusters
        clusters[i] = clusters[i] + clusters[j]
        del clusters[j]

        # Update distance matrix
        distance_mat[i, j] = np.inf
        distance_mat[j, :] = np.inf
        distance_mat[:, j] = np.inf
        distance_mat[i, i] = np.inf

        for l in range(len(clusters)):
            if l != i:
                distance_mat[i, l] = min([distance_mat[x, y] for x in clusters[i] for y in clusters[l]])
                distance_mat[l, i] = distance_mat[i, l]

    # Assign each point to its final cluster
    C = np.zeros((m, 1))
    for i in range(k):
        for j in clusters[i]:
            C[j] = i

    return C


def simple_test():
    # load sample data (this is just an example code, don't forget the other part)
    data = np.load('mnist_all.npz')
    X = np.concatenate((data['train0'], data['train1']))
    m, d = X.shape

    # run single-linkage
    c = singlelinkage(X, k=10)

    assert isinstance(c, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert c.shape[0] == m and c.shape[1] == 1, f"The shape of the output should be ({m}, 1)"


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()

    # here you may add any code that uses the above functions to solve question 2
