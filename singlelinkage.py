import numpy as np
from scipy.spatial import distance


def singlelinkage(X, k):
    """
    :param X: numpy array of size (m, d) containing the test samples
    :param k: the number of clusters
    :return: a column vector of length m, where C(i) ∈ {1, . . . , k} is the identity of the cluster in which x_i has been assigned.
    """
    m, d = X.shape
    distance_mat = get_initial_distance_mat(X)
    # Initialize the clusters as singletons
    clusters = np.array(range(m))

    for _ in range(m - k):
        i, j = get_closest_pair_of_clusters(distance_mat, m)
        update_distance_mat(i, j, distance_mat, clusters)

    clusters = clusters.reshape(m, 1)
    return clusters


def get_initial_distance_mat(X):
    distance_mat = distance.squareform(distance.pdist(X))
    np.fill_diagonal(distance_mat, np.inf)
    return distance_mat


def get_closest_pair_of_clusters(distance_mat, m):
    return np.unravel_index(np.argmin(distance_mat), (m, m))


def update_distance_mat(i, j, distance_mat, clusters):
    for cluster in range(len(clusters)):
        if cluster == i or cluster == j:
            continue
        min_distance = min(distance_mat[i][cluster], distance_mat[j][cluster])
        distance_mat[i][cluster] = min_distance
        distance_mat[cluster][i] = min_distance

    distance_mat[j, :] = np.inf
    distance_mat[:, j] = np.inf

    clusters[clusters == j] = i


def simple_test():
    # load sample data (this is just an example code, don't forget the other part)
    data = np.load('mnist_all.npz')
    data0 = data['train0']
    data1 = data['train1']
    X = np.concatenate((data0[np.random.choice(data0.shape[0], 30)], data1[np.random.choice(data1.shape[0], 30)]))
    m, d = X.shape

    # run single-linkage
    c = singlelinkage(X, k=10)

    assert isinstance(c, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert c.shape[0] == m and c.shape[1] == 1, f"The shape of the output should be ({m}, 1)"


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()

    # here you may add any code that uses the above functions to solve question 2
