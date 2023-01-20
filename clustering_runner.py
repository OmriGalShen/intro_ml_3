import numpy as np

from kmeans import kmeans
from singlelinkage import singlelinkage


def question_2_c():
    k = 10
    clusters = kmeans(X, k=10, t=10)
    calculate_clustering_error("Question 2c", "kmeans", clusters, k=k, total_size=1000)


def question_2_d():
    k = 10
    clusters = singlelinkage(X, k=k)
    calculate_clustering_error("Question 2d", "singlelinkage", clusters, k=k, total_size=1000)


def question_2_e():
    k = 6
    clusters = kmeans(X, k=k, t=10)
    calculate_clustering_error("Question 2e - kmeans", "kmeans", clusters, k=k, total_size=1000)

    clusters = singlelinkage(X, k=k)
    calculate_clustering_error("Question 2e - singlelinkage", "singlelinkage", clusters, k=k, total_size=1000)


def calculate_clustering_error(title, clustering_type, clusters, k, total_size):
    print(f"-----------------------------------")
    print(f"{title}")
    print(f"clustering type = {clustering_type}, k={k} ")
    print(f"------------------------------------")
    correct_labels_counter = sum([calc_single_cluster(clusters, cluster_index) for cluster_index in range(k)])
    clustering_error = 1 - correct_labels_counter / total_size
    print(f"---------------Results---------------")
    print(f"clustering_error={clustering_error:.3f}")
    print(f"------------------------------------")
    print()


def calc_single_cluster(clusters, cluster_index):
    indexes = np.where(clusters == cluster_index)[0]
    labels_counts, _ = np.histogram(indexes // 100, range(11))
    common_label = np.argmax(labels_counts)

    cluster_size = np.count_nonzero(clusters == cluster_index)
    labels_total_count = sum(labels_counts)
    percentage = 0
    if labels_total_count > 0:
        percentage = labels_counts[common_label] / labels_total_count

    percentage = f"{percentage:.2f}"

    print(
        f"cluster={cluster_index}, cluster_size={cluster_size}, common_label={common_label},percentage={percentage}")
    return labels_counts[common_label]


if __name__ == '__main__':
    data = np.load('mnist_all.npz')
    data_examples = []
    for i in range(10):
        curr_data = data[f"train{i}"]
        indices = np.random.choice(range(curr_data.shape[0]), 100)
        data_examples.append(curr_data[indices])
    X = np.concatenate(data_examples)

    question_2_c()
    question_2_d()
    question_2_e()
