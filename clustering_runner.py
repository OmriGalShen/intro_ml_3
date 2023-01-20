import numpy as np

from kmeans import kmeans
from singlelinkage import singlelinkage

K_MEAN_SAMPLE_SIZE = 100
SINGLELINKAGE_SAMPLE_SIZE = 30


def get_sample(sample_size):
    data_examples = []
    for i in range(10):
        curr_data = data[f"train{i}"]
        indices = np.random.choice(range(len(curr_data)), sample_size)
        data_examples.append(curr_data[indices])
    return np.concatenate(data_examples)


def question_2_c():
    k = 10
    X = get_sample(K_MEAN_SAMPLE_SIZE)
    clusters = kmeans(X, k=10, t=10)
    calculate_clustering_error("Question 2c", "kmeans", clusters, k=k, total_size=1000)


def question_2_d():
    k = 10
    X = get_sample(SINGLELINKAGE_SAMPLE_SIZE)
    clusters = singlelinkage(X, k=k)
    for cluster_number, i in enumerate(np.unique(clusters)):
        clusters[clusters == i] = cluster_number
    calculate_clustering_error("Question 2d", "singlelinkage", clusters, k=k, total_size=1000)


def question_2_e():
    k = 6

    X = get_sample(K_MEAN_SAMPLE_SIZE)
    clusters = kmeans(X, k=k, t=10)
    calculate_clustering_error("Question 2e - kmeans", "kmeans", clusters, k=k, total_size=1000)

    X = get_sample(SINGLELINKAGE_SAMPLE_SIZE)
    clusters = singlelinkage(X, k=k)
    for cluster_number, i in enumerate(np.unique(clusters)):
        clusters[clusters == i] = cluster_number
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
    cluster_indexes = np.where(clusters == cluster_index)[0]
    labels_counts, _ = np.histogram(cluster_indexes // 100, range(10))
    common_label = np.argmax(labels_counts)

    cluster_size = np.count_nonzero(clusters == cluster_index)

    labels_total_count = sum(labels_counts)
    percentage = labels_counts[common_label] / labels_total_count
    percentage = f"{percentage * 100:.2f}%"

    print(
        f"cluster={cluster_index}, cluster_size={cluster_size}, common_label={common_label},percentage={percentage}")
    return labels_counts[common_label]


if __name__ == '__main__':
    data = np.load('mnist_all.npz')

    question_2_c()
    question_2_d()
    question_2_e()
