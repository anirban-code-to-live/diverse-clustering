import numpy as np


# Metric to capture the total variance of all groups. It should be high ideally.
def evaluated_total_group_diversity(group_embedding_list, group_count):
    assert len(group_embedding_list) == group_count
    total_diversity = 0
    for i in range(group_count):
        group_i = group_embedding_list[i]
        embedding_dimension = group_i.shape[1]
        group_covariance = np.cov(group_i, rowvar=False)
        if embedding_dimension == 1:
            group_diversity = group_covariance
        else:
            group_diversity = np.linalg.det(group_covariance)
        total_diversity += group_diversity
    return total_diversity


# Metric to capture the uniformity across groups. Should be low ideally.
def evaluate_variation_group_means(group_embedding_list, group_count):
    assert len(group_embedding_list) == group_count
    embedding_dimension = group_embedding_list[0].shape[1]
    print('Embedding dimension :: ', embedding_dimension)
    group_mean_mat = np.zeros((group_count, embedding_dimension))
    for i in range(group_count):
        group_i = group_embedding_list[i]
        group_mean = np.mean(group_i, axis=0)
        group_mean_mat[i] = group_mean
    group_mean_cov_mat = np.cov(group_mean_mat, rowvar=False)
    total_variation_group_means = np.linalg.det(group_mean_cov_mat)
    return total_variation_group_means


# Metric to capture the uniformity across groups. Should be low ideally.
def evaluate_variation_group_variance(group_embedding_list, group_count):
    assert len(group_embedding_list) == group_count
    embedding_dimension = group_embedding_list[0].shape[1]
    print('Embedding dimension :: ', embedding_dimension)
    avg_cov_mat = _calculate_average_group_covariance(group_embedding_list, group_count)
    variation_group_variance = 0
    for i in range(group_count):
        group_i = group_embedding_list[i]
        group_cov = np.cov(group_i, rowvar=False)
        group_i_variation = np.linalg.norm(group_cov - avg_cov_mat)
        variation_group_variance += group_i_variation
    return variation_group_variance


def _calculate_average_group_covariance(group_embedding_list, group_count):
    assert len(group_embedding_list) == group_count
    embedding_dimension = group_embedding_list[0].shape[1]
    print('Embedding dimension :: ', embedding_dimension)
    avg_group_cov_mat = np.zeros((embedding_dimension, embedding_dimension))
    for i in range(group_count):
        group_i = group_embedding_list[i]
        group_cov = np.cov(group_i, rowvar=False)
        avg_group_cov_mat += group_cov
    avg_group_cov_mat = avg_group_cov_mat/group_count
    return avg_group_cov_mat
