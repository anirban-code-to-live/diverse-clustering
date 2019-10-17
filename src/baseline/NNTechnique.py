from sklearn.cluster import KMeans
import math
import numpy as np


class NearestNeighbor:

    def __init__(self, embeddings, groups=2):
        # Model parameters - alpha and beta
        self._alpha = 1
        self._beta = 1
        print(embeddings.shape)
        self._embeddings = embeddings
        self._group_count = groups
        self._item_count = embeddings.shape[0]
        self._embedding_size = embeddings.shape[1]
        self._assigned_group = None
        self._group_centers = None
        self._group_item_counts = None
        self._find_initial_k_centers()

    def _find_initial_k_centers(self):
        kmeans = KMeans(n_clusters=self._group_count, n_jobs=-1).fit(self._embeddings)
        self._group_centers = kmeans.cluster_centers_

    # def maximize_diversity_similarity(self):
    #     for i in range(self._item_count):
    #         diversity_similarity_values = []
    #         for j in range(self._group_count):
    #             variance = math.pow(np.linalg.norm(self._embeddings[i, :] - self._group_centers[j]), 2)
    #             new_mean =