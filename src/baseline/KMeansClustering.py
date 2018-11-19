from sklearn.cluster import KMeans
import numpy as np


class StandardKMeans:

    def __init__(self, embeddings, groups=2):
        self._embeddings = embeddings
        self._group_count = groups
        self._embedding_dimension = embeddings.shape[1]
        self._cluster_embeddings = None

    def cluster(self):
        cluster_labels = KMeans(n_clusters=self._group_count, n_jobs=-1).fit_predict(self._embeddings)
        clusters = []
        for label in range(self._group_count):
            cluster_i_index = [j for j in range(len(cluster_labels)) if cluster_labels[j] == label]
            clusters.append(cluster_i_index)
        cluster_embeddings = []
        for i in range(len(clusters)):
            cluster_i_indices = clusters[i]
            cluster_embed = np.zeros((len(cluster_i_indices), self._embedding_dimension))
            for j in range(len(cluster_i_indices)):
                embed_j = self._embeddings[cluster_i_indices[j], :]
                cluster_embed[j] = embed_j
            cluster_embeddings.append(cluster_embed)
        self._cluster_embeddings = cluster_embeddings
        return self._cluster_embeddings, clusters
