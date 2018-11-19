import random
import numpy as np


class RandomPartitions:

    def __init__(self, embeddings, groups=2):
        self._embeddings = embeddings
        self._group_count = groups
        self._node_count = self._embeddings.shape[0]
        self._embedding_dimension = self._embeddings.shape[1]
        self._clusters_embeddings = None

    def cluster(self):
        node_indices_list = np.arange(self._node_count)
        np.random.shuffle(node_indices_list)
        print(node_indices_list)
        element_count_each_group = int(self._node_count / self._group_count)
        clusters = []
        for index in range(self._group_count):
            if index != self._group_count - 1:
                cluster_i_index = node_indices_list[index*element_count_each_group: (index+1)*element_count_each_group]
            else:
                cluster_i_index = node_indices_list[index*element_count_each_group: -1]
            clusters.append(cluster_i_index)

        cluster_embeddings = []
        for i in range(self._group_count):
            cluster_i_indices = clusters[i]
            cluster_embed = np.zeros((len(cluster_i_indices), self._embedding_dimension))
            for j in range(len(cluster_i_indices)):
                embed_j = self._embeddings[cluster_i_indices[j], :]
                cluster_embed[j] = embed_j
            cluster_embeddings.append(cluster_embed)
        self._clusters_embeddings = cluster_embeddings
        return self._clusters_embeddings, clusters

