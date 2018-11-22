import numpy as np
import math
from sklearn.cluster import KMeans


class ModularityTechnique:

    def __init__(self, embeddings, groups=2):
        # Model parameter lambda
        self._lambda = 1
        print('Diversity based Modularity technique')
        self._embeddings = embeddings
        self._group_count = groups
        self._node_count = embeddings.shape[0]
        self._node_dimension = embeddings.shape[1]
        self._dissimilarity_matrix = None
        self._degree_list = []
        self._total_degree = None
        self._Q_matrix = None
        self._M_matrix = None
        self._prepare_dissimilarity_matrix()
        self._prepare_degree_list()
        self._prepare_Q_matrix()
        self._prepare_M_matrix()
        # self._divide_into_clusters()

    def _prepare_dissimilarity_matrix(self):
        self._dissimilarity_matrix = np.zeros((self._node_count, self._node_count))
        for i in range(self._node_count):
            for j in range(self._node_count):
                distance = math.pow(np.linalg.norm(self._embeddings[i, :] - self._embeddings[j, :]), 2)
                self._dissimilarity_matrix[i][j] = distance

    def _prepare_degree_list(self):
        for i in range(self._node_count):
            degree_i = np.sum(self._dissimilarity_matrix[i, :])
            self._degree_list.append(degree_i)
        self._total_degree = np.sum(self._degree_list)

    def _prepare_Q_matrix(self):
        self._Q_matrix = np.zeros((self._node_count, self._node_count))
        for i in range(self._node_count):
            for j in range(self._node_count):
                q_ij = self._dissimilarity_matrix[i][j] - (self._degree_list[i] * self._degree_list[j])/self._total_degree
                self._Q_matrix[i][j] = q_ij

    def _prepare_M_matrix(self):
        self._M_matrix = np.zeros((self._node_count, self._node_count))
        all_one_matrix = self._lambda * np.ones((self._node_count, self._node_count))
        self._M_matrix = self._Q_matrix - all_one_matrix
        # print(all_one_matrix)

    def divide_into_clusters(self):
        eigen_vals, eigen_vecs = np.linalg.eigh(self._M_matrix)
        X_data = np.array(eigen_vecs[-1]).reshape((len(eigen_vecs[-1]), 1))
        kmeans = KMeans(n_clusters=self._group_count).fit(X_data)
        clusters_indices_list = []
        for i in range(self._group_count):
            cluster_i = [j for j in range(len(kmeans.labels_)) if kmeans.labels_[j] == i]
            clusters_indices_list.append(cluster_i)
        # negative_cluster = [i for i in range(len(eigen_vecs[-1])) if eigen_vecs[-1][i] < 0]
        # positive_cluster = [i for i in range(len(eigen_vecs[-1])) if eigen_vecs[-1][i] >= 0]
        # positive_cluster_embedding = np.zeros((len(positive_cluster), self._node_dimension))
        # negative_cluster_embedding = np.zeros((len(negative_cluster), self._node_dimension))
        # for i in range(len(positive_cluster)):
        #     positive_cluster_embedding[i] = self._embeddings[positive_cluster[i]]
        # for i in range(len(negative_cluster)):
        #     negative_cluster_embedding[i] = self._embeddings[negative_cluster[i]]
        # assert len(negative_cluster) + len(positive_cluster) == self._node_count
        # # print(positive_cluster)
        # # print(negative_cluster)
        # # print(positive_cluster_embedding.shape)
        # # print(negative_cluster_embedding.shape)
        # embeddings = [positive_cluster_embedding, negative_cluster_embedding]
        # print('Second largest eigen vector :: ', eigen_vecs[-1])
        embeddings = []
        for i in range(self._group_count):
            cluster_i_indices = clusters_indices_list[i]
            embeddings_i = np.zeros((len(cluster_i_indices), self._node_dimension))
            for j in range(len(cluster_i_indices)):
                embeddings_i[j] = self._embeddings[cluster_i_indices[j]]
            embeddings.append(embeddings_i)
        return embeddings, clusters_indices_list
