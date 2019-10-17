import numpy as np
from sklearn.cluster import KMeans


class DivSimGrouping:

    def __init__(self, adjacency_mat, groups=2):
        self._adj_mat = adjacency_mat
        self._group_count = groups
        self._node_count = adjacency_mat.shape[0]
        # print('Adjacency matrix dimension :: ', adjacency_mat.shape)
        self._degree_vec = self._prepare_degree_vector()

    def _prepare_degree_vector(self):
        one_mat = np.ones((self._node_count, 1))
        degree_vec = np.matmul(self._adj_mat, one_mat)
        return degree_vec

    def cluster(self):
        # print('Degree vector dimension :: ', np.matmul(self._degree_vec, self._degree_vec.T))
        modified_adj_mat = self._adj_mat + np.matmul(self._degree_vec, self._degree_vec.T) + np.ones((self._node_count, self._node_count))
        # print('Modified adjacency matrix dimension :: ', modified_adj_mat.shape)
        eigen_vals, eigen_vecs = np.linalg.eigh(modified_adj_mat)
        min_eigen_vec = np.array(eigen_vecs[0]).reshape((eigen_vecs.shape[0], 1))

        # clusters_indices_list = []
        # cluster_0 = [j for j in range(self._node_count) if min_eigen_vec[j] < 0]
        # cluster_1 = [j for j in range(self._node_count) if min_eigen_vec[j] >= 0]
        # clusters_indices_list.append(cluster_0)
        # clusters_indices_list.append(cluster_1)

        X_data = np.array(eigen_vecs[0]).reshape((eigen_vecs.shape[0], 1))
        kmeans = KMeans(n_clusters=self._group_count).fit(X_data)
        clusters_indices_list = []
        for i in range(self._group_count):
            cluster_i = [j for j in range(len(kmeans.labels_)) if kmeans.labels_[j] == i]
            clusters_indices_list.append(cluster_i)

        embeddings = []
        for i in range(self._group_count):
            cluster_i_indices = clusters_indices_list[i]
            embeddings_i = np.zeros((len(cluster_i_indices), self._node_count))
            for j in range(len(cluster_i_indices)):
                embeddings_i[j] = self._adj_mat[cluster_i_indices[j]]
            embeddings.append(embeddings_i)

        # print(clusters_indices_list)
        print(len(clusters_indices_list[0]))
        print(len(clusters_indices_list[1]))
        return embeddings, clusters_indices_list