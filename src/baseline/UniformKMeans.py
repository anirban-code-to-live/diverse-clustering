from sklearn.cluster import KMeans
import numpy as np


class UniformKMeans:

    def __init__(self, embeddings, groups=2):
        self._embeddings = embeddings
        self._group_count = groups
        self._embedding_dimension = embeddings.shape[1]
        self._element_count = embeddings.shape[0]
        self._cluster_embeddings = None

    def cluster(self):
        cluster_labels = KMeans(n_clusters=self._group_count, n_jobs=-1).fit_predict(self._embeddings)
        cluster_indices_list = []
        for label in range(self._group_count):
            cluster_i_indices = [j for j in range(len(cluster_labels)) if cluster_labels[j] == label]
            cluster_indices_list.append(cluster_i_indices)

        assert len(cluster_indices_list) == self._group_count
        print('Before shuffling -- Cluster indices list :: ', cluster_indices_list)

        # Shuffle each cluster
        for i in range(self._group_count):
            # np.random.shuffle(cluster_embeddings[i])
            np.random.shuffle(cluster_indices_list[i])

        print('After shuffling -- Cluster indices list :: ', cluster_indices_list)

        cluster_embeddings = []
        for i in range(self._group_count):
            cluster_i_indices = cluster_indices_list[i]
            cluster_embed = np.zeros((len(cluster_i_indices), self._embedding_dimension))
            for j in range(len(cluster_i_indices)):
                embed_j = self._embeddings[cluster_i_indices[j], :]
                cluster_embed[j] = embed_j
            cluster_embeddings.append(cluster_embed)

        assert len(cluster_indices_list) == len(cluster_embeddings)

        # Calculate uniform clusters
        cluster_lengths = []  # List containing length of each cluster
        for i in range(self._group_count):
            cluster_lengths.append(len(cluster_indices_list[i]))
        min_cluster_length = np.min(cluster_lengths)
        print('Minimum cluster length :: ', min_cluster_length)
        # List stores number of elements in each uniform cluster contributed by every cluster obtained earlier
        cluster_iteration_count = [int(length/self._group_count) for length in cluster_lengths]
        print('Cluster iteration count :: ', cluster_iteration_count)
        # End
        # List stores number of elements remaining in each cluster after the previous iteration count
        cluster_remaining_count = [cluster_lengths[i] - self._group_count*cluster_iteration_count[i]
                                   for i in range(self._group_count)]
        # End
        # Find out length of each uniform cluster
        uniform_cluster_lengths = [int(self._element_count/self._group_count) for _ in range(self._group_count)]
        print('Uniform cluster lengths :: ', uniform_cluster_lengths)
        remaining_element_count = int(self._element_count - int(self._element_count/self._group_count)*self._group_count)
        print('Remaining element count :: ', remaining_element_count)
        for i in range(remaining_element_count):
            uniform_cluster_lengths[i] += uniform_cluster_lengths[i] + 1
        print('Uniform cluster lengths after adding remaining counts :: ', uniform_cluster_lengths)
        # End

        uniform_cluster_embeddings = []  # List containing embedding matrix of each cluster
        uniform_cluster_indices_list = []  # List containing node indices of each uniform cluster
        for i in range(self._group_count):
            uniform_cluster_length = uniform_cluster_lengths[i]
            uniform_cluster_i = np.zeros((uniform_cluster_length, self._embedding_dimension))
            uniform_cluster_embeddings.append(uniform_cluster_i)
            uniform_cluster_indices_list.append([])

        uniform_cluster_item_count = [0 for _ in range(self._group_count)]

        for i in range(len(cluster_iteration_count)):
            iteration_count = cluster_iteration_count[i]
            if iteration_count > 0:
                temp_count = 0
                cluster_indices_i = cluster_indices_list[i]
                while temp_count < iteration_count:
                    for j in range(self._group_count):
                        cluster_j_item = uniform_cluster_item_count[j]
                        if cluster_j_item < uniform_cluster_lengths[j]:
                            uniform_cluster_embedding = uniform_cluster_embeddings[j]
                            embedding_index = cluster_indices_i[temp_count*self._group_count + j]
                            uniform_cluster_embedding[cluster_j_item] = self._embeddings[embedding_index]
                            uniform_cluster_indices_list[j].append(embedding_index)
                            uniform_cluster_item_count[j] += 1
                    temp_count += 1

        cluster_index = 0
        for i in range(len(cluster_remaining_count)):
            remaining_count = cluster_remaining_count[i]
            temp_count = 0
            while temp_count < remaining_count:
                uniform_cluster_embedding = uniform_cluster_embeddings[cluster_index]
                cluster_indices_i = cluster_indices_list[i]
                cluster_j_item = uniform_cluster_item_count[cluster_index]
                if cluster_j_item < uniform_cluster_lengths[cluster_index]:
                    iteration_count = cluster_iteration_count[i]
                    embedding_index = cluster_indices_i[iteration_count*self._group_count + temp_count]
                    uniform_cluster_embedding[cluster_j_item] = self._embeddings[embedding_index]
                    uniform_cluster_indices_list[cluster_index].append(embedding_index)
                    uniform_cluster_item_count[cluster_index] += 1
                    temp_count += 1
                cluster_index += 1
                cluster_index %= self._group_count
        self._cluster_embeddings = uniform_cluster_embeddings
        return self._cluster_embeddings, uniform_cluster_indices_list

