import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


# Visualize a network with different clusters having different colored nodes
def visualize_graph(graph, cluster_indices_list, technique, graph_name):
    color_list = ['r', 'g', 'b']
    cluster_count = len(cluster_indices_list)
    if cluster_count > len(color_list):
        print('Increase the available colors in your color list.')
        return
    # print('No. of clusters :: ', cluster_count)
    node_list = list(graph.nodes())
    # print('Node list :: ', node_list)
    # print('Cluster-1 :: ', cluster_indices_list[0])
    # print('Cluster-2 :: ', cluster_indices_list[1])
    pos = nx.spring_layout(graph)
    print(pos)

    # Special handling for karate graph
    if graph_name == 'karate':
        for i in range(cluster_count):
            cluster_i = cluster_indices_list[i]
            for j in range(len(cluster_i)):
                cluster_i[j] = cluster_i[j] + 1
    else:
        for i in range(cluster_count):
            cluster_i = cluster_indices_list[i]
            for j in range(len(cluster_i)):
                cluster_i[j] = cluster_i[j]

    # print('Cluster-1 :: ', cluster_indices_list[0])
    # print('Cluster-2 :: ', cluster_indices_list[1])

    label_dict = {}
    for i in range(len(node_list)):
        label_dict[node_list[i]] = node_list[i]
    # for i in range(len(self._nodes)):
    #     if self._nodes[i] in cluster1:
    #         pos[self._nodes[i]][1] += 2
    for i in range(cluster_count):
        cluster_i = cluster_indices_list[i]
        nx.draw_networkx_nodes(graph, pos, cluster_i, node_color=color_list[i], node_shape='o', node_size=100)

    # nx.draw_networkx_nodes(graph, pos, cluster2, node_color='g', node_shape='s', node_size=100)
    # nx.draw(dolphin_graph, with_labels=True, node_color=node_colors.ravel())
    nx.draw_networkx_edges(graph, pos, alpha=0.8)
    nx.draw_networkx_labels(graph, pos, labels=label_dict, font_size=7)
    plt.axis('off')
    # plt.show()
    plt.savefig('../plots/' + graph_name + '/graph_' + technique + '.png')
    plt.clf()


def tsne_visualization(cluster_embeddings, graph, cluster_indices_list, technique, graph_name):
    assert len(cluster_embeddings) == len(cluster_indices_list)
    node_count = np.sum([len(cluster_i) for cluster_i in cluster_indices_list])
    # print('Total number of nodes in the network :: ', node_count)
    embedding_dimension = cluster_embeddings[0].shape[1]
    # print('Dimension of embedding :: ', embedding_dimension)
    embeddings = np.zeros((node_count, embedding_dimension))

    # Special handling for karate graph
    if graph_name == 'karate':
        for i in range(len(cluster_indices_list)):
            cluster_i = cluster_indices_list[i]
            for j in range(len(cluster_i)):
                cluster_i[j] = cluster_i[j] - 1
    else:
        for i in range(len(cluster_indices_list)):
            cluster_i = cluster_indices_list[i]
            for j in range(len(cluster_i)):
                cluster_i[j] = cluster_i[j]

    for i in range(len(cluster_indices_list)):
        cluster_i = cluster_indices_list[i]
        embeddings_i = cluster_embeddings[i]
        for j in range(len(cluster_i)):
            embeddings[cluster_i[j]] = embeddings_i[j]
    tsne_embeddings = TSNE(n_components=2).fit_transform(embeddings)
    # assert tsne_embeddings.shape == embeddings.shape
    tsne_clusters = []
    for i in range(len(cluster_indices_list)):
        cluster_i_indices = cluster_indices_list[i]
        tsne_embeddings_i = np.zeros((len(cluster_i_indices), 2))
        for j in range(len(cluster_i_indices)):
            tsne_embeddings_i[j] = tsne_embeddings[cluster_i_indices[j]]
        tsne_clusters.append(tsne_embeddings_i)

    # Plot the 2-dimensional tsne embeddings
    color_list = ['r', 'g', 'b']
    ax = plt.subplot(111)
    ax.grid(linestyle='--', linewidth=1)
    for i in range(len(cluster_indices_list)):
        tsne_cluster_i = tsne_clusters[i]
        x_data = tsne_cluster_i[:, 0]
        y_data = tsne_cluster_i[:, 1]
        ax.scatter(x_data, y_data, color=color_list[i])

    # plt.show()
    plt.savefig('../plots/' + graph_name + '/tsne_' + technique + '.png')
    plt.clf()




