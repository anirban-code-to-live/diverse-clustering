import networkx as nx
import pandas as pd
import os
from scipy.sparse import csgraph
import matplotlib
from matplotlib import pyplot as plt
from src import ModularityTechnique as mt
from src import EvaluationMetric
import numpy as np
from src.baseline.KMeansClustering import StandardKMeans as kmeans
from src.baseline.RandomPartition import RandomPartitions as random_partitions
from src.baseline.UniformKMeans import UniformKMeans
from src.visualization import Visualization as vs


if __name__ == '__main__':
    print('Welcome to the world full of diversity!!\n')

    graph_name = 'karate'
    group_count = 2
    # Use this code snippet if you want to use adjacency matrix directly
    graph_path = '../data/' + graph_name + '/graph.gml'
    graph = nx.read_gml(graph_path, label='id')
    graph_embedding_matrix = nx.to_numpy_matrix(graph)
    print('Shape of adjacency matrix :: ', graph_embedding_matrix.shape)
    # end

    # Use this code snippet for generated embeddings using node2vec
    graph_embed_path = '../data/' + graph_name + '/graph_embed.csv'
    graph_embedding_matrix = pd.read_csv(graph_embed_path, header=None, sep=' ').values
    print('Embedding shape :: ', graph_embedding_matrix.shape)
    # End

    # Modularity technique
    print('Diversity based clustering using Modularity technique')
    modularity_approach = mt.ModularityTechnique(embeddings=graph_embedding_matrix, groups=group_count)
    embeddings, clusters = modularity_approach.divide_into_clusters()
    # print(positive_cluster)
    # print(negative_cluster)
    # total_group_diversity = EvaluationMetric.evaluated_total_group_diversity(embeddings, group_count=2)
    # print('Total group diversity :: ', total_group_diversity)
    # variation_group_means = EvaluationMetric.evaluate_variation_group_means(embeddings, group_count=2)
    # print('Variation in group means :: ', variation_group_means)
    # variation_group_variance = EvaluationMetric.evaluate_variation_group_variance(embeddings, group_count=2)
    # print('Variation in group variance :: ', variation_group_variance)
    total_intra_group_diversity = EvaluationMetric.evaluate_intra_diversity(embeddings, group_count=group_count)
    print('Total intra-group diversity :: ', total_intra_group_diversity)
    total_inter_group_diversity = EvaluationMetric.evaluate_inter_diversity(embeddings, group_count=group_count)
    print('Total inter-group diversity :: ', total_inter_group_diversity)
    # Visualize graph with diverse clusters
    vs.visualize_graph(graph, clusters, technique='modularity', graph_name=graph_name)
    vs.tsne_visualization(embeddings, graph, clusters, technique='modularity', graph_name=graph_name)

    # Baseline method - KMeans
    print('Diversity based clustering using standard KMeans')
    standard_kmeans = kmeans(embeddings=graph_embedding_matrix, groups=group_count)
    embeddings, clusters = standard_kmeans.cluster()
    # print(clusters)
    # total_group_diversity = EvaluationMetric.evaluated_total_group_diversity(embeddings, group_count=2)
    # print('Total group diversity :: ', total_group_diversity)
    # variation_group_means = EvaluationMetric.evaluate_variation_group_means(embeddings, group_count=2)
    # print('Variation in group means :: ', variation_group_means)
    # variation_group_variance = EvaluationMetric.evaluate_variation_group_variance(embeddings, group_count=2)
    # print('Variation in group variance :: ', variation_group_variance)
    total_intra_group_diversity = EvaluationMetric.evaluate_intra_diversity(embeddings, group_count=group_count)
    print('Total intra-group diversity :: ', total_intra_group_diversity)
    total_inter_group_diversity = EvaluationMetric.evaluate_inter_diversity(embeddings, group_count=group_count)
    print('Total inter-group diversity :: ', total_inter_group_diversity)
    # # Visualize graph with diverse clusters
    vs.visualize_graph(graph, clusters, technique='standard-kmeans', graph_name=graph_name)
    vs.tsne_visualization(embeddings, graph, clusters, technique='standard-kmeans', graph_name=graph_name)

    # Baseline method - Random Partition
    print('Diversity based clustering using Random Partition')
    random_partition = random_partitions(embeddings=graph_embedding_matrix, groups=group_count)
    embeddings, clusters = random_partition.cluster()
    # total_group_diversity = EvaluationMetric.evaluated_total_group_diversity(embeddings, group_count=2)
    # print('Total group diversity :: ', total_group_diversity)
    # variation_group_means = EvaluationMetric.evaluate_variation_group_means(embeddings, group_count=2)
    # print('Variation in group means :: ', variation_group_means)
    # variation_group_variance = EvaluationMetric.evaluate_variation_group_variance(embeddings, group_count=2)
    # print('Variation in group variance :: ', variation_group_variance)
    total_intra_group_diversity = EvaluationMetric.evaluate_intra_diversity(embeddings, group_count=group_count)
    print('Total intra-group diversity :: ', total_intra_group_diversity)
    total_inter_group_diversity = EvaluationMetric.evaluate_inter_diversity(embeddings, group_count=group_count)
    print('Total inter-group diversity :: ', total_inter_group_diversity)
    # # Visualize graph with diverse clusters
    vs.visualize_graph(graph, clusters, technique='random-partition', graph_name=graph_name)
    vs.tsne_visualization(embeddings, graph, clusters, technique='random-partition', graph_name=graph_name)

    # Baseline method - Uniform KMeans
    print('Diversity based clustering using Uniform KMeans')
    uniform_kmeans = UniformKMeans(embeddings=graph_embedding_matrix, groups=group_count)
    embeddings, clusters = uniform_kmeans.cluster()
    # print(clusters)
    # total_group_diversity = EvaluationMetric.evaluated_total_group_diversity(embeddings, group_count=2)
    # print('Total group diversity :: ', total_group_diversity)
    # variation_group_means = EvaluationMetric.evaluate_variation_group_means(embeddings, group_count=2)
    # print('Variation in group means :: ', variation_group_means)
    # variation_group_variance = EvaluationMetric.evaluate_variation_group_variance(embeddings, group_count=2)
    # print('Variation in group variance :: ', variation_group_variance)
    total_intra_group_diversity = EvaluationMetric.evaluate_intra_diversity(embeddings, group_count=group_count)
    print('Total intra-group diversity :: ', total_intra_group_diversity)
    total_inter_group_diversity = EvaluationMetric.evaluate_inter_diversity(embeddings, group_count=group_count)
    print('Total inter-group diversity :: ', total_inter_group_diversity)
    # # Visualize graph with diverse clusters
    vs.visualize_graph(graph, clusters, technique='uniform-kmeans', graph_name=graph_name)
    vs.tsne_visualization(embeddings, graph, clusters, technique='uniform-kmeans', graph_name=graph_name)


