import numpy as np
import matplotlib.pyplot as plt

# Generate example data
np.random.seed(0)  # For reproducibility
data = np.random.rand(5, 2)  # 5 points in 2D

# Function to compute the Euclidean distance matrix
def compute_distance_matrix(data):
    num_points = data.shape[0]
    dist_matrix = np.sqrt(((data[:, np.newaxis] - data) ** 2).sum(axis=2))
    return dist_matrix

# Function to perform Single Linkage Hierarchical Clustering
def single_linkage_clustering(data):
    num_points = data.shape[0]
    
    # Step 1: Compute initial distance matrix
    dist_matrix = compute_distance_matrix(data)
    
    # Step 2: Initialize clusters
    clusters = [[i] for i in range(num_points)]
    
    # Step 3: Initialize the linkage matrix
    linkage_matrix = []
    
    # Step 4: Perform clustering
    while len(clusters) > 1:
        # Find the pair of clusters with the minimum distance
        min_dist = np.inf
        to_merge = (0, 0)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = np.min(dist_matrix[np.ix_(clusters[i], clusters[j])])
                if dist < min_dist:
                    min_dist = dist
                    to_merge = (i, j)
        
        # Merge the clusters
        i, j = to_merge
        new_cluster = clusters[i] + clusters[j]
        clusters = [clusters[k] for k in range(len(clusters)) if k not in (i, j)] + [new_cluster]
        
        # Update the distance matrix
        new_dist_matrix = np.full((len(clusters), len(clusters)), np.inf)
        for m in range(len(clusters)):
            for n in range(m + 1, len(clusters)):
                dist = np.min(dist_matrix[np.ix_(clusters[m], clusters[n])])
                new_dist_matrix[m, n] = new_dist_matrix[n, m] = dist
        
        dist_matrix = new_dist_matrix
        
        # Save the linkage information
        linkage_matrix.append([i, j, min_dist, len(new_cluster)])
    
    return linkage_matrix

# Perform clustering
linkage_matrix = single_linkage_clustering(data)

# Print linkage matrix
print("Linkage Matrix:")
for link in linkage_matrix:
    print(link)

# Plotting for visualization
def plot_clusters(data, clusters):
    plt.figure(figsize=(10, 7))
    colors = plt.cm.jet(np.linspace(0, 1, len(clusters)))
    for cluster, color in zip(clusters, colors):
        cluster_points = data[cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=color, label=f'Cluster {cluster[0]}')
    plt.title('Single Linkage Clustering')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.show()

# Plot initial data
plot_clusters(data, [[i] for i in range(10)])
