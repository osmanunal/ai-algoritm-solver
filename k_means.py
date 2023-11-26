def k_means(values, initial_centroids, k, max_iterations=100):
    centroids = initial_centroids[:]
    for iteration in range(max_iterations):
        distances = calculate_distances(values, centroids)
        clusters = assign_to_clusters(distances)
        new_centroids = update_centroids(values, clusters, k)

        print_iteration_info(iteration, values, centroids, distances, clusters, new_centroids)

        if centroids == new_centroids:
            print("Converged after", iteration + 1, "iterations.")
            break

        centroids = new_centroids

def calculate_distances(values, centroids):
    distances = []
    for value in values:
        distances.append([abs(value - centroid) for centroid in centroids])
    return distances

def assign_to_clusters(distances):
    return [min(enumerate(d), key=lambda x: x[1])[0] for d in distances]

def update_centroids(values, clusters, k):
    new_centroids = []
    for cluster_num in range(k):
        cluster_values = [values[i] for i, c in enumerate(clusters) if c == cluster_num]
        if cluster_values:
            new_centroids.append(sum(cluster_values) / len(cluster_values))
        else:
            # If a cluster is empty, keep the centroid unchanged
            new_centroids.append(None)
    return new_centroids

def print_iteration_info(iteration, values, centroids, distances, clusters, new_centroids):
    print(f"Iteration {iteration + 1}:")
    print("Values:", [f'{val:.4f}' for val in values])
    print("Centroids:", [f'{centroid:.4f}' for centroid in centroids])
    print("Distances:", [[f'{dist:.4f}' for dist in row] for row in distances])
    print("Nearest Cluster:", clusters)
    print("New Centroids:", [f'{centroid:.4f}' if centroid is not None else None for centroid in new_centroids])
    print("-------------------")

