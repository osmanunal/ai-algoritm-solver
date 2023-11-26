import yaml
from knn import knn
from k_means import k_means
if __name__ == "__main__":
    with open('questions.yml', 'r') as file:
        data = yaml.safe_load(file)

    if 'KNN' in data:
        k = data['KNN']['k']
        quiz = data['KNN']['quiz']
        knn(data, k, quiz)

    if 'K-Means' in data:
        values = data['K-Means']['values']
        initial_centroids = data['K-Means']['cendroids']
        k = len(initial_centroids)
        k_means(values, initial_centroids, k)