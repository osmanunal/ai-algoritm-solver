import yaml
from knn import knn
if __name__ == "__main__":
    with open('questions.yml', 'r') as file:
        data = yaml.safe_load(file)

    k = data['KNN']['k']
    quiz = data['KNN']['quiz']

    if data['KNN'] is not None:
        knn(data, k, quiz)
