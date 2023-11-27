from util import knn_euclidean_distance, min_max_scaling
import math

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

def knn(data, k, tests):
    # Min-Max Scaling Test
    x_train_values = [point['X'] for point in data['knn']['points']]
    y_train_values = [point['Y'] for point in data['knn']['points']]
    z_train_values = [point['Z'] for point in data['knn']['points']]

    train_min_x, train_max_x = min(x_train_values), max(x_train_values)
    train_min_y, train_max_y = min(y_train_values), max(y_train_values)
    train_min_z, train_max_z = min(z_train_values), max(z_train_values)

    # Min-Max Scaling Test
    print("\nMin Max Değerleri:")
    print(f"Min X: {train_min_x:.4f}, Max X: {train_max_x:.4f}")
    print(f"Min Y: {train_min_y:.4f}, Max Y: {train_max_y:.4f}")
    print(f"Min Z: {train_min_z:.4f}, Max Z: {train_max_z:.4f}")

    for test in tests:
        test['X'] = min_max_scaling(test['X'], train_min_x, train_max_x)
        test['Y'] = min_max_scaling(test['Y'], train_min_y, train_max_y)
        test['Z'] = min_max_scaling(test['Z'], train_min_z, train_max_z)
        # test normalization
        print("\nNORMALİZE EDİLMEMİŞ TEST VERİLERİ")
        print(f"X={test['X']:.4f}, Y={test['Y']:.4f}, Z={test['Z']:.4f}")

    print("-----------------------------------------------\n")


    for point in data['knn']['points']:
        point['X'] = min_max_scaling(point['X'], train_min_x, train_max_x)
        point['Y'] = min_max_scaling(point['Y'], train_min_y, train_max_y)
        point['Z'] = min_max_scaling(point['Z'], train_min_z, train_max_z)

    # Test verilerini kullanarak tahmin yapma

    for test in tests:
        distances = []
        print("NORMALİZE EDİLMİŞ VERİLER")
        for id,point in enumerate(data['knn']['points']):
            distance = knn_euclidean_distance(point, test)
            distances.append((id+1,point, distance))
            print(f"{id+1}) X={point['X']:.4f}, Y={point['Y']:.4f}, Z={point['Z']:.4f}, Label={point['Label']}, Uzaklık={distance:.4f}")



        distances.sort(key=lambda x: x[2])  # Uzaklıklara göre sıralama

        k_nearest_neighbors = distances[:k]


        print(f"\nKNN Sonuçları:")
        print(f"TEST POINT: X={test['X']:.4f}, Y={test['Y']:.4f}, Label='', Uzaklık={distances[0][2]:.4f}")
        print(f"K en Yakın Komşular(uzaklıklarına göre sıralanmış ilk {k} veri):")
        for  sira,neighbor, distance in k_nearest_neighbors:
           # print(f"X={neighbor['X']:.4f}, Y={neighbor['Y']:.4f}, Label={neighbor['Label']}, Uzaklık={distance:.4f}")
            print(f"{sira}) X={neighbor['X']:.4f}, Y={neighbor['Y']:.4f}, Z={neighbor['Z']:.4f}, Label={neighbor['Label']}, Uzaklık={distance:.4f}")

        labels = [neighbor[1]['Label'] for neighbor in k_nearest_neighbors]
        result_label = max(set(labels), key=labels.count)

        print(f"\nKNN Prediction: {result_label}")
        print("--------------------------------------------------\n")

