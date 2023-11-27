from util import knn_euclidean_distance, min_max_scaling
import math

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

def knn(data, k, tests):
    # Min-Max Scaling Train
    x_train_values = [point['X'] for point in data['knn']['points']]
    y_train_values = [point['Y'] for point in data['knn']['points']]
    z_train_values = [point['Z'] for point in data['knn']['points']]

    train_min_x, train_max_x = min(x_train_values), max(x_train_values)
    train_min_y, train_max_y = min(y_train_values), max(y_train_values)
    train_min_z, train_max_z = min(z_train_values), max(z_train_values)

    # Min-Max Scaling Test
    x_test_values = [test['X'] for test in tests]
    y_test_values = [test['Y'] for test in tests]
    z_test_values = [test['Z'] for test in tests]

    test_min_x, test_max_x =  min(x_test_values), max(x_test_values)
    test_min_y, test_max_y =  min(y_test_values), max(y_test_values)
    test_min_z, test_max_z =  min(z_test_values), max(z_test_values)

    print("\nMin Max Train Değerleri:")
    print(f"Min X: {train_min_x:.4f}, Max X: {train_max_x:.4f}")
    print(f"Min Y: {train_min_y:.4f}, Max Y: {train_max_y:.4f}")
    print(f"Min Z: {train_min_z:.4f}, Max Z: {train_max_z:.4f}")
    print("----------------------------------------------")
    print("\nMin Max Test Değerleri:")
    print(f"Min X: {test_min_x:.4f}, Max X: {train_max_x:.4f}")
    print(f"Min Y: {test_min_y:.4f}, Max Y: {train_max_y:.4f}")
    print(f"Min Z: {test_min_z:.4f}, Max Z: {train_max_z:.4f}")


    for test in tests:
        test['X'] = min_max_scaling(test['X'], test_min_x, test_max_x, 0, 1)
        test['Y'] = min_max_scaling(test['Y'], test_min_y, test_max_y, 0, 1)
        test['Z'] = min_max_scaling(test['Z'], test_min_z, test_max_z , 0, 1)


    for point in data['knn']['points']:
        point['X'] = min_max_scaling(point['X'], train_min_x, train_max_x, 0, 1)
        point['Y'] = min_max_scaling(point['Y'], train_min_y, train_max_y, 0, 1)
        point['Z'] = min_max_scaling(point['Z'], train_min_z, train_max_z, 0, 1)

    # Test verilerini kullanarak tahmin yapma

    for test in tests:
        distances = []
        print("NORMALİZE EDİLMİŞ VERİLER")
        for id,point in enumerate(data['knn']['points']):
            distance = knn_euclidean_distance(point, test)
            distances.append((point, distance))
            print(f"{id+1}) X={point['X']:.4f}, Y={point['Y']:.4f}, Z={point['Z']:.4f}, Label={point['Label']}, Uzaklık={distance:.4f}")

        distances.sort(key=lambda x: x[1])  # Uzaklıklara göre sıralama

        k_nearest_neighbors = distances[:k]


        print(f"KNN Sonuçları:")
        print(f"test Point: X={test['X']:.4f}, Y={test['Y']:.4f}, Label='', Uzaklık={distances[0][1]:.4f}")
        print(f"\nK en Yakın Komşular(uzaklıklarına göre sıralanmış ilk {k} veri):")
        for neighbor, distance in k_nearest_neighbors:
           # print(f"X={neighbor['X']:.4f}, Y={neighbor['Y']:.4f}, Label={neighbor['Label']}, Uzaklık={distance:.4f}")
            print(f"X={neighbor['X']:.4f}, Y={neighbor['Y']:.4f}, Z={neighbor['Z']:.4f}, Label={neighbor['Label']}, Uzaklık={distance:.4f}")

        labels = [neighbor[0]['Label'] for neighbor in k_nearest_neighbors]
        result_label = max(set(labels), key=labels.count)

        print(f"\nKNN Prediction: {result_label}")
        print("--------------------------------------------------")

