from util import knn_euclidean_distance, min_max_scaling

def knn(data, k, quiz):
    # Min-Max Scaling
    x_values = [point['X'] for point in data['knn']['points']]
    y_values = [point['Y'] for point in data['knn']['points']]

    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)

    quiz['X'] = min_max_scaling(quiz['X'], min_x, max_x, 0, 1)
    quiz['Y'] = min_max_scaling(quiz['Y'], min_y, max_y, 0, 1)

    for point in data['knn']['points']:
        point['X'] = min_max_scaling(point['X'], min_x, max_x, 0, 1)
        point['Y'] = min_max_scaling(point['Y'], min_y, max_y, 0, 1)

    distances = []

    for point in data['knn']['points']:
        distance = knn_euclidean_distance(point, quiz)
        distances.append((point, distance))

    distances.sort(key=lambda x: x[1])  # Uzaklıklara göre sıralama

    k_nearest_neighbors = distances[:k]


    print(f"KNN Sonuçları:")
    print(f"Quiz Point: X={quiz['X']:.4f}, Y={quiz['Y']:.4f}, Label='', Uzaklık={distances[0][1]:.4f}")
    print("\nK en Yakın Komşular:")
    for neighbor, distance in k_nearest_neighbors:
        print(f"X={neighbor['X']:.4f}, Y={neighbor['Y']:.4f}, Label={neighbor['Label']}, Uzaklık={distance:.4f}")

    labels = [neighbor[0]['Label'] for neighbor in k_nearest_neighbors]
    result_label = max(set(labels), key=labels.count)

    print(f"\nKNN Tahmini: {result_label}")
