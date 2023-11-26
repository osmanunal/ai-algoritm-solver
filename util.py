import math

def min_max_scaling(value, min_val, max_val, new_min, new_max):
    return ((value - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min

def knn_euclidean_distance(point1, point2):
    return math.sqrt((point1['X'] - point2['X'])**2 + (point1['Y'] - point2['Y'])**2)
