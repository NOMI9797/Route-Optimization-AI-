from sklearn.cluster import KMeans
from typing import List, Tuple
import numpy as np

def cluster_cities(cities: List[Tuple[str, float, float]], n_clusters: int, random_state: int = 42):
    coords = np.array([[city[1], city[2]] for city in cities])
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(coords)
    cluster_centers = kmeans.cluster_centers_
    return cluster_labels, cluster_centers 