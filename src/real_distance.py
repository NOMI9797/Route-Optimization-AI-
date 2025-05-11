import openrouteservice
import numpy as np

def get_distance_matrix(coords, api_key):
    """
    Fetch a real-world driving distance matrix from OpenRouteService.
    Args:
        coords: List of [lon, lat] pairs
        api_key: Your ORS API key
    Returns:
        np.ndarray: 2D array of distances in kilometers
    """
    client = openrouteservice.Client(key=api_key)
    matrix = client.distance_matrix(
        locations=coords,
        profile='driving-car',
        metrics=['distance'],
        units='km'
    )
    return np.array(matrix['distances']) 