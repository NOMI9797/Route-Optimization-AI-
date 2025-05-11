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


def get_route_geometry(coord_from, coord_to, api_key):
    """
    Fetch the route geometry (polyline) between two coordinates from OpenRouteService.
    Args:
        coord_from: [lon, lat] of the start point
        coord_to: [lon, lat] of the end point
        api_key: Your ORS API key
    Returns:
        List of [lat, lon] points representing the route
    """
    client = openrouteservice.Client(key=api_key)
    route = client.directions(
        coordinates=[coord_from, coord_to],
        profile='driving-car',
        format='geojson'
    )
    geometry = route['features'][0]['geometry']['coordinates']
    # Convert [lon, lat] to [lat, lon] for folium
    return [[lat, lon] for lon, lat in geometry] 