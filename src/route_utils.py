import pandas as pd
import numpy as np
from typing import List, Tuple
from geopy.distance import geodesic

def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate the distance between two points using the geodesic distance.
    
    Args:
        point1 (Tuple[float, float]): First point (latitude, longitude)
        point2 (Tuple[float, float]): Second point (latitude, longitude)
        
    Returns:
        float: Distance in kilometers
    """
    return geodesic(point1, point2).kilometers

def calculate_total_distance(route: List[int], cities: List[Tuple[str, float, float]]) -> float:
    """
    Calculate the total distance of a route visiting all cities.
    
    Args:
        route (List[int]): List of city indices representing the route
        cities (List[Tuple[str, float, float]]): List of cities with their coordinates
        
    Returns:
        float: Total distance of the route in kilometers
    """
    if not route or not cities:
        raise ValueError("Route and cities list cannot be empty")
        
    if len(route) != len(cities):
        raise ValueError("Route length must match number of cities")
        
    total_distance = 0.0
    
    # Calculate distance between consecutive cities
    for i in range(len(route)):
        # Get current city coordinates
        current_city = cities[route[i]]
        current_point = (current_city[1], current_city[2])  # (latitude, longitude)
        
        # Get next city coordinates (wrap around to first city)
        next_city = cities[route[(i + 1) % len(route)]]
        next_point = (next_city[1], next_city[2])  # (latitude, longitude)
        
        # Add distance to total
        total_distance += calculate_distance(current_point, next_point)
    
    return total_distance

def get_coordinates(city):
    pass

def validate_route(route):
    pass

def load_cities(file_path: str) -> List[Tuple[str, float, float]]:
    """
    Load and parse cities.csv file containing city coordinates.
    
    Args:
        file_path (str): Path to the cities.csv file
        
    Returns:
        List[Tuple[str, float, float]]: List of tuples containing (city_name, latitude, longitude)
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Verify required columns exist
        required_columns = ['city', 'latitude', 'longitude']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV file must contain columns: {required_columns}")
        
        # Convert DataFrame to list of tuples
        cities = list(zip(df['city'], df['latitude'], df['longitude']))
        
        return cities
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find cities file at: {file_path}")
    except pd.errors.EmptyDataError:
        raise ValueError("The cities file is empty")
    except Exception as e:
        raise Exception(f"Error loading cities: {str(e)}") 