import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
import os

def plot_route(route: List[int], cities: List[Tuple[str, float, float]], save_path: str = None) -> None:
    """
    Plot the route on a map with points and arrows connecting cities.
    
    Args:
        route (List[int]): List of city indices representing the route
        cities (List[Tuple[str, float, float]]): List of cities with coordinates
        save_path (str, optional): Path to save the plot. If None, plot is displayed.
    """
    # Extract coordinates
    lats = [cities[i][1] for i in route]
    lons = [cities[i][2] for i in route]
    names = [cities[i][0] for i in route]
    
    # Create figure and axis
    plt.figure(figsize=(12, 8))
    
    # Plot cities as points
    plt.scatter(lons, lats, c='red', s=100, zorder=2)
    
    # Plot route lines
    plt.plot(lons, lats, 'b-', alpha=0.6, zorder=1)
    
    # Add arrows to show direction
    for i in range(len(route)-1):
        plt.annotate('', 
                    xy=(lons[i+1], lats[i+1]),
                    xytext=(lons[i], lats[i]),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2, alpha=0.6))
    
    # Add city names
    for i, name in enumerate(names):
        plt.annotate(name, 
                    (lons[i], lats[i]),
                    xytext=(5, 5),
                    textcoords='offset points')
    
    # Add title and labels
    plt.title('Optimized Route')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save or show plot
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_fitness_curve(scores: List[float], save_path: str = None) -> None:
    """
    Plot the fitness curve showing improvement over generations.
    
    Args:
        scores (List[float]): List of best fitness scores per generation
        save_path (str, optional): Path to save the plot. If None, plot is displayed.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot fitness curve
    plt.plot(scores, 'b-', label='Best Fitness')
    
    # Add moving average for smoother visualization
    window_size = min(10, len(scores))
    if window_size > 1:
        moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(scores)), moving_avg, 'r--', 
                label=f'{window_size}-Generation Moving Average')
    
    # Add title and labels
    plt.title('Fitness Curve Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Total Distance (km)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save or show plot
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def save_route_plot(route: List[int], cities: List[Tuple[str, float, float]], filename: str = 'best_route.png') -> None:
    """
    Save the route plot to the results directory.
    
    Args:
        route (List[int]): List of city indices representing the route
        cities (List[Tuple[str, float, float]]): List of cities with coordinates
        filename (str): Name of the file to save
    """
    save_path = os.path.join('results', filename)
    plot_route(route, cities, save_path)

def save_fitness_plot(scores: List[float], filename: str = 'fitness_plot.png') -> None:
    """
    Save the fitness curve plot to the results directory.
    
    Args:
        scores (List[float]): List of best fitness scores per generation
        filename (str): Name of the file to save
    """
    save_path = os.path.join('results', filename)
    plot_fitness_curve(scores, save_path) 