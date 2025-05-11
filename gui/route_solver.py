from src.genetic_algorithm import GeneticAlgorithm
from src.route_utils import calculate_total_distance
from typing import List, Tuple

# This module handles route optimization logic for both intra-cluster and inter-cluster TSP.

def solve_intra_cluster_routes(clustered_cities: List[List[Tuple[str, float, float]]], pop_size: int, generations: int):
    """
    Solve TSP within each cluster using the Genetic Algorithm.
    Returns a list of intra-cluster routes and their distances.
    """
    ga = GeneticAlgorithm()
    intra_routes = []
    intra_distances = []
    for cluster in clustered_cities:
        if len(cluster) == 1:
            intra_routes.append([0])
            intra_distances.append(0)
        else:
            route, dist, _ = ga.optimize(
                cities=cluster,
                pop_size=pop_size,
                generations=generations,
                fitness_func=calculate_total_distance
            )
            intra_routes.append(route)
            intra_distances.append(dist)
    return intra_routes, intra_distances

def solve_inter_cluster_route(cluster_centers: List[Tuple[str, float, float]], pop_size: int, generations: int):
    """
    Solve TSP between cluster centroids using the Genetic Algorithm.
    Returns the best inter-cluster route and its distance.
    """
    ga = GeneticAlgorithm()
    route, dist, _ = ga.optimize(
        cities=cluster_centers,
        pop_size=pop_size,
        generations=generations,
        fitness_func=calculate_total_distance
    )
    return route, dist

def combine_cluster_routes(inter_route, intra_routes, clustered_cities, all_cities):
    """
    Combine intra-cluster routes in the order given by the inter-cluster route.
    Returns the full route as a list of indices into the all_cities list.
    """
    full_route = []
    for cluster_idx in inter_route:
        cluster = clustered_cities[cluster_idx]
        route = intra_routes[cluster_idx]
        global_indices = [all_cities.index(cluster[i]) for i in route]
        full_route.extend(global_indices)
    return full_route 