import streamlit as st
import sys
import os
import time
import pandas as pd
from typing import List, Tuple
from streamlit_folium import st_folium
import folium
from geopy.geocoders import Nominatim
from sklearn.cluster import KMeans
import random
from scipy.spatial import ConvexHull
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.genetic_algorithm import GeneticAlgorithm
from src.route_utils import load_cities, calculate_total_distance
from src.visualization import save_route_plot, save_fitness_plot
from src.real_distance import get_distance_matrix, get_route_geometry
from src.analytics import calculate_analytics
from gui.city_io import load_cities_from_csv, load_cities_from_session
from gui.clustering import cluster_cities
from gui.route_solver import solve_intra_cluster_routes, solve_inter_cluster_route, combine_cluster_routes

def calculate_route_distance(route, distance_matrix):
    total = 0
    for i in range(len(route)):
        total += distance_matrix[route[i]][route[(i+1)%len(route)]]
    return total

def reverse_geocode(lat, lon):
    geolocator = Nominatim(user_agent="route_optimizer_app")
    try:
        location = geolocator.reverse((lat, lon), language='en', addressdetails=True, timeout=10)
        if location and 'address' in location.raw:
            address = location.raw['address']
            for key in ['city', 'town', 'village', 'hamlet', 'state', 'county', 'country']:
                if key in address:
                    return address[key]
        # If no name found, return coordinates
        return f"({lat:.4f}, {lon:.4f})"
    except Exception:
        return f"({lat:.4f}, {lon:.4f})"

def main():
    st.title("Route Optimizer")
    st.write("""
    Upload a CSV file with columns: `city,latitude,longitude` or use the example dataset. You can also add cities interactively on the map below. Adjust the parameters and click 'Optimize Route' to find the shortest path.
    """)
    
    # Sidebar for parameters
    st.sidebar.header("Parameters")
    
    # Clustering toggle
    use_clustering = st.sidebar.checkbox("Cluster cities before optimization (ML + GA)", value=False)
    n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=6, value=3) if use_clustering else None
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload Cities CSV", type=['csv'])
    use_example = st.sidebar.button("Use Example Dataset")
    
    # API Key input (pre-filled for demo)
    api_key = st.sidebar.text_input("OpenRouteService API Key", value="5b3ce3597851110001cf6248eba90df2cf024c91bf92b1e39b06212b")
    
    # GA Parameters
    st.sidebar.subheader("Genetic Algorithm Parameters")
    pop_size = st.sidebar.slider(
        "Population Size",
        min_value=50,
        max_value=500,
        value=100,
        step=50
    )
    mutation_rate = st.sidebar.slider(
        "Mutation Rate",
        min_value=0.01,
        max_value=0.5,
        value=0.1,
        step=0.01
    )
    generations = st.sidebar.slider(
        "Number of Generations",
        min_value=50,
        max_value=500,
        value=100,
        step=50
    )

    # Analytics Parameters
    st.sidebar.subheader("Analytics Parameters")
    avg_speed = st.sidebar.number_input("Average Speed (km/h)", min_value=10, max_value=200, value=60)
    fuel_efficiency = st.sidebar.number_input("Fuel Efficiency (km/l)", min_value=1.0, max_value=50.0, value=15.0)
    fuel_price = st.sidebar.number_input("Fuel Price ($/l)", min_value=0.1, max_value=10.0, value=1.0)

    # --- Interactive Map for City Editing ---
    if 'map_cities' not in st.session_state:
        st.session_state['map_cities'] = []

    st.write("### Add Cities on Map")
    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)  # Centered on USA
    # Add existing markers
    for city in st.session_state['map_cities']:
        folium.Marker([city['lat'], city['lon']], popup=city['name']).add_to(m)
    # Add map to Streamlit
    map_data = st_folium(m, width=700, height=500)
    # Add city on click
    if map_data and map_data['last_clicked']:
        lat = map_data['last_clicked']['lat']
        lon = map_data['last_clicked']['lng']
        city_name = reverse_geocode(lat, lon)
        # Prevent duplicate points
        if not any(abs(city['lat']-lat)<1e-6 and abs(city['lon']-lon)<1e-6 for city in st.session_state['map_cities']):
            st.session_state['map_cities'].append({'name': city_name, 'lat': lat, 'lon': lon})
    # Show current city list
    st.write("#### Current Cities")
    for i, city in enumerate(st.session_state['map_cities']):
        st.write(f"{i+1}. {city['name']} ({city['lat']:.4f}, {city['lon']:.4f})")
    if st.button("Clear All Cities"):
        st.session_state['map_cities'] = []

    # --- Load cities from uploaded file, example, or default ---
    cities = None
    use_real_distance = False
    distance_matrix = None
    if st.session_state['map_cities']:
        # Use cities from map if present
        cities = load_cities_from_session(st.session_state['map_cities'])
        st.info(f"Using {len(cities)} cities from the interactive map.")
        # Prepare coordinates for ORS (lon, lat)
        coords = [[city[2], city[1]] for city in cities]
        if api_key:
            try:
                with st.spinner("Fetching real-world distance matrix from OpenRouteService..."):
                    distance_matrix = get_distance_matrix(coords, api_key)
                    use_real_distance = True
                    st.success("Fetched real-world distance matrix!")
            except Exception as e:
                st.warning(f"Could not fetch real-world distances: {e}. Falling back to geodesic distance.")
    elif uploaded_file is not None and not use_example:
        with open('data/cities.csv', 'wb') as f:
            f.write(uploaded_file.getvalue())
        try:
            cities = load_cities_from_csv('data/cities.csv')
            st.sidebar.success(f"Successfully loaded {len(cities)} cities!")
        except Exception as e:
            st.error(f"Error loading cities: {str(e)}")
    elif use_example:
        try:
            cities = load_cities_from_csv('data/cities.csv')
            st.sidebar.info(f"Example dataset loaded with {len(cities)} cities.")
        except Exception as e:
            st.error("Example dataset not found or invalid.")
    else:
        # Try to load default dataset (for backward compatibility)
        try:
            cities = load_cities_from_csv('data/cities.csv')
            st.sidebar.info(f"Using default dataset with {len(cities)} cities.")
        except Exception as e:
            st.warning("No valid dataset found. Please upload a CSV file or use the example dataset.")

    if cities:
        st.write("### Route Optimization")
        clustered_cities = None
        cluster_labels = None
        cluster_centers = None
        if use_clustering and len(cities) >= n_clusters:
            # Prepare data for clustering
            cluster_labels, cluster_centers = cluster_cities(cities, n_clusters)
            # Group cities by cluster
            clustered_cities = [[] for _ in range(n_clusters)]
            for idx, label in enumerate(cluster_labels):
                clustered_cities[label].append(cities[idx])
            st.info(f"Cities clustered into {n_clusters} groups. Each cluster will be optimized separately, then clusters will be connected.")
        if st.button("Optimize Route"):
            with st.spinner("Optimizing route..."):
                ga = GeneticAlgorithm(
                    mutation_rate=mutation_rate,
                    tournament_size=3
                )
                start_time = time.time()
                if use_clustering and clustered_cities is not None and len(clustered_cities) == n_clusters:
                    # 1. Solve TSP within each cluster
                    intra_routes, intra_distances = solve_intra_cluster_routes(clustered_cities, pop_size, generations)
                    intra_analytics = [calculate_analytics(dist, avg_speed, fuel_efficiency, fuel_price) for dist in intra_distances]
                    # 2. Solve TSP between cluster centroids
                    centroids = [(f"Cluster {i+1}", cluster_centers[i][0], cluster_centers[i][1]) for i in range(n_clusters)]
                    inter_route, inter_dist = solve_inter_cluster_route(centroids, pop_size, generations)
                    # 3. Combine intra-cluster routes in inter-cluster order
                    full_route = combine_cluster_routes(inter_route, intra_routes, clustered_cities, cities)
                    # 4. Calculate total distance for the full route
                    if use_real_distance and distance_matrix is not None:
                        best_distance = calculate_route_distance(full_route, distance_matrix)
                    else:
                        best_distance = calculate_total_distance(full_route, cities)
                    best_route = full_route
                    performance_history = []  # Not meaningful for combined route
                    time_taken = time.time() - start_time
                    save_route_plot(best_route, cities)
                    save_fitness_plot(performance_history)
                    analytics = calculate_analytics(
                        total_distance=best_distance,
                        avg_speed=avg_speed,
                        fuel_efficiency=fuel_efficiency,
                        fuel_price=fuel_price
                    )
                    # Also run non-clustered GA for comparison
                    best_route_nc, best_distance_nc, _ = ga.optimize(
                        cities=cities,
                        pop_size=pop_size,
                        generations=generations,
                        fitness_func=calculate_total_distance
                    )
                    analytics_nc = calculate_analytics(
                        total_distance=best_distance_nc,
                        avg_speed=avg_speed,
                        fuel_efficiency=fuel_efficiency,
                        fuel_price=fuel_price
                    )
                    # Store results in session state
                    st.session_state['best_route'] = best_route
                    st.session_state['best_distance'] = best_distance
                    st.session_state['performance_history'] = performance_history
                    st.session_state['cities'] = cities
                    st.session_state['analytics'] = analytics
                    st.session_state['time_taken'] = time_taken
                    st.session_state['route_just_optimized'] = True
                    st.session_state['cluster_labels'] = cluster_labels
                    st.session_state['intra_analytics'] = intra_analytics
                    st.session_state['best_route_nc'] = best_route_nc
                    st.session_state['best_distance_nc'] = best_distance_nc
                    st.session_state['analytics_nc'] = analytics_nc
                else:
                    if use_real_distance and distance_matrix is not None:
                        best_route, best_distance, performance_history = ga.optimize(
                            cities=cities,
                            pop_size=pop_size,
                            generations=generations,
                            fitness_func=lambda route, _: calculate_route_distance(route, distance_matrix)
                        )
                    else:
                        best_route, best_distance, performance_history = ga.optimize(
                            cities=cities,
                            pop_size=pop_size,
                            generations=generations,
                            fitness_func=calculate_total_distance
                        )
                    time_taken = time.time() - start_time
                    save_route_plot(best_route, cities)
                    save_fitness_plot(performance_history)
                    analytics = calculate_analytics(
                        total_distance=best_distance,
                        avg_speed=avg_speed,
                        fuel_efficiency=fuel_efficiency,
                        fuel_price=fuel_price
                    )
                    # Store results in session state
                    st.session_state['best_route'] = best_route
                    st.session_state['best_distance'] = best_distance
                    st.session_state['performance_history'] = performance_history
                    st.session_state['cities'] = cities
                    st.session_state['analytics'] = analytics
                    st.session_state['time_taken'] = time_taken
                    st.session_state['route_just_optimized'] = True
                    st.session_state['cluster_labels'] = None
                    st.session_state['intra_analytics'] = None
                    st.session_state['best_route_nc'] = None
                    st.session_state['best_distance_nc'] = None
                    st.session_state['analytics_nc'] = None

        # Show results if available in session state
        if 'best_route' in st.session_state and st.session_state['best_route'] is not None:
            best_route = st.session_state['best_route']
            best_distance = st.session_state['best_distance']
            performance_history = st.session_state['performance_history']
            cities = st.session_state['cities']
            analytics = st.session_state['analytics']
            time_taken = st.session_state.get('time_taken', None)
            cluster_labels = st.session_state.get('cluster_labels', None)
            intra_analytics = st.session_state.get('intra_analytics', None)
            best_route_nc = st.session_state.get('best_route_nc', None)
            best_distance_nc = st.session_state.get('best_distance_nc', None)
            analytics_nc = st.session_state.get('analytics_nc', None)
            st.success("Optimization complete!")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Distance", f"{best_distance:.2f} km")
            with col2:
                if time_taken is not None:
                    st.metric("Time Taken", f"{time_taken:.2f} seconds")
            st.write("### Route Analytics")
            st.write(f"**Total Distance:** {analytics['total_distance']:.2f} km")
            st.write(f"**Estimated Time:** {analytics['estimated_time']:.2f} hours")
            st.write(f"**Fuel Consumption:** {analytics['fuel_consumption']:.2f} liters")
            st.write(f"**Estimated Cost:** ${analytics['estimated_cost']:.2f}")
            # Show analytics per cluster if available
            if cluster_labels is not None and intra_analytics is not None:
                st.write("### Analytics Per Cluster")
                table_md = "| Cluster | Distance (km) | Time (h) | Fuel (l) | Cost ($) |\n"
                table_md += "|---------|--------------|----------|----------|----------|\n"
                for i, a in enumerate(intra_analytics):
                    table_md += f"| {i+1} | {a['total_distance']:.2f} | {a['estimated_time']:.2f} | {a['fuel_consumption']:.2f} | {a['estimated_cost']:.2f} |\n"
                st.markdown(table_md)
            # Show comparison with non-clustered GA
            if cluster_labels is not None and best_distance_nc is not None and analytics_nc is not None:
                st.write("### Clustered vs. Non-Clustered Comparison")
                comp_md = "| Method | Distance (km) | Time (h) | Fuel (l) | Cost ($) |\n"
                comp_md += "|--------|--------------|----------|----------|----------|\n"
                comp_md += f"| Clustered | {analytics['total_distance']:.2f} | {analytics['estimated_time']:.2f} | {analytics['fuel_consumption']:.2f} | {analytics['estimated_cost']:.2f} |\n"
                comp_md += f"| Non-Clustered | {analytics_nc['total_distance']:.2f} | {analytics_nc['estimated_time']:.2f} | {analytics_nc['fuel_consumption']:.2f} | {analytics_nc['estimated_cost']:.2f} |\n"
                st.markdown(comp_md)
            st.write("### Interactive Route Map")
            # Create a new Folium map centered on the mean location
            mean_lat = sum(city[1] for city in cities) / len(cities)
            mean_lon = sum(city[2] for city in cities) / len(cities)
            route_map = folium.Map(location=[mean_lat, mean_lon], zoom_start=5)

            # Draw all possible connections (network)
            for i in range(len(cities)):
                for j in range(i + 1, len(cities)):
                    folium.PolyLine(
                        locations=[(cities[i][1], cities[i][2]), (cities[j][1], cities[j][2])],
                        color='gray',
                        weight=1,
                        opacity=0.3
                    ).add_to(route_map)

            # Draw clusters with different colors and boundaries if clustering is enabled
            cluster_colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightblue', 'darkgreen']
            if use_clustering and cluster_labels is not None:
                for cidx in range(n_clusters):
                    cluster_points = np.array([[city[1], city[2]] for idx, city in enumerate(cities) if cluster_labels[idx] == cidx])
                    color = cluster_colors[cidx % len(cluster_colors)]
                    # Draw convex hull (boundary) if enough points
                    if len(cluster_points) >= 3:
                        hull = ConvexHull(cluster_points)
                        hull_points = cluster_points[hull.vertices]
                        folium.Polygon(locations=[(lat, lon) for lat, lon in hull_points], color=color, fill=True, fill_opacity=0.1).add_to(route_map)
                for idx, city in enumerate(cities):
                    color = cluster_colors[cluster_labels[idx] % len(cluster_colors)]
                    folium.CircleMarker(
                        location=(city[1], city[2]),
                        radius=8,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.7,
                        popup=f"{city[0]} (Cluster {cluster_labels[idx]+1})"
                    ).add_to(route_map)
            else:
                # Highlight cities in the optimized route
                for idx, i in enumerate(best_route):
                    folium.CircleMarker(
                        location=(cities[i][1], cities[i][2]),
                        radius=8,
                        color='blue',
                        fill=True,
                        fill_color='yellow',
                        fill_opacity=0.9,
                        popup=f"{cities[i][0]} (Stop {idx+1})"
                    ).add_to(route_map)

            # Draw the optimized route (real driving route if possible)
            if use_clustering and best_route_nc is not None:
                # Draw non-clustered route for comparison (dashed blue)
                route_coords_nc = [(cities[i][1], cities[i][2]) for i in best_route_nc] + [(cities[best_route_nc[0]][1], cities[best_route_nc[0]][2])]
                folium.PolyLine(
                    locations=route_coords_nc,
                    color='blue',
                    weight=3,
                    opacity=0.5,
                    dash_array='10,10',
                    tooltip='Non-Clustered Route'
                ).add_to(route_map)
            if use_real_distance and api_key:
                for idx in range(len(best_route)):
                    start_idx = best_route[idx]
                    end_idx = best_route[(idx + 1) % len(best_route)]
                    coord_from = [cities[start_idx][2], cities[start_idx][1]]  # [lon, lat]
                    coord_to = [cities[end_idx][2], cities[end_idx][1]]
                    try:
                        geometry = get_route_geometry(coord_from, coord_to, api_key)
                        folium.PolyLine(
                            locations=geometry,
                            color='red',
                            weight=5,
                            opacity=0.8
                        ).add_to(route_map)
                    except Exception as e:
                        # Fallback to straight line if API fails
                        folium.PolyLine(
                            locations=[(cities[start_idx][1], cities[start_idx][2]), (cities[end_idx][1], cities[end_idx][2])],
                            color='red',
                            weight=5,
                            opacity=0.8
                        ).add_to(route_map)
            else:
                # Fallback: straight lines
                route_coords = [(cities[i][1], cities[i][2]) for i in best_route] + [(cities[best_route[0]][1], cities[best_route[0]][2])]
                folium.PolyLine(
                    locations=route_coords,
                    color='red',
                    weight=5,
                    opacity=0.8
                ).add_to(route_map)

            # Add city names for all cities
            for city in cities:
                folium.Marker(
                    location=(city[1], city[2]),
                    popup=city[0],
                    icon=folium.Icon(color='gray', icon='info-sign')
                ).add_to(route_map)

            st_folium(route_map, width=700, height=500)
            st.write("### Optimized Route")
            st.image("results/best_route.png")
            with open("results/best_route.png", "rb") as f:
                st.download_button("Download Route Map", f, file_name="best_route.png")
            st.write("### Optimization Progress")
            st.image("results/fitness_plot.png")
            with open("results/fitness_plot.png", "rb") as f:
                st.download_button("Download Fitness Plot", f, file_name="fitness_plot.png")
            st.write("### Route Details")
            route_cities = [cities[i][0] for i in best_route]
            n = len(route_cities)
            # Build table header
            table_md = "| Stop | City Name | Previous City | Next City |\n"
            table_md += "|------|-----------|---------------|-----------|\n"
            for idx, city in enumerate(route_cities):
                prev_city = route_cities[idx-1] if idx > 0 else "-"
                next_city = route_cities[idx+1] if idx < n-1 else "-"
                stop_label = f"{idx+1}"
                if idx == 0:
                    stop_label += " (Start)"
                if idx == n-1:
                    stop_label += " (End)"
                table_md += f"| {stop_label} | {city} | {prev_city} | {next_city} |\n"
            st.markdown(table_md)
    else:
        st.info("Please upload a CSV file with city data (columns: city, latitude, longitude), use the example dataset, or add cities on the map above.")

if __name__ == "__main__":
    main() 