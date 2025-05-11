import streamlit as st
import sys
import os
import time
import pandas as pd
from typing import List, Tuple
from streamlit_folium import st_folium
import folium

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.genetic_algorithm import GeneticAlgorithm
from src.route_utils import load_cities, calculate_total_distance
from src.visualization import save_route_plot, save_fitness_plot
from src.real_distance import get_distance_matrix
from src.analytics import calculate_analytics

def calculate_route_distance(route, distance_matrix):
    total = 0
    for i in range(len(route)):
        total += distance_matrix[route[i]][route[(i+1)%len(route)]]
    return total

def main():
    st.title("Route Optimization using Genetic Algorithm")
    st.write("""
    Upload a CSV file with columns: `city,latitude,longitude` or use the example dataset. You can also add cities interactively on the map below. Adjust the parameters and click 'Optimize Route' to find the shortest path.
    """)
    
    # Sidebar for parameters
    st.sidebar.header("Parameters")
    
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
        city_name = f"City {len(st.session_state['map_cities'])+1}"
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
        cities = [(city['name'], city['lat'], city['lon']) for city in st.session_state['map_cities']]
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
            cities = load_cities('data/cities.csv')
            st.sidebar.success(f"Successfully loaded {len(cities)} cities!")
        except Exception as e:
            st.error(f"Error loading cities: {str(e)}")
    elif use_example:
        try:
            cities = load_cities('data/cities.csv')
            st.sidebar.info(f"Example dataset loaded with {len(cities)} cities.")
        except Exception as e:
            st.error("Example dataset not found or invalid.")
    else:
        # Try to load default dataset (for backward compatibility)
        try:
            cities = load_cities('data/cities.csv')
            st.sidebar.info(f"Using default dataset with {len(cities)} cities.")
        except Exception as e:
            st.warning("No valid dataset found. Please upload a CSV file or use the example dataset.")

    if cities:
        st.write("### Route Optimization")
        if st.button("Optimize Route"):
            with st.spinner("Optimizing route..."):
                ga = GeneticAlgorithm(
                    mutation_rate=mutation_rate,
                    tournament_size=3
                )
                start_time = time.time()
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
                st.success("Optimization complete!")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Distance", f"{best_distance:.2f} km")
                with col2:
                    st.metric("Time Taken", f"{time_taken:.2f} seconds")
                # --- Analytics Calculation and Display ---
                analytics = calculate_analytics(
                    total_distance=best_distance,
                    avg_speed=avg_speed,
                    fuel_efficiency=fuel_efficiency,
                    fuel_price=fuel_price
                )
                st.write("### Route Analytics")
                st.write(f"**Total Distance:** {analytics['total_distance']:.2f} km")
                st.write(f"**Estimated Time:** {analytics['estimated_time']:.2f} hours")
                st.write(f"**Fuel Consumption:** {analytics['fuel_consumption']:.2f} liters")
                st.write(f"**Estimated Cost:** ${analytics['estimated_cost']:.2f}")
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
                st.write(" → ".join(route_cities))
    else:
        st.info("Please upload a CSV file with city data (columns: city, latitude, longitude), use the example dataset, or add cities on the map above.")

if __name__ == "__main__":
    main() 