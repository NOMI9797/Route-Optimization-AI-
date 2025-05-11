import streamlit as st
import sys
import os
import time
import pandas as pd
from typing import List, Tuple

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.genetic_algorithm import GeneticAlgorithm
from src.route_utils import load_cities, calculate_total_distance
from src.visualization import save_route_plot, save_fitness_plot

def main():
    st.title("Route Optimization using Genetic Algorithm")
    st.write("""
    Upload a CSV file with columns: `city,latitude,longitude` or use the default dataset. Adjust the parameters and click 'Optimize Route' to find the shortest path.
    """)
    
    # Sidebar for parameters
    st.sidebar.header("Parameters")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload Cities CSV", type=['csv'])
    
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

    # Load cities from uploaded file or default
    cities = None
    if uploaded_file is not None:
        with open('data/cities.csv', 'wb') as f:
            f.write(uploaded_file.getvalue())
        try:
            cities = load_cities('data/cities.csv')
            st.sidebar.success(f"Successfully loaded {len(cities)} cities!")
        except Exception as e:
            st.error(f"Error loading cities: {str(e)}")
    else:
        # Try to load default dataset
        try:
            cities = load_cities('data/cities.csv')
            st.sidebar.info(f"Using default dataset with {len(cities)} cities.")
        except Exception as e:
            st.warning("No valid dataset found. Please upload a CSV file.")

    if cities:
        st.write("### Route Optimization")
        if st.button("Optimize Route"):
            with st.spinner("Optimizing route..."):
                ga = GeneticAlgorithm(
                    mutation_rate=mutation_rate,
                    tournament_size=3
                )
                start_time = time.time()
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
        st.info("Please upload a CSV file with city data (columns: city, latitude, longitude)")

if __name__ == "__main__":
    main() 