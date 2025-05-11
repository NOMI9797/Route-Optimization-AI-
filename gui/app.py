import streamlit as st
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def main():
    st.title("Route Optimization using Genetic Algorithm")
    
    # Sidebar for parameters
    st.sidebar.header("Parameters")
    
    # Main content area
    st.write("Welcome to the Route Optimization Tool!")

if __name__ == "__main__":
    main() 