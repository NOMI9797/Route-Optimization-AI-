import numpy as np
from typing import List, Tuple, Callable
import random

class GeneticAlgorithm:
    def __init__(self, mutation_rate: float = 0.1, tournament_size: int = 3):
        """
        Initialize the Genetic Algorithm with parameters.
        
        Args:
            mutation_rate (float): Probability of mutation (0 to 1)
            tournament_size (int): Number of individuals in tournament selection
        """
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size

    def generate_initial_population(self, pop_size: int, city_list: List[Tuple[str, float, float]]) -> List[List[int]]:
        """
        Generate initial population of random routes.
        
        Args:
            pop_size (int): Size of the population (number of routes to generate)
            city_list (List[Tuple[str, float, float]]): List of cities with their coordinates
            
        Returns:
            List[List[int]]: List of routes, where each route is a list of indices representing city order
        """
        if not city_list:
            raise ValueError("City list cannot be empty")
        
        if pop_size <= 0:
            raise ValueError("Population size must be positive")
            
        num_cities = len(city_list)
        
        # Generate random permutations for each individual in the population
        population = []
        for _ in range(pop_size):
            # Create a random permutation of city indices
            route = list(np.random.permutation(num_cities))
            population.append(route)
            
        return population

    def initialize_population(self):
        pass

    def fitness_function(self):
        pass

    def select_parents(self, population: List[List[int]], fitness_scores: List[float]) -> Tuple[List[int], List[int]]:
        """
        Select two parents using tournament selection.
        
        Args:
            population (List[List[int]]): Current population of routes
            fitness_scores (List[float]): Fitness scores for each route (lower is better)
            
        Returns:
            Tuple[List[int], List[int]]: Two selected parent routes
        """
        def tournament_select():
            # Select random individuals for tournament
            tournament_indices = random.sample(range(len(population)), self.tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            
            # Return the best individual from tournament
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
            return population[winner_idx]
        
        # Select two parents
        parent1 = tournament_select()
        parent2 = tournament_select()
        
        return parent1, parent2

    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Perform Order Crossover (OX1) between two parents.
        
        Args:
            parent1 (List[int]): First parent route
            parent2 (List[int]): Second parent route
            
        Returns:
            Tuple[List[int], List[int]]: Two offspring routes
        """
        size = len(parent1)
        
        # Select random crossover points
        point1, point2 = sorted(random.sample(range(size), 2))
        
        def create_offspring(p1: List[int], p2: List[int]) -> List[int]:
            # Initialize offspring with -1
            offspring = [-1] * size
            
            # Copy segment from first parent
            offspring[point1:point2] = p1[point1:point2]
            
            # Fill remaining positions with elements from second parent
            remaining = [x for x in p2 if x not in offspring[point1:point2]]
            j = 0
            for i in range(size):
                if offspring[i] == -1:
                    offspring[i] = remaining[j]
                    j += 1
            
            return offspring
        
        # Create two offspring
        offspring1 = create_offspring(parent1, parent2)
        offspring2 = create_offspring(parent2, parent1)
        
        return offspring1, offspring2

    def mutate(self, route: List[int]) -> List[int]:
        """
        Perform mutation by randomly swapping two cities.
        
        Args:
            route (List[int]): Route to mutate
            
        Returns:
            List[int]: Mutated route
        """
        if random.random() < self.mutation_rate:
            # Select two random positions
            pos1, pos2 = random.sample(range(len(route)), 2)
            
            # Swap cities
            route[pos1], route[pos2] = route[pos2], route[pos1]
        
        return route

    def optimize(self, cities: List[Tuple[str, float, float]], 
                pop_size: int = 100, 
                generations: int = 100,
                fitness_func: Callable[[List[int], List[Tuple[str, float, float]]], float] = None) -> Tuple[List[int], float, List[float]]:
        """
        Run the genetic algorithm optimization.
        
        Args:
            cities (List[Tuple[str, float, float]]): List of cities with coordinates
            pop_size (int): Size of the population
            generations (int): Number of generations to run
            fitness_func (Callable): Function to calculate route fitness
            
        Returns:
            Tuple[List[int], float, List[float]]: 
                - Best route found
                - Best fitness score
                - List of best scores per generation
        """
        # Generate initial population
        population = self.generate_initial_population(pop_size, cities)
        best_route = None
        best_fitness = float('inf')
        best_scores = []  # Track best score for each generation
        
        for generation in range(generations):
            # Calculate fitness for all routes
            fitness_scores = [fitness_func(route, cities) for route in population]
            
            # Update best route if found
            min_fitness_idx = np.argmin(fitness_scores)
            current_best_fitness = fitness_scores[min_fitness_idx]
            
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_route = population[min_fitness_idx].copy()
            
            # Store best score for this generation
            best_scores.append(current_best_fitness)
            
            # Create new population
            new_population = [best_route]  # Elitism: keep best route
            
            # Generate rest of new population
            while len(new_population) < pop_size:
                # Select parents
                parent1, parent2 = self.select_parents(population, fitness_scores)
                
                # Create offspring
                offspring1, offspring2 = self.crossover(parent1, parent2)
                
                # Mutate offspring
                offspring1 = self.mutate(offspring1)
                offspring2 = self.mutate(offspring2)
                
                # Add to new population
                new_population.extend([offspring1, offspring2])
            
            # Update population
            population = new_population[:pop_size]  # Trim if we added too many
        
        return best_route, best_fitness, best_scores

    # ... existing methods ... 