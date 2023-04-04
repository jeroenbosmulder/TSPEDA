import numpy as np
import random
from typing import List, Tuple

distances = np.array([
    [0, 1, 5, 6],
    [1, 0, 4, 8],
    [5, 4, 0, 2],
    [6, 8, 2, 0]
])

def tsp_objective_function(individual: np.ndarray) -> int:
    """
    Calculates the total cost of a TSP solution.
    
    Args:
        individual: A numpy array representing a TSP solution.
        
    Returns:
        The total cost of the TSP solution.
    """
    cost = 0
    for i in range(len(individual) - 1):
        cost += distances[individual[i], individual[i + 1]]
    cost += distances[individual[-1], individual[0]]
    return cost

def initialize_population(pop_size: int, n_cities: int) -> List[np.ndarray]:
    """
    Initializes a population of TSP solutions.
    
    Args:
        pop_size: The size of the population.
        n_cities: The number of cities in the TSP instance.
        
    Returns:
        A list of numpy arrays, each representing a TSP solution.
    """
    return [np.random.permutation(n_cities) for _ in range(pop_size)]

def selection(population: List[np.ndarray], objective_function: callable, n_elites: int) -> List[np.ndarray]:
    """
    Selects the elite solutions from the population based on their fitness.
    
    Args:
        population: A list of TSP solutions.
        objective_function: The objective function to evaluate the fitness of the solutions.
        n_elites: The number of elites to select from the population.
        
    Returns:
        A list of elite solutions.
    """
    fitness = [objective_function(individual) for individual in population]
    elites_idx = np.argsort(fitness)[:n_elites]
    return [population[idx] for idx in elites_idx]

def estimate_probabilities(elites: List[np.ndarray], n_cities: int) -> np.ndarray:
    """
    Estimates the probabilities for each city to be at each position based on the elites.
    
    Args:
        elites: A list of elite TSP solutions.
        n_cities: The number of cities in the TSP instance.
        
    Returns:
        A numpy array with the estimated probabilities.
    """
    city_position_counts = np.zeros((n_cities, n_cities))
    for elite in elites:
        for pos, city in enumerate(elite):
            city_position_counts[city, pos] += 1
    return city_position_counts / len(elites)

def sample_individual(probabilities: np.ndarray) -> np.ndarray:
    """
    Samples a new TSP solution based on the given probabilities.
    
    Args:
        probabilities: A numpy array with probabilities for each city to be at each position.
        
    Returns:
        A numpy array representing a new TSP solution.
    """
    individual = np.zeros_like(probabilities[:, 0], dtype=int)
    prob_copy = probabilities.copy()

    for pos in range(len(prob_copy)):
        remaining_indices = np.where(prob_copy[:, pos] != 0)[0]
        if remaining_indices.size > 0:
            prob_copy[:, pos][remaining_indices] = 1 / len(remaining_indices)
            prob_copy[:, pos] /= prob_copy[:, pos].sum()
        else:
            prob_copy[:, pos] = 1 / len(prob_copy)

        individual[pos] = np.random.choice(len(prob_copy), p=prob_copy[:, pos])
        prob_copy[individual[pos], :] = 0

    return individual

def explore_exploit(probabilities: np.ndarray, exploration_factor: float) -> np.ndarray:
    """
    Apply an exploration-exploitation mechanism to the given probabilities.

    Parameters:
    ----------
    probabilities: np.ndarray
        The probability matrix.
    exploration_factor: float
        The exploration factor that controls the balance between exploration and exploitation.

    Returns:
    -------
    np.ndarray
        The updated probability matrix after applying the exploration-exploitation mechanism.
    """
    new_probabilities = probabilities.copy()
    for row in new_probabilities:
        row += exploration_factor / len(row)
    new_probabilities /= new_probabilities.sum(axis=1, keepdims=True)
    return new_probabilities

def eda_tsp(distances: np.ndarray, pop_size: int, n_elites: int, max_iter: int,
            exploration_initial: float, exploration_final: float) -> Tuple[np.ndarray, float]:
    """
    Solve the Traveling Salesman Problem using an Estimation of Distribution Algorithm.

    Parameters:
    ----------
    distances: np.ndarray
        A square matrix containing the pairwise distances between cities.
    pop_size: int
        The population size.
    n_elites: int
        The number of elites selected in each iteration.
    max_iter: int
        The maximum number of iterations.
    exploration_initial: float
        The initial exploration factor.
    exploration_final: float
        The final exploration factor.

    Returns:
    -------
    Tuple[np.ndarray, float]
        The best solution found and its cost.
    """
    n_cities = len(distances)
    population = initialize_population(pop_size, n_cities)
    best_solution = None
    best_fitness = float("inf")

    for i in range(max_iter):
        elites = selection(population, tsp_objective_function, n_elites)
        probabilities = estimate_probabilities(elites, n_cities)

        exploration_factor = exploration_initial * (exploration_final / exploration_initial) ** (i / (max_iter - 1))
        probabilities_explore_exploit = explore_exploit(probabilities, exploration_factor)

        population = [sample_individual(probabilities_explore_exploit) for _ in range(pop_size)]

        current_best_solution = min(population, key=tsp_objective_function)
        current_best_fitness = tsp_objective_function(current_best_solution)
        if current_best_fitness < best_fitness:
            best_solution = current_best_solution
            best_fitness = current_best_fitness

    return best_solution, best_fitness

pop_size = 100
n_elites = 10
max_iter = 100
exploration_initial = 1.0
exploration_final = 0.01

solution, cost = eda_tsp(distances, pop_size, n_elites, max_iter, exploration_initial, exploration_final)
print("Best solution:", solution)
print("Cost:", cost)
