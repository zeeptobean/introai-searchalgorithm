from util.define import *
from util.util import *
import numpy as np
from util.result import ContinuousResult, DiscreteResult
from function.continuous_function import ContinuousProblem
from function.discrete_function import DiscreteProblem

def _tournament_selection(population_size: int, fitness: np.ndarray, k: int, rng: np.random.Generator) -> int:
    """Selects an individual using tournament selection."""
    candidates = rng.integers(0, population_size, size=k)
    return candidates[np.argmin(fitness[candidates])]

def _crossover(parent1: FloatVector, parent2: FloatVector, crossover_rate: float, alpha: float, rng: np.random.Generator) -> tuple[FloatVector, FloatVector]:
    """Performs blend crossover (BLX-alpha)."""
    if rng.random() > crossover_rate:
        return parent1.copy(), parent2.copy()

    d = np.abs(parent1 - parent2)
    lower = np.minimum(parent1, parent2) - alpha * d
    upper = np.maximum(parent1, parent2) + alpha * d
    
    child1 = rng.uniform(lower, upper, size=parent1.shape)
    child2 = rng.uniform(lower, upper, size=parent2.shape)
    
    return child1, child2

def _mutation(individual: npt.NDArray[np.float64], mutation_rate: float, mutation_strength: float, lower_bound: float, upper_bound: float, rng: np.random.Generator):
    """Performs Gaussian mutation."""
    mutation_mask = rng.random(individual.shape) < mutation_rate
    noise = rng.normal(0, mutation_strength, size=individual.shape)
    individual += noise * mutation_mask
    np.clip(individual, lower_bound, upper_bound, out=individual)

def genetic_algorithm_continuous(
    problem: ContinuousProblem,
    population_size: int = 100,
    generation: int = 100,
    tournament_k: int = 3,
    crossover_alpha: float = 0.5,
    crossover_rate: float = 0.9,
    mutation_strength: float = 0.1,
    mutation_rate: float = 0.1,
    rng_seed: int | None = None
) -> ContinuousResult:
    """
    Genetic Algorithm (GA) for continuous optimization problems.

    Args:
        problem: The continuous optimization problem to solve.
        population_size: The number of individuals in the population.
        generation: The number of generations to run.
        tournament_k: The number of individuals in each tournament for selection.
        crossover_alpha: The alpha parameter for blend crossover.
        crossover_rate: The probability of crossover.
        mutation_strength: The standard deviation of the Gaussian noise for mutation.
        mutation_rate: The probability of mutation for each gene.
        rng_seed: Seed for the random number generator.
    """
    rng_wrapper = RNGWrapper(rng_seed)
    lower_bound = problem.lower_bound if problem.lower_bound is not None else -100
    upper_bound = problem.upper_bound if problem.upper_bound is not None else 100

    history = HistoryEntry()

    timer = TimerWrapper()
    timer.start()

    # Initialize population
    population: list[FloatVector] = [rng_wrapper.uniform(lower_bound, upper_bound, size=problem.dimension) for _ in range(population_size)]
    fitness = np.array([problem.evaluate(p) for p in population])

    history.add([p.copy() for p in population], list(fitness))

    for _ in range(generation):
        # Selection
        selected_indices = [_tournament_selection(population_size, fitness, tournament_k, rng_wrapper.rng) for _ in range(population_size)]
        
        # Create next generation
        children: list[FloatVector] = []
        for i in range(0, population_size, 2):
            p1_idx, p2_idx = selected_indices[i], selected_indices[i+1]
            parent1, parent2 = population[p1_idx], population[p2_idx]
            
            # Crossover
            child1, child2 = _crossover(parent1, parent2, crossover_rate, crossover_alpha, rng_wrapper.rng)
            children.extend([child1, child2])

        # Mutation
        for child in children:
            _mutation(child, mutation_rate, mutation_strength, lower_bound, upper_bound, rng_wrapper.rng)

        # Evaluate new generation and replace old one
        population = children
        fitness = np.array([problem.evaluate(p) for p in population])

        history.add([p.copy() for p in population], list(fitness))

    total_time = timer.stop()
    best_x, best_value = history.get_best_value()

    return ContinuousResult(
        algorithm="Genetic Algorithm",
        problem=problem,
        time=total_time,
        last_x=population,
        last_value=list(fitness),
        best_x=best_x,
        best_value=best_value,
        iterations=generation,
        rng_seed=rng_wrapper.get_seed(),
        history=history
    )


# ==================== Discrete GA Functions ====================

def _order_crossover(parent1: FloatVector, parent2: FloatVector, rng: np.random.Generator) -> tuple[FloatVector, FloatVector]:
    """Order Crossover (OX) for permutation problems like TSP."""
    size = len(parent1)
    
    # Choose two crossover points
    point1, point2 = sorted(rng.integers(0, size, size=2))
    
    # Create children
    child1 = np.full(size, -1, dtype=float)
    child2 = np.full(size, -1, dtype=float)
    
    # Copy segment from parents
    child1[point1:point2] = parent1[point1:point2]
    child2[point1:point2] = parent2[point1:point2]
    
    # Fill remaining positions
    def fill_child(child, parent):
        parent_idx = point2
        child_idx = point2
        while -1 in child:
            if parent[parent_idx % size] not in child:
                child[child_idx % size] = parent[parent_idx % size]
                child_idx += 1
            parent_idx += 1
        return child
    
    child1 = fill_child(child1, parent2)
    child2 = fill_child(child2, parent1)
    
    return child1, child2


def _uniform_crossover(parent1: FloatVector, parent2: FloatVector, crossover_rate: float, rng: np.random.Generator) -> tuple[FloatVector, FloatVector]:
    """Uniform crossover for binary problems like Knapsack."""
    if rng.random() > crossover_rate:
        return parent1.copy(), parent2.copy()
    
    mask = rng.random(len(parent1)) < 0.5
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)
    
    return child1, child2


def genetic_algorithm_discrete(
    problem: DiscreteProblem,
    population_size: int = 100,
    generation: int = 100,
    tournament_k: int = 3,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.1,
    mutation_step: float = 1.0,
    rng_seed: int | None = None
) -> DiscreteResult:
    """
    Genetic Algorithm (GA) for discrete optimization problems.
    
    Uses problem-specific functions:
    - problem.random_solution() for initialization
    - problem.neighbor() for mutation (leverages existing neighbor functions)
    - Order Crossover for permutation problems (TSP)
    - Uniform Crossover for binary problems (Knapsack, Graph Coloring)

    Args:
        problem: The discrete optimization problem to solve.
        population_size: The number of individuals in the population.
        generation: The number of generations to run.
        tournament_k: The number of individuals in each tournament for selection.
        crossover_rate: The probability of crossover.
        mutation_rate: The probability of mutation.
        mutation_step: Step size for problem.neighbor() function.
        rng_seed: Seed for the random number generator.
    """
    if population_size < 2:
        raise ValueError("population_size must be at least 2")
    if population_size % 2 != 0:
        raise ValueError("population_size must be even")
    if generation <= 0:
        raise ValueError("generation must be positive")
    if tournament_k < 2 or tournament_k > population_size:
        raise ValueError("tournament_k must be between 2 and population_size")
    
    rng_wrapper = RNGWrapper(rng_seed)

    history = HistoryEntry(is_max_value_problem=problem.is_max_value_problem())
    
    timer = TimerWrapper()
    timer.start()

    population: list[FloatVector] = [problem.random_solution(rng_wrapper) for _ in range(population_size)]
    fitness = np.array([problem.evaluate(p) for p in population])

    sample = population[0].astype(int)
    is_permutation = len(np.unique(sample)) == len(sample)

    history.add([p.copy() for p in population], list(fitness))

    for _ in range(generation):
        # Selection - Tournament selection
        selected_indices = [_tournament_selection(population_size, fitness, tournament_k, rng_wrapper.rng) for _ in range(population_size)]
        
        # Create next generation
        children: list[FloatVector] = []
        for i in range(0, population_size, 2):
            p1_idx, p2_idx = selected_indices[i], selected_indices[i+1]
            parent1, parent2 = population[p1_idx], population[p2_idx]
            
            if is_permutation:
                if rng_wrapper.rng.random() < crossover_rate:
                    child1, child2 = _order_crossover(parent1, parent2, rng_wrapper.rng)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
            else:
                child1, child2 = _uniform_crossover(parent1, parent2, crossover_rate, rng_wrapper.rng)
            
            children.extend([child1, child2])

        # Mutation using problem.neighbor() function
        for i, child in enumerate(children):
            if rng_wrapper.rng.random() < mutation_rate:
                children[i] = problem.neighbor(child, mutation_step, rng_wrapper)

        # Evaluate new generation and replace old one
        population = children
        fitness = np.array([problem.evaluate(p) for p in population])

        history.add([p.copy() for p in population], list(fitness))

    total_time = timer.stop()
    best_x, best_value = history.get_best_value()

    return DiscreteResult(
        algorithm="Genetic Algorithm",
        problem=problem,
        time=total_time,
        last_x=population,
        last_value=list(fitness),
        best_x=best_x,
        best_value=best_value,
        iterations=generation,
        rng_seed=rng_wrapper.get_seed(),
        history=history
    )