from util.define import *
from util.util import *
import numpy as np
from util.result import ContinuousResult
from function.continuous_function import ContinuousProblem

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

    timer = TimerWrapper()
    timer.start()

    # Initialize population
    population: list[FloatVector] = [rng_wrapper.uniform(lower_bound, upper_bound, size=problem.dimension) for _ in range(population_size)]
    fitness = np.array([problem.evaluate(p) for p in population])

    history_x: list[list[FloatVector]] = [[p.copy() for p in population]]
    history_value: list[list[Float]] = [list(fitness)]

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

        history_x.append([p.copy() for p in population])
        history_value.append(list(fitness))

    total_time = timer.stop()
    history_info: list[str | None] = [None] * len(history_x)
    best_x, best_value = get_min_2d(history_x, history_value)

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
        history_x=history_x,
        history_value=history_value,
        history_info=history_info
    )