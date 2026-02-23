from util.define import *
from util.util import *
import numpy as np
from util.result import ContinuousResult
from function.continuous_function import ContinuousProblem

def firefly_continuous(
    problem: ContinuousProblem,
    population_size: int = 50,
    generation: int = 100,
    alpha: float = 0.5,
    beta0: float = 1.0,
    gamma: float = 1.0,
    rng_seed: int | None = None
) -> ContinuousResult:
    """
    Implements the Firefly Algorithm (FA) for continuous optimization problems.

    Args:
        problem: The continuous optimization problem to solve.
        population_size: The number of fireflies in the population.
        generation: The number of iterations to run the algorithm.
        alpha: Randomization parameter for the random movement component.
        beta0: Attractiveness at distance r=0.
        gamma: Light absorption coefficient, controlling how attractiveness decreases with distance.
        rng_seed: Seed for the random number generator for reproducibility.
    """
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if generation <= 0:
        raise ValueError("generation must be positive")

    rng_wrapper = RNGWrapper(rng_seed)
    lower_bound = problem.lower_bound if problem.lower_bound is not None else -100
    upper_bound = problem.upper_bound if problem.upper_bound is not None else 100

    timer = TimerWrapper()
    timer.start()

    population: list[FloatVector] = [rng_wrapper.uniform(lower_bound, upper_bound, size=problem.dimension) for _ in range(population_size)]
    fitness = np.array([problem.evaluate(p) for p in population]) 

    history_x: list[list[FloatVector]] = [[p.copy() for p in population]]
    history_value: list[list[Float]] = [list(fitness)]

    for _ in range(generation):
        for i in range(population_size):
            for j in range(population_size):
                # If firefly j is brighter than firefly i, move i towards j
                if fitness[j] < fitness[i]:
                    # Calculate distance and attractiveness
                    r_squared = np.sum((population[i] - population[j])**2)
                    beta = beta0 * np.exp(-gamma * r_squared)
                    attractiveness = beta * (population[j] - population[i])
                    epsilon = rng_wrapper.rng.uniform(-0.5, 0.5, size=problem.dimension)
                    
                    # Update position
                    new_solution = population[i] + attractiveness + alpha * epsilon
                    new_solution = np.clip(new_solution, lower_bound, upper_bound)
                    
                    new_fitness = problem.evaluate(new_solution)
                    
                    # update straight into if better
                    if new_fitness < fitness[i]:
                        population[i] = new_solution
                        fitness[i] = new_fitness
        
        # Find the best firefly and move it randomly, then evaluate and update if better
        best_idx = np.argmin(fitness)
        best_firefly = population[best_idx]
        
        epsilon = rng_wrapper.rng.uniform(-0.5, 0.5, size=problem.dimension)
        new_solution = best_firefly + alpha * epsilon
        new_solution = np.clip(new_solution, lower_bound, upper_bound)
        new_fitness = problem.evaluate(new_solution)

        if new_fitness < fitness[best_idx]:
            population[best_idx] = new_solution
            fitness[best_idx] = new_fitness

        history_x.append([p.copy() for p in population])
        history_value.append(list(fitness))

    total_time = timer.stop()
    history_info: list[str | None] = [None] * len(history_x)
    best_x, best_value = get_min_2d(history_x, history_value)

    return ContinuousResult(
        algorithm="Firefly Algorithm",
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