from util.define import *
from util.util import *
import numpy as np
from util.result import ContinuousResult, DiscreteResult
from function.continuous_function import ContinuousProblem
from function.discrete_function import DiscreteProblem

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

    history = HistoryEntry()

    timer = TimerWrapper()
    timer.start()

    population: list[FloatVector] = [rng_wrapper.uniform(lower_bound, upper_bound, size=problem.dimension) for _ in range(population_size)]
    fitness = np.array([problem.evaluate(p) for p in population]) 

    history.add([p.copy() for p in population], list(fitness))

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

        history.add([p.copy() for p in population], list(fitness))

    total_time = timer.stop()
    best_x, best_value = history.get_best_value()

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
        history=history
    )

def firefly_discrete(
    problem: DiscreteProblem,
    population_size: int = 50,
    generation: int = 100,
    alpha: float = 0.5,
    beta0: float = 1.0,
    gamma: float = 1.0,
    step_size: Float = 1.0,
    rng_seed: int | None = None
) -> DiscreteResult:
    """
    Implements the Firefly Algorithm (FA) for discrete optimization problems.

    Args:
        problem: The discrete optimization problem to solve.
        population_size: The number of fireflies in the population.
        generation: The number of iterations to run the algorithm.
        alpha: Randomization parameter controlling random movement probability.
        beta0: Base attractiveness value.
        gamma: Light absorption coefficient, controlling how attractiveness decreases with distance.
        step_size: Step size for neighbor generation.
        rng_seed: Seed for the random number generator for reproducibility.
    """
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if generation <= 0:
        raise ValueError("generation must be positive")
    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be in [0, 1]")

    rng_wrapper = RNGWrapper(rng_seed)

    timer = TimerWrapper()
    timer.start()

    population: list[FloatVector] = [problem.random_solution(rng_wrapper) for _ in range(population_size)]
    fitness = np.array([problem.evaluate(p) for p in population])

    history = HistoryEntry(is_max_value_problem=problem.is_max_value_problem())

    history.add([p.copy() for p in population], list(fitness))

    is_permutation = (len(np.unique(population[0])) == len(population[0]) and 
                        np.allclose(np.sort(population[0]), np.arange(len(population[0]))))
    
    for gen in range(generation):
        for i in range(population_size):
            for j in range(population_size):
                if fitness[j] < fitness[i]:
                    hamming_distance = np.sum(population[i] != population[j])
                    
                    normalized_distance = hamming_distance / problem.dimension
                    beta = beta0 * np.exp(-gamma * normalized_distance)
                    
                    if rng_wrapper.random() < beta:
                        effective_step = max(1, int(step_size * beta))
                        new_solution = problem.neighbor(population[i], effective_step, rng_wrapper)
                        
                        if not is_permutation and rng_wrapper.random() < beta:
                            blend_mask = rng_wrapper.rng.random(problem.dimension) < beta
                            new_solution = np.where(blend_mask, population[j], new_solution)
                        
                        new_fitness = problem.evaluate(new_solution)
                        
                        if new_fitness < fitness[i]:
                            population[i] = new_solution
                            fitness[i] = new_fitness
                            break  
        
        best_idx = np.argmin(fitness)
        if rng_wrapper.random() < alpha:
            new_solution = problem.neighbor(population[best_idx], step_size, rng_wrapper)
            new_fitness = problem.evaluate(new_solution)
            
            if new_fitness < fitness[best_idx]:
                population[best_idx] = new_solution
                fitness[best_idx] = new_fitness
        
        for i in range(population_size):
            if rng_wrapper.random() < alpha * 0.1: 
                new_solution = problem.neighbor(population[i], step_size, rng_wrapper)
                new_fitness = problem.evaluate(new_solution)
                
                if new_fitness < fitness[i]:
                    population[i] = new_solution
                    fitness[i] = new_fitness

        history.add([p.copy() for p in population], list(fitness))

    total_time = timer.stop()
    best_x, best_value = history.get_best_value()

    return DiscreteResult(
        algorithm="Firefly Algorithm",
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