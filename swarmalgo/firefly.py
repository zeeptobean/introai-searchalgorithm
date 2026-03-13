from util.define import *
from util.util import *
import numpy as np
from util.result import ContinuousResult, DiscreteResult
from function.continuous_function import ContinuousProblem
from function.discrete_function import DiscreteProblem, TSPFunction, KnapsackFunction, GraphColoringFunction

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

def firefly_discrete_tsp(
    problem: TSPFunction,
    population_size: int = 50,
    generation: int = 100,
    alpha: float = 0.5,
    beta0: float = 1.0,
    gamma: float = 1.0,
    rng_seed: int | None = None
) -> DiscreteResult:
    """
    Firefly Algorithm for TSP problems.
    Movement based on Hamming distance and swap/insert operators.
    
    Args:
        problem: The TSP problem to solve.
        population_size: The number of fireflies in the population.
        generation: The number of iterations to run the algorithm.
        alpha: Randomization parameter for random movement.
        beta0: Base attractiveness at distance 0.
        gamma: Light absorption coefficient.
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

    for gen in range(generation):
        for i in range(population_size):
            for j in range(population_size):
                # If firefly j is brighter (better fitness) than firefly i
                if fitness[j] < fitness[i]:
                    # Calculate Hamming distance
                    hamming_distance = np.sum(population[i] != population[j])
                    
                    # Calculate attractiveness based on Hamming distance
                    normalized_distance = hamming_distance / problem.dimension
                    beta = beta0 * np.exp(-gamma * normalized_distance)
                    
                    # Move i towards j with probability based on attractiveness
                    if rng_wrapper.random() < beta:
                        # Number of move operations based on Hamming distance and attractiveness
                        num_ops = max(1, int(hamming_distance * beta / 3))
                        
                        new_solution = population[i].copy()
                        
                        # Apply swap/insert operators to move towards j
                        for _ in range(num_ops):
                            # Find positions where i and j differ
                            diff_positions = np.where(new_solution != population[j])[0]
                            
                            if len(diff_positions) > 0:
                                # Select a position to fix
                                pos = rng_wrapper.rng.choice(diff_positions)
                                
                                # Find where the value that should be at pos (from j) currently is
                                target_value = population[j][pos]
                                current_pos = np.where(new_solution == target_value)[0][0]
                                
                                # Swap to move closer to j
                                new_solution[pos], new_solution[current_pos] = new_solution[current_pos], new_solution[pos]
                        
                        new_fitness = problem.evaluate(new_solution)
                        
                        if new_fitness < fitness[i]:
                            population[i] = new_solution
                            fitness[i] = new_fitness
                            break
        
        # Random movement for best firefly
        best_idx = np.argmin(fitness)
        if rng_wrapper.random() < alpha:
            # Random 2-opt swap
            indices = sorted(rng_wrapper.rng.choice(problem.dimension, size=2, replace=False))
            i_swap, j_swap = indices[0], indices[1]
            new_solution = population[best_idx].copy()
            new_solution[i_swap:j_swap+1] = new_solution[i_swap:j_swap+1][::-1]
            
            new_fitness = problem.evaluate(new_solution)
            
            if new_fitness < fitness[best_idx]:
                population[best_idx] = new_solution
                fitness[best_idx] = new_fitness

        history.add([p.copy() for p in population], list(fitness))

    total_time = timer.stop()
    best_x, best_value = history.get_best_value()

    return DiscreteResult(
        algorithm="Firefly-TSP",
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


def firefly_discrete_knapsack(
    problem: KnapsackFunction,
    population_size: int = 50,
    generation: int = 100,
    alpha: float = 0.5,
    beta0: float = 1.0,
    gamma: float = 1.0,
    rng_seed: int | None = None
) -> DiscreteResult:
    """
    Firefly Algorithm for Knapsack problems.
    Movement based on Hamming distance and bit flipping.
    
    Args:
        problem: The Knapsack problem to solve.
        population_size: The number of fireflies in the population.
        generation: The number of iterations to run the algorithm.
        alpha: Randomization parameter for random movement.
        beta0: Base attractiveness at distance 0.
        gamma: Light absorption coefficient.
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

    for gen in range(generation):
        for i in range(population_size):
            for j in range(population_size):
                # If firefly j is brighter (better fitness) than firefly i
                if fitness[j] < fitness[i]:
                    # Calculate Hamming distance
                    hamming_distance = np.sum(population[i] != population[j])
                    
                    # Calculate attractiveness based on Hamming distance
                    normalized_distance = hamming_distance / problem.dimension
                    beta = beta0 * np.exp(-gamma * normalized_distance)
                    
                    # Move i towards j by flipping bits with probability beta
                    new_solution = population[i].copy()
                    
                    # For each dimension where they differ, flip with probability beta
                    for d in range(problem.dimension):
                        if population[i][d] != population[j][d]:
                            if rng_wrapper.random() < beta:
                                # Flip bit to match brighter firefly
                                new_solution[d] = population[j][d]
                    
                    new_fitness = problem.evaluate(new_solution)
                    
                    if new_fitness < fitness[i]:
                        population[i] = new_solution
                        fitness[i] = new_fitness
                        break
        
        # Random movement for best firefly
        best_idx = np.argmin(fitness)
        if rng_wrapper.random() < alpha:
            # Random bit flip
            new_solution = population[best_idx].copy()
            flip_idx = rng_wrapper.rng.integers(0, problem.dimension)
            new_solution[flip_idx] = 1 - new_solution[flip_idx]
            
            new_fitness = problem.evaluate(new_solution)
            
            if new_fitness < fitness[best_idx]:
                population[best_idx] = new_solution
                fitness[best_idx] = new_fitness
        
        # Random exploration for all fireflies with small probability
        for i in range(population_size):
            if rng_wrapper.random() < alpha * 0.1:
                new_solution = population[i].copy()
                flip_idx = rng_wrapper.rng.integers(0, problem.dimension)
                new_solution[flip_idx] = 1 - new_solution[flip_idx]
                
                new_fitness = problem.evaluate(new_solution)
                
                if new_fitness < fitness[i]:
                    population[i] = new_solution
                    fitness[i] = new_fitness

        history.add([p.copy() for p in population], list(fitness))

    total_time = timer.stop()
    best_x, best_value = history.get_best_value()

    return DiscreteResult(
        algorithm="Firefly-Knapsack",
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


def firefly_discrete_graphcoloring(
    problem: GraphColoringFunction,
    population_size: int = 50,
    generation: int = 100,
    alpha: float = 0.5,
    beta0: float = 1.0,
    gamma: float = 1.0,
    rng_seed: int | None = None
) -> DiscreteResult:
    """
    Firefly Algorithm for Graph Coloring problems.
    Movement based on Hamming distance and color reassignment.
    
    Args:
        problem: The Graph Coloring problem to solve.
        population_size: The number of fireflies in the population.
        generation: The number of iterations to run the algorithm.
        alpha: Randomization parameter for random movement.
        beta0: Base attractiveness at distance 0.
        gamma: Light absorption coefficient.
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

    for gen in range(generation):
        for i in range(population_size):
            for j in range(population_size):
                # If firefly j is brighter (better fitness) than firefly i
                if fitness[j] < fitness[i]:
                    # Calculate Hamming distance
                    hamming_distance = np.sum(population[i] != population[j])
                    
                    # Calculate attractiveness based on Hamming distance
                    normalized_distance = hamming_distance / problem.dimension
                    beta = beta0 * np.exp(-gamma * normalized_distance)
                    
                    # Move i towards j by changing colors with probability beta
                    new_solution = population[i].copy()
                    
                    # For each vertex where colors differ, change to j's color with probability beta
                    for d in range(problem.dimension):
                        if population[i][d] != population[j][d]:
                            if rng_wrapper.random() < beta:
                                # Change color to match brighter firefly
                                new_solution[d] = population[j][d]
                    
                    new_fitness = problem.evaluate(new_solution)
                    
                    if new_fitness < fitness[i]:
                        population[i] = new_solution
                        fitness[i] = new_fitness
                        break
        
        # Random movement for best firefly
        best_idx = np.argmin(fitness)
        if rng_wrapper.random() < alpha:
            # Random color change
            new_solution = population[best_idx].copy()
            vertex = rng_wrapper.rng.integers(0, problem.dimension)
            max_color = int(np.max(new_solution))
            new_solution[vertex] = rng_wrapper.rng.integers(0, max_color + 1)
            
            new_fitness = problem.evaluate(new_solution)
            
            if new_fitness < fitness[best_idx]:
                population[best_idx] = new_solution
                fitness[best_idx] = new_fitness
        
        # Random exploration for all fireflies with small probability
        for i in range(population_size):
            if rng_wrapper.random() < alpha * 0.1:
                new_solution = population[i].copy()
                vertex = rng_wrapper.rng.integers(0, problem.dimension)
                max_color = int(np.max(population[i]))
                new_solution[vertex] = rng_wrapper.rng.integers(0, max_color + 1)
                
                new_fitness = problem.evaluate(new_solution)
                
                if new_fitness < fitness[i]:
                    population[i] = new_solution
                    fitness[i] = new_fitness

        history.add([p.copy() for p in population], list(fitness))

    total_time = timer.stop()
    best_x, best_value = history.get_best_value()

    return DiscreteResult(
        algorithm="Firefly-GraphColoring",
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
    Firefly Algorithm dispatcher for discrete optimization problems.
    Automatically selects the appropriate specialized implementation based on problem type.
    
    Movement is based on brightness (fitness) and Hamming distance:
    - TSPFunction: Uses swap/insert operators
    - KnapsackFunction: Uses bit flipping
    - GraphColoringFunction: Uses color reassignment
    - Generic: Uses neighbor function
    
    Args:
        problem: The discrete optimization problem to solve.
        population_size: The number of fireflies in the population.
        generation: The number of iterations to run the algorithm.
        alpha: Randomization parameter for random movement.
        beta0: Base attractiveness at distance 0.
        gamma: Light absorption coefficient.
        step_size: Step size for neighbor generation (only for generic problems).
        rng_seed: Seed for the random number generator for reproducibility.
    
    Returns:
        DiscreteResult containing the optimization results.
    """
    if isinstance(problem, TSPFunction):
        return firefly_discrete_tsp(problem, population_size, generation, alpha, beta0, gamma, rng_seed)
    elif isinstance(problem, KnapsackFunction):
        return firefly_discrete_knapsack(problem, population_size, generation, alpha, beta0, gamma, rng_seed)
    elif isinstance(problem, GraphColoringFunction):
        return firefly_discrete_graphcoloring(problem, population_size, generation, alpha, beta0, gamma, rng_seed)
    else:
        # Generic firefly for other discrete problems
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