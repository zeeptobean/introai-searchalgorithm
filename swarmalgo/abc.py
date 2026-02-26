from util.define import *
from util.util import *
import numpy as np
from util.result import ContinuousResult, DiscreteResult
from function.continuous_function import ContinuousProblem
from function.discrete_function import DiscreteProblem

def abc_continuous(
    problem: ContinuousProblem,
    population_size: int = 50,
    generation: int = 100,
    limit: int = 10,
    rng_seed: int | None = None,
    single_dimension_update: bool = False
) -> ContinuousResult:
    """
    Artificial Bee Colony (ABC) algorithm for continuous optimization problems, with whole dimension update.

    Args:
        problem: The continuous optimization problem to solve.
        population_size: The total number of bees (half employed, half onlookers).
        generation: The number of iterations to run the algorithm.
        limit: The number of trials after which a food source is abandoned.
        rng_seed: Seed for the random number generator for reproducibility.
        single_dimension_update: If True, update one dimension at a time (original ABC). If False, update all dimensions at once.
    """
    if population_size <= 0 or population_size % 2 != 0:
        raise ValueError("population_size must be a positive even number")
    if generation <= 0:
        raise ValueError("generation must be positive")
    if limit <= 0:
        raise ValueError("limit must be positive")

    rng_wrapper = RNGWrapper(rng_seed)
    lower_bound = problem.lower_bound if problem.lower_bound is not None else -100
    upper_bound = problem.upper_bound if problem.upper_bound is not None else 100

    timer = TimerWrapper()
    timer.start()

    num_employed = population_size // 2
    
    population: list[FloatVector] = [rng_wrapper.uniform(lower_bound, upper_bound, size=problem.dimension) for _ in range(num_employed)]
    fitness = np.array([problem.evaluate(p) for p in population])
    trials = np.zeros(num_employed)

    history_x: list[list[FloatVector]] = [[p.copy() for p in population]]
    history_value: list[list[Float]] = [list(fitness)]
    history_info: list[str | None] = [None] 

    for _ in range(generation):
        # Employed Bee Phase 
        for i in range(num_employed):
            # Select a random partner solution
            partner_idx = rng_wrapper.rng.choice([idx for idx in range(num_employed) if idx != i])
            partner = population[partner_idx]
            
            if single_dimension_update:
                # Update one dimension at a time (original ABC)
                j = rng_wrapper.rng.integers(0, problem.dimension)  # Randomly select a dimension
                phi = rng_wrapper.rng.uniform(-1, 1)
                new_solution = population[i].copy()
                new_solution[j] += phi * (population[i][j] - partner[j])
            else:
                phi = rng_wrapper.rng.uniform(-1, 1, size=problem.dimension)
                new_solution = population[i] + phi * (population[i] - partner)

            new_solution = np.clip(new_solution, lower_bound, upper_bound)
            new_fitness = problem.evaluate(new_solution)
            
            # Greedy selection
            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness
                trials[i] = 0
            else:
                trials[i] += 1

        # Onlooker Bee Phase
        # Taking inverse for minimization problems
        # transform fitness to avoid non-negative
        transformed_fitness = np.where(
            fitness >= 0,
            1.0 / (1.0 + fitness),
            1.0 + np.abs(fitness)
        )
        probabilities = 1.0 / (transformed_fitness + 1e-9) 
        probabilities /= np.sum(probabilities)

        for _ in range(num_employed): # Onlookers = employed bees
            # Select a food source based on probability (roulette wheel selection)
            i = rng_wrapper.rng.choice(num_employed, p=probabilities)
            
            # Select a random partner solution
            partner_idx = rng_wrapper.rng.choice([idx for idx in range(num_employed) if idx != i])
            partner = population[partner_idx]
            
            # Generate a new candidate solution
            phi = rng_wrapper.rng.uniform(-1, 1, size=problem.dimension)
            new_solution = population[i] + phi * (population[i] - partner)
            new_solution = np.clip(new_solution, lower_bound, upper_bound)
            new_fitness = problem.evaluate(new_solution)
            
            # Greedy selection
            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness
                trials[i] = 0
            else:
                trials[i] += 1

        # Scout Bee Phase
        # reset if trial exceeds limit
        for i in range(num_employed):
            if trials[i] > limit:
                population[i] = rng_wrapper.uniform(lower_bound, upper_bound, size=problem.dimension)
                fitness[i] = problem.evaluate(population[i])
                trials[i] = 0

        history_x.append([p.copy() for p in population])
        history_value.append(list(fitness))
        history_info.append(f"prob: {probabilities}, trials: {trials}")

    total_time = timer.stop()
    best_x, best_value = get_min_2d(history_x, history_value)

    return ContinuousResult(
        algorithm="Artificial Bee Colony",
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

def abc_discrete(
    problem: DiscreteProblem,
    population_size: int = 50,
    generation: int = 100,
    limit: int = 10,
    step_size: Float = 1.0,
    rng_seed: int | None = None
) -> DiscreteResult:
    """
    Artificial Bee Colony (ABC) algorithm for discrete optimization problems.

    Args:
        problem: The discrete optimization problem to solve.
        population_size: The total number of bees (half employed, half onlookers).
        generation: The number of iterations to run the algorithm.
        limit: The number of trials after which a food source is abandoned.
        step_size: The step size for neighborhood generation.
        rng_seed: Seed for the random number generator for reproducibility.
    """
    if population_size <= 0 or population_size % 2 != 0:
        raise ValueError("population_size must be a positive even number")
    if generation <= 0:
        raise ValueError("generation must be positive")
    if limit <= 0:
        raise ValueError("limit must be positive")

    rng_wrapper = RNGWrapper(rng_seed)

    timer = TimerWrapper()
    timer.start()

    num_employed = population_size // 2
    
    population: list[FloatVector] = [problem.random_solution(rng_wrapper) for _ in range(num_employed)]
    fitness = np.array([problem.evaluate(p) for p in population])
    trials = np.zeros(num_employed)

    history_x: list[list[FloatVector]] = [[p.copy() for p in population]]
    history_value: list[list[Float]] = [list(fitness)]
    history_info: list[str | None] = [None] 

    for gen in range(generation):
        # Employed Bee Phase 
        for i in range(num_employed):
            new_solution = problem.neighbor(population[i], step_size, rng_wrapper)
            new_fitness = problem.evaluate(new_solution)
            
            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness
                trials[i] = 0
            else:
                trials[i] += 1

        transformed_fitness = np.where(
            fitness >= 0,
            1.0 / (1.0 + fitness),
            1.0 + np.abs(fitness)
        )
        probabilities = transformed_fitness / np.sum(transformed_fitness)

        for _ in range(num_employed):
            i = rng_wrapper.rng.choice(num_employed, p=probabilities)
            
            new_solution = problem.neighbor(population[i], step_size, rng_wrapper)
            new_fitness = problem.evaluate(new_solution)
            
            # Greedy selection
            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness
                trials[i] = 0
            else:
                trials[i] += 1

        for i in range(num_employed):
            if trials[i] > limit:
                population[i] = problem.random_solution(rng_wrapper)
                fitness[i] = problem.evaluate(population[i])
                trials[i] = 0

        history_x.append([p.copy() for p in population])
        history_value.append(list(fitness))
        history_info.append(f"gen={gen+1}, trials={trials.tolist()}, abandoned={np.sum(trials > limit)}")

    total_time = timer.stop()
    best_x, best_value = get_min_2d(history_x, history_value)

    return DiscreteResult(
        algorithm="Artificial Bee Colony",
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