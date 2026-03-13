from util.define import *
from util.util import *
import numpy as np
from util.result import ContinuousResult, DiscreteResult
from function.continuous_function import ContinuousProblem
from function.discrete_function import DiscreteProblem, TSPFunction, KnapsackFunction, GraphColoringFunction

def tlbo_continuous(
    problem: ContinuousProblem,
    population_size: int = 50,
    generation: int = 100,
    rng_seed: int | None = None
) -> ContinuousResult:
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if generation <= 0:
        raise ValueError("generation must be positive")

    rng_wrapper = RNGWrapper(rng_seed)
    lower_bound = problem.lower_bound if problem.lower_bound is not None else -100
    upper_bound = problem.upper_bound if problem.upper_bound is not None else 100

    history = HistoryEntry(is_max_value_problem=False)

    timer = TimerWrapper()
    timer.start()

    # Initialize population (learners)
    population: list[FloatVector] = [rng_wrapper.uniform(lower_bound, upper_bound, size=problem.dimension) for _ in range(population_size)]
    fitness = np.array([problem.evaluate(p) for p in population])

    history.add([p.copy() for p in population], list(fitness))

    for _ in range(generation):
        # Teacher phase
        teacher_idx = np.argmin(fitness)
        teacher = population[teacher_idx]
        mean = np.mean(population, axis=0)
        
        # Teaching factor (TF) is either 1 or 2, chosen randomly
        teaching_factor = rng_wrapper.rng.integers(1, 2, endpoint=True)
        
        for i in range(population_size):
            r = rng_wrapper.random() # random [0, 1)
            
            # Update solution based on teacher
            new_solution = population[i] + r * (teacher - teaching_factor * mean)
            new_solution = np.clip(new_solution, lower_bound, upper_bound)
            
            new_fitness = problem.evaluate(new_solution)
            
            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness

        # --- Learner Phase ---
        for i in range(population_size):
            # Select another learner randomly
            j_idx = rng_wrapper.rng.choice([idx for idx in range(population_size) if idx != i])
            
            r = rng_wrapper.random()
            
            # Update based on interaction with another learner (can't use abs here as it lose directionality)
            if fitness[i] < fitness[j_idx]:
                new_solution = population[i] + r * (population[i] - population[j_idx])
            else:
                new_solution = population[i] + r * (population[j_idx] - population[i])
            
            new_solution = np.clip(new_solution, lower_bound, upper_bound)
            new_fitness = problem.evaluate(new_solution)
            
            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness

        history.add([p.copy() for p in population], list(fitness))

    total_time = timer.stop()
    best_x, best_value = history.get_best_value()

    return ContinuousResult(
        algorithm=f"Teaching-Learning-Based Optimization(NP={population_size}, gen={generation})",
        short_name="TLBO",
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

def tlbo_discrete_tsp(
    problem: TSPFunction,
    population_size: int = 50,
    generation: int = 100,
    step_size: Float = 1.0,
    rng_seed: int | None = None
) -> DiscreteResult:
    """
    Teaching-Learning-Based Optimization (TLBO) for TSP problems.
    Uses Insert/Swap operators to move towards the teacher or peer learner.
    
    Args:
        problem: The TSP problem to solve.
        population_size: The number of learners in the class.
        generation: The number of iterations to run the algorithm.
        step_size: Base step size for determining number of swaps.
        rng_seed: Seed for the random number generator for reproducibility.
    """
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if generation <= 0:
        raise ValueError("generation must be positive")

    rng_wrapper = RNGWrapper(rng_seed)
    timer = TimerWrapper()
    timer.start()

    # Initialize population
    population: list[FloatVector] = [problem.random_solution(rng_wrapper) for _ in range(population_size)]
    fitness = np.array([problem.evaluate(p) for p in population])

    history = HistoryEntry(is_max_value_problem=problem.is_max_value_problem())
    history.add([p.copy() for p in population], list(fitness))

    for gen in range(generation):
        # Teacher Phase
        teacher_idx = np.argmin(fitness)
        teacher = population[teacher_idx].copy()
        teaching_factor = rng_wrapper.rng.integers(1, 2, endpoint=True)
        
        for i in range(population_size):
            r = rng_wrapper.random()
            
            # Use Insert/Swap operators to move towards teacher
            # Count positions where current solution differs from teacher
            diff_positions = np.where(population[i] != teacher)[0]
            num_diffs = len(diff_positions)
            
            if num_diffs > 0:
                # Determine number of operations based on teaching factor and randomness
                num_ops = max(1, min(int(step_size * teaching_factor * r), num_diffs // 2))
                new_solution = population[i].copy()
                
                # Perform Insert/Swap operations to get closer to teacher
                for _ in range(num_ops):
                    # Choose random position that differs
                    diff_idx = rng_wrapper.rng.choice(len(diff_positions))
                    pos = diff_positions[diff_idx]
                    
                    # Find where the value that should be at pos (according to teacher) currently is
                    target_value = teacher[pos]
                    current_pos = np.where(new_solution == target_value)[0][0]
                    
                    # Swap or insert to move closer to teacher configuration
                    if rng_wrapper.random() < 0.5:
                        # Swap
                        new_solution[pos], new_solution[current_pos] = new_solution[current_pos], new_solution[pos]
                    else:
                        # Insert: extract and reinsert
                        value = new_solution[current_pos]
                        new_solution = np.delete(new_solution, current_pos)
                        new_solution = np.insert(new_solution, pos, value)
                
                new_fitness = problem.evaluate(new_solution)
                if new_fitness < fitness[i]:
                    population[i] = new_solution
                    fitness[i] = new_fitness

        # Learner Phase
        for i in range(population_size):
            j_idx = rng_wrapper.rng.choice([idx for idx in range(population_size) if idx != i])
            r = rng_wrapper.random()
            
            # Determine direction based on fitness
            if fitness[i] < fitness[j_idx]:
                # Current learner is better, smaller step
                num_ops = max(1, int(step_size * r))
            else:
                # Other learner is better, learn more from them
                num_ops = max(1, int(step_size * r * 1.5))
            
            # Use 2-opt style swaps for learner interaction
            new_solution = population[i].copy()
            for _ in range(num_ops):
                indices = sorted(rng_wrapper.rng.choice(problem.dimension, size=2, replace=False))
                i_swap, j_swap = indices[0], indices[1]
                new_solution[i_swap:j_swap+1] = new_solution[i_swap:j_swap+1][::-1]
            
            new_fitness = problem.evaluate(new_solution)
            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness

        history.add([p.copy() for p in population], list(fitness))

    total_time = timer.stop()
    best_x, best_value = history.get_best_value()

    return DiscreteResult(
        algorithm=f"Teaching-Learning-Based Optimization(NP={population_size}, gen={generation})",
        short_name="TLBO-TSP",
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


def tlbo_discrete_knapsack(
    problem: KnapsackFunction,
    population_size: int = 50,
    generation: int = 100,
    rng_seed: int | None = None
) -> DiscreteResult:
    """
    Teaching-Learning-Based Optimization (TLBO) for Knapsack problems.
    Uses rounding of continuous blend between solutions to generate binary selections.
    
    Args:
        problem: The Knapsack problem to solve.
        population_size: The number of learners in the class.
        generation: The number of iterations to run the algorithm.
        rng_seed: Seed for the random number generator for reproducibility.
    """
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if generation <= 0:
        raise ValueError("generation must be positive")

    rng_wrapper = RNGWrapper(rng_seed)
    timer = TimerWrapper()
    timer.start()

    # Initialize population
    population: list[FloatVector] = [problem.random_solution(rng_wrapper) for _ in range(population_size)]
    fitness = np.array([problem.evaluate(p) for p in population])

    history = HistoryEntry(is_max_value_problem=problem.is_max_value_problem())
    history.add([p.copy() for p in population], list(fitness))

    for gen in range(generation):
        # Teacher Phase
        teacher_idx = np.argmin(fitness)
        teacher = population[teacher_idx].copy()
        
        # Calculate mean as the mode of each dimension
        mean = np.zeros(problem.dimension)
        for d in range(problem.dimension):
            values, counts = np.unique([population[k][d] for k in range(population_size)], return_counts=True)
            mean[d] = values[np.argmax(counts)]
        
        teaching_factor = rng_wrapper.rng.integers(1, 2, endpoint=True)
        
        for i in range(population_size):
            r = rng_wrapper.random()
            
            # Create continuous blend towards teacher
            # new = current + r * (teacher - TF * mean)
            continuous_solution = population[i] + r * (teacher - teaching_factor * mean)
            
            # Round to binary {0, 1}
            # Use sigmoid-like probability for rounding
            probabilities = 1 / (1 + np.exp(-4 * (continuous_solution - 0.5)))
            new_solution = (rng_wrapper.rng.random(problem.dimension) < probabilities).astype(float)
            
            new_fitness = problem.evaluate(new_solution)
            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness

        # Learner Phase
        for i in range(population_size):
            j_idx = rng_wrapper.rng.choice([idx for idx in range(population_size) if idx != i])
            r = rng_wrapper.random()
            
            # Create continuous blend based on peer interaction
            if fitness[i] < fitness[j_idx]:
                # Current learner is better, move away from j
                continuous_solution = population[i] + r * (population[i] - population[j_idx])
            else:
                # Other learner is better, move towards j
                continuous_solution = population[i] + r * (population[j_idx] - population[i])
            
            # Round to binary
            probabilities = 1 / (1 + np.exp(-4 * (continuous_solution - 0.5)))
            new_solution = (rng_wrapper.rng.random(problem.dimension) < probabilities).astype(float)
            
            new_fitness = problem.evaluate(new_solution)
            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness

        history.add([p.copy() for p in population], list(fitness))

    total_time = timer.stop()
    best_x, best_value = history.get_best_value()

    return DiscreteResult(
        algorithm=f"Teaching-Learning-Based Optimization(NP={population_size}, gen={generation})",
        short_name="TLBO-KP",
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


def tlbo_discrete_graphcoloring(
    problem: GraphColoringFunction,
    population_size: int = 50,
    generation: int = 100,
    rng_seed: int | None = None
) -> DiscreteResult:
    """
    Teaching-Learning-Based Optimization (TLBO) for Graph Coloring problems.
    Uses rounding of continuous blend between solutions to generate integer colors.
    
    Args:
        problem: The Graph Coloring problem to solve.
        population_size: The number of learners in the class.
        generation: The number of iterations to run the algorithm.
        rng_seed: Seed for the random number generator for reproducibility.
    """
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if generation <= 0:
        raise ValueError("generation must be positive")

    rng_wrapper = RNGWrapper(rng_seed)
    timer = TimerWrapper()
    timer.start()

    # Initialize population
    population: list[FloatVector] = [problem.random_solution(rng_wrapper) for _ in range(population_size)]
    fitness = np.array([problem.evaluate(p) for p in population])

    history = HistoryEntry(is_max_value_problem=problem.is_max_value_problem())
    history.add([p.copy() for p in population], list(fitness))

    for gen in range(generation):
        # Teacher Phase
        teacher_idx = np.argmin(fitness)
        teacher = population[teacher_idx].copy()
        
        # Calculate mean as the mode of each dimension
        mean = np.zeros(problem.dimension)
        for d in range(problem.dimension):
            values, counts = np.unique([population[k][d] for k in range(population_size)], return_counts=True)
            mean[d] = values[np.argmax(counts)]
        
        teaching_factor = rng_wrapper.rng.integers(1, 2, endpoint=True)
        
        for i in range(population_size):
            r = rng_wrapper.random()
            
            # Create continuous blend towards teacher
            continuous_solution = population[i] + r * (teacher - teaching_factor * mean)
            
            # Round to nearest integer and clip to valid color range
            # Determine max color from teacher to avoid unnecessary colors
            max_color = int(np.max(teacher))
            new_solution = np.clip(np.round(continuous_solution), 0, max_color).astype(float)
            
            new_fitness = problem.evaluate(new_solution)
            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness

        # Learner Phase
        for i in range(population_size):
            j_idx = rng_wrapper.rng.choice([idx for idx in range(population_size) if idx != i])
            r = rng_wrapper.random()
            
            # Create continuous blend based on peer interaction
            if fitness[i] < fitness[j_idx]:
                # Current learner is better
                continuous_solution = population[i] + r * (population[i] - population[j_idx])
            else:
                # Other learner is better
                continuous_solution = population[i] + r * (population[j_idx] - population[i])
            
            # Round to integer colors
            max_color = int(np.max([np.max(population[i]), np.max(population[j_idx])]))
            new_solution = np.clip(np.round(continuous_solution), 0, max_color).astype(float)
            
            new_fitness = problem.evaluate(new_solution)
            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness

        history.add([p.copy() for p in population], list(fitness))

    total_time = timer.stop()
    best_x, best_value = history.get_best_value()

    return DiscreteResult(
        algorithm=f"Teaching-Learning-Based Optimization(NP={population_size}, gen={generation})",
        short_name="TLBO-GC",
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


def tlbo_discrete(
    problem: DiscreteProblem,
    population_size: int = 50,
    generation: int = 100,
    step_size: Float = 1.0,
    rng_seed: int | None = None
) -> DiscreteResult:
    """
    Teaching-Learning-Based Optimization (TLBO) dispatcher for discrete optimization problems.
    Automatically selects the appropriate specialized TLBO implementation based on problem type.
    
    Supported problem types:
    - TSPFunction: Uses Insert/Swap operators
    - KnapsackFunction: Uses rounding for binary selection
    - GraphColoringFunction: Uses rounding for integer colors
    
    Args:
        problem: The discrete optimization problem to solve.
        population_size: The number of learners in the class.
        generation: The number of iterations to run the algorithm.
        step_size: Base step size (only used for TSP).
        rng_seed: Seed for the random number generator for reproducibility.
    
    Returns:
        DiscreteResult containing the optimization results.
    
    Raises:
        ValueError: If problem type is not supported.
    """
    if isinstance(problem, TSPFunction):
        return tlbo_discrete_tsp(problem, population_size, generation, step_size, rng_seed)
    elif isinstance(problem, KnapsackFunction):
        return tlbo_discrete_knapsack(problem, population_size, generation, rng_seed)
    elif isinstance(problem, GraphColoringFunction):
        return tlbo_discrete_graphcoloring(problem, population_size, generation, rng_seed)
    else:
        raise ValueError(f"Unsupported problem type: {type(problem).__name__}. Supported types: TSPFunction, KnapsackFunction, GraphColoringFunction")