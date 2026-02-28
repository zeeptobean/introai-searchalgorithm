from util.define import *
from util.util import *
import numpy as np
from util.result import ContinuousResult, DiscreteResult
from function.continuous_function import ContinuousProblem
from function.discrete_function import DiscreteProblem

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
        algorithm="Teaching-Learning-Based Optimization",
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
    Teaching-Learning-Based Optimization (TLBO) for discrete optimization problems.
    
    TLBO simulates the teaching-learning process in a classroom:
    - Teacher Phase: Students learn from the best student (teacher)
    - Learner Phase: Students learn from each other through interaction
    
    For discrete problems, we adapt TLBO by:
    - Using neighbor function for generating new solutions
    - Using Hamming distance to measure difference between solutions
    - Using probabilistic blending for non-permutation problems

    Args:
        problem: The discrete optimization problem to solve.
        population_size: The number of learners in the class.
        generation: The number of iterations to run the algorithm.
        step_size: Base step size for neighbor generation.
        rng_seed: Seed for the random number generator for reproducibility.
    """
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if generation <= 0:
        raise ValueError("generation must be positive")

    rng_wrapper = RNGWrapper(rng_seed)

    timer = TimerWrapper()
    timer.start()

    population: list[FloatVector] = [problem.random_solution(rng_wrapper) for _ in range(population_size)]
    fitness = np.array([problem.evaluate(p) for p in population])

    is_permutation = (len(np.unique(population[0])) == len(population[0]) and 
                        np.allclose(np.sort(population[0]), np.arange(len(population[0]))))

    history = HistoryEntry(is_max_value_problem=problem.is_max_value_problem())

    history.add([p.copy() for p in population], list(fitness))

    for gen in range(generation):
        teacher_idx = np.argmin(fitness)
        teacher = population[teacher_idx].copy()
        
        if is_permutation:
            mean = teacher.copy()
        else:
            mean = np.zeros(problem.dimension)
            for d in range(problem.dimension):
                values, counts = np.unique([population[i][d] for i in range(population_size)], return_counts=True)
                mean[d] = values[np.argmax(counts)]
        
        teaching_factor = rng_wrapper.rng.integers(1, 2, endpoint=True)
        
        for i in range(population_size):
            r = rng_wrapper.random()
            
            move_probability = r  
            
            if is_permutation:
                hamming_dist = np.sum(population[i] != teacher)
                if hamming_dist > 0:
                    effective_step = max(1, min(int(step_size * teaching_factor * move_probability), 
                                                int(hamming_dist / 2)))
                    new_solution = problem.neighbor(population[i], effective_step, rng_wrapper)
                else:
                    new_solution = population[i].copy()
            else:
                blend_mask = rng_wrapper.rng.random(problem.dimension) < move_probability
                
                if teaching_factor == 2:
                    new_solution = np.where(blend_mask, teacher, population[i])
                else:
                    towards_teacher = rng_wrapper.rng.random(problem.dimension) < 0.7
                    new_solution = np.where(blend_mask & towards_teacher, teacher, population[i])
            
            new_fitness = problem.evaluate(new_solution)
            
            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness

        for i in range(population_size):
            j_idx = rng_wrapper.rng.choice([idx for idx in range(population_size) if idx != i])
            
            r = rng_wrapper.random()
            
            if is_permutation:
                if fitness[i] < fitness[j_idx]:
                    effective_step = max(1, int(step_size * r))
                else:
                    effective_step = max(1, int(step_size * r * 1.5))
                
                new_solution = problem.neighbor(population[i], effective_step, rng_wrapper)
            else:
                learning_prob = r
                
                if fitness[i] < fitness[j_idx]:
                    blend_mask = rng_wrapper.rng.random(problem.dimension) < learning_prob * 0.3
                    new_solution = np.where(blend_mask, population[j_idx], population[i])
                else:
                    blend_mask = rng_wrapper.rng.random(problem.dimension) < learning_prob * 0.6
                    new_solution = np.where(blend_mask, population[j_idx], population[i])
            
            new_fitness = problem.evaluate(new_solution)
            
            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness

        history.add([p.copy() for p in population], list(fitness))

    total_time = timer.stop()
    best_x, best_value = history.get_best_value()

    return DiscreteResult(
        algorithm="Teaching-Learning-Based Optimization",
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