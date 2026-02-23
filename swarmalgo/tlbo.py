from util.define import *
from util.util import *
import numpy as np
from util.result import ContinuousResult
from function.continuous_function import ContinuousProblem

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

    timer = TimerWrapper()
    timer.start()

    # Initialize population (learners)
    population: list[FloatVector] = [rng_wrapper.uniform(lower_bound, upper_bound, size=problem.dimension) for _ in range(population_size)]
    fitness = np.array([problem.evaluate(p) for p in population])

    history_x: list[list[FloatVector]] = [[p.copy() for p in population]]
    history_value: list[list[Float]] = [list(fitness)]

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

        history_x.append([p.copy() for p in population])
        history_value.append(list(fitness))

    total_time = timer.stop()
    history_info: list[str | None] = [None] * len(history_x)
    best_x, best_value = get_min_2d(history_x, history_value)

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
        history_x=history_x,
        history_value=history_value,
        history_info=history_info
    )