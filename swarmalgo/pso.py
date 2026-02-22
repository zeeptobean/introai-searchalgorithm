from util.define import *
from util.util import *
import numpy as np

def pso_continuous(
    problem: ContinuousProblem,
    population_size: int = 50,
    generation: int = 100,
    inertia_weight: float = 0.5,
    cognitive_coeff: float = 1.5,
    social_coeff: float = 1.5,
    rng_seed: int | None = None
) -> ContinuousResult:
    """
    Particle Swarm Optimization (PSO) for continuous optimization problems.

    Args:
        problem: The continuous optimization problem to solve.
        population_size: The number of particles in the swarm.
        generation: The number of iterations to run the algorithm.
        inertia_weight (w): Controls the momentum of the particle.
        cognitive_coeff (c1): Weight for the particle's personal best position.
        social_coeff (c2): Weight for the swarm's global best position.
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

    # Initialize population (particles)
    population: list[FloatVector] = [rng_wrapper.uniform(lower_bound, upper_bound, size=problem.dimension) for _ in range(population_size)]
    velocities: list[FloatVector] = [np.zeros(problem.dimension) for _ in range(population_size)]
    fitness = np.array([problem.evaluate(p) for p in population])

    # Initialize personal and global bests
    personal_best_positions = [p.copy() for p in population]
    personal_best_fitness = fitness.copy()
    
    global_best_idx = np.argmin(fitness)
    global_best_position = population[global_best_idx].copy()
    global_best_fitness = fitness[global_best_idx]

    history_x: list[list[FloatVector]] = [[p.copy() for p in population]]
    history_value: list[list[Float]] = [list(fitness)]
    history_info: list[str | None] = [None]

    for _ in range(generation):
        for i in range(population_size):
            # Update velocity
            r1, r2 = rng_wrapper.random(), rng_wrapper.random()
            cognitive_component = cognitive_coeff * r1 * (personal_best_positions[i] - population[i])
            social_component = social_coeff * r2 * (global_best_position - population[i])
            velocities[i] = inertia_weight * velocities[i] + cognitive_component + social_component

            # Update position
            population[i] = np.clip(population[i] + velocities[i], lower_bound, upper_bound)
            
            # Evaluate fitness
            fitness[i] = problem.evaluate(population[i])

            # Update personal best
            if fitness[i] < personal_best_fitness[i]:
                personal_best_fitness[i] = fitness[i]
                personal_best_positions[i] = population[i].copy()

        # Update global best
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < global_best_fitness:
            global_best_fitness = fitness[current_best_idx]
            global_best_position = population[current_best_idx].copy()

        history_x.append([p.copy() for p in population])
        history_value.append(list(fitness))
        history_info.append(f"Global best: {global_best_fitness:.4f} at {global_best_position}")

    total_time = timer.stop()

    return ContinuousResult(
        algorithm="Particle Swarm Optimization",
        objective_function=repr(problem),
        time=total_time,
        last_x=population,
        last_value=list(fitness),
        best_x=global_best_position,
        best_value=global_best_fitness,
        iterations=generation,
        rng_seed=rng_wrapper.get_seed(),
        history_x=history_x,
        history_value=history_value,
        history_info=history_info
    )