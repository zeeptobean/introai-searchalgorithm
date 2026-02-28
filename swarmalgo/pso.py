from util.define import *
from util.util import *
import numpy as np
from util.result import ContinuousResult, DiscreteResult
from function.continuous_function import ContinuousProblem
from function.discrete_function import DiscreteProblem

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

    history = HistoryEntry()

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

    history.add([p.copy() for p in population], list(fitness))

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

        history.add([p.copy() for p in population], list(fitness), f"Global best: {global_best_fitness:.4f} at {global_best_position}")

    total_time = timer.stop()

    return ContinuousResult(
        algorithm="Particle Swarm Optimization",
        problem=problem,
        time=total_time,
        last_x=population,
        last_value=list(fitness),
        best_x=global_best_position,
        best_value=global_best_fitness,
        iterations=generation,
        rng_seed=rng_wrapper.get_seed(),
        history=history
    )

def pso_discrete(
    problem: DiscreteProblem,
    population_size: int = 50,
    generation: int = 100,
    inertia_weight: float = 0.5,
    cognitive_coeff: float = 1.5,
    social_coeff: float = 1.5,
    step_size: Float = 1.0,
    rng_seed: int | None = None
) -> DiscreteResult:
    """
    Particle Swarm Optimization (PSO) for discrete optimization problems.
    
    For discrete problems, we adapt PSO by:
    - Using velocity as a probability measure for moving towards personal/global best
    - Using neighbor function to generate valid discrete solutions
    - Maintaining the core PSO philosophy of balancing exploration and exploitation

    Args:
        problem: The discrete optimization problem to solve.
        population_size: The number of particles in the swarm.
        generation: The number of iterations to run the algorithm.
        inertia_weight (w): Controls the momentum/exploration tendency.
        cognitive_coeff (c1): Weight for the particle's personal best influence.
        social_coeff (c2): Weight for the swarm's global best influence.
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

    personal_best_positions = [p.copy() for p in population]
    personal_best_fitness = fitness.copy()
    
    global_best_idx = np.argmin(fitness)
    global_best_position = population[global_best_idx].copy()
    global_best_fitness = fitness[global_best_idx]

    velocities: list[Float] = [rng_wrapper.random() * 0.5 for _ in range(population_size)]

    is_permutation = (len(np.unique(population[0])) == len(population[0]) and 
                        np.allclose(np.sort(population[0]), np.arange(len(population[0]))))

    history = HistoryEntry(is_max_value_problem=problem.is_max_value_problem())

    history.add([p.copy() for p in population], list(fitness))

    for gen in range(generation):
        for i in range(population_size):
            r1, r2 = rng_wrapper.random(), rng_wrapper.random()
            
            personal_distance = np.sum(population[i] != personal_best_positions[i]) / problem.dimension
            global_distance = np.sum(population[i] != global_best_position) / problem.dimension
            
            velocities[i] = (inertia_weight * velocities[i] + 
                           cognitive_coeff * r1 * personal_distance + 
                           social_coeff * r2 * global_distance)
            
            velocities[i] = np.clip(velocities[i], 0.0, 1.0)
            
            new_solution = population[i].copy()
            
            move_to_global = velocities[i] * (1 - inertia_weight) * social_coeff / (cognitive_coeff + social_coeff)
            move_to_personal = velocities[i] * (1 - inertia_weight) * cognitive_coeff / (cognitive_coeff + social_coeff)
            explore = velocities[i] * inertia_weight
            
            total_prob = move_to_global + move_to_personal + explore
            if total_prob > 0:
                move_to_global /= total_prob
                move_to_personal /= total_prob
                explore /= total_prob
            
            rand_val = rng_wrapper.random()
            
            if rand_val < move_to_global:
                if is_permutation:
                    effective_step = max(1, int(step_size * velocities[i]))
                    new_solution = problem.neighbor(population[i], effective_step, rng_wrapper)
                else:
                    blend_prob = 0.3 + 0.4 * (1 - velocities[i])  
                    blend_mask = rng_wrapper.rng.random(problem.dimension) < blend_prob
                    new_solution = np.where(blend_mask, global_best_position, population[i])
                    
            elif rand_val < move_to_global + move_to_personal:
                if is_permutation:
                    effective_step = max(1, int(step_size * velocities[i] * 0.7))
                    new_solution = problem.neighbor(population[i], effective_step, rng_wrapper)
                else:
                    blend_prob = 0.3 + 0.4 * (1 - velocities[i])
                    blend_mask = rng_wrapper.rng.random(problem.dimension) < blend_prob
                    new_solution = np.where(blend_mask, personal_best_positions[i], population[i])
            else:
                effective_step = max(1, int(step_size * (1 + velocities[i])))
                new_solution = problem.neighbor(population[i], effective_step, rng_wrapper)
            
            new_fitness = problem.evaluate(new_solution)
            
            population[i] = new_solution
            fitness[i] = new_fitness

            if fitness[i] < personal_best_fitness[i]:
                personal_best_fitness[i] = fitness[i]
                personal_best_positions[i] = population[i].copy()

        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < global_best_fitness:
            global_best_fitness = fitness[current_best_idx]
            global_best_position = population[current_best_idx].copy()

        history.add([p.copy() for p in population], list(fitness), f"gen={gen+1}, global_best={global_best_fitness:.4f}, avg_velocity={np.mean(velocities):.4f}")

    total_time = timer.stop()

    return DiscreteResult(
        algorithm="Particle Swarm Optimization",
        problem=problem,
        time=total_time,
        last_x=population,
        last_value=list(fitness),
        best_x=global_best_position,
        best_value=global_best_fitness,
        iterations=generation,
        rng_seed=rng_wrapper.get_seed(),
        history=history
    )