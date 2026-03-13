from util.define import *
from util.util import *
import numpy as np
from util.result import ContinuousResult, DiscreteResult
from function.continuous_function import ContinuousProblem
from function.discrete_function import DiscreteProblem, TSPFunction, KnapsackFunction, GraphColoringFunction

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
        algorithm=f"Particle Swarm Optimization(NP={population_size}, gen={generation}, w={inertia_weight}, c1={cognitive_coeff}, c2={social_coeff})",
        short_name="PSO",
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


def pso_discrete_tsp(
    problem: DiscreteProblem,
    population_size: int = 50,
    generation: int = 100,
    inertia_weight: float = 0.5,
    cognitive_coeff: float = 1.5,
    social_coeff: float = 1.5,
    rng_seed: int | None = None
) -> DiscreteResult:
    """
    Particle Swarm Optimization (PSO) for TSP using Swap Operators.
    
    This implementation uses swap sequences as velocity representation, where velocity
    is a list of swap operations that move the current solution toward personal/global best.
    
    Args:
        problem: The TSP problem to solve.
        population_size: The number of particles in the swarm.
        generation: The number of iterations to run the algorithm.
        inertia_weight (w): Controls the momentum (portion of previous velocity to keep).
        cognitive_coeff (c1): Weight for the particle's personal best influence.
        social_coeff (c2): Weight for the swarm's global best influence.
        rng_seed: Seed for the random number generator for reproducibility.
    """
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if generation <= 0:
        raise ValueError("generation must be positive")
    if not isinstance(problem, TSPFunction):
        raise ValueError("pso_discrete_tsp requires a TSPFunction problem instance")

    rng_wrapper = RNGWrapper(rng_seed)
    timer = TimerWrapper()
    timer.start()

    # Initialize population
    population: list[FloatVector] = [problem.random_solution(rng_wrapper) for _ in range(population_size)]
    fitness = np.array([problem.evaluate(p) for p in population])

    # Initialize personal and global bests
    personal_best_positions = [p.copy() for p in population]
    personal_best_fitness = fitness.copy()
    
    global_best_idx = np.argmin(fitness)
    global_best_position = population[global_best_idx].copy()
    global_best_fitness = fitness[global_best_idx]

    # Velocity is a list of swap operations for each particle
    velocities: list[list[tuple[int, int]]] = [[] for _ in range(population_size)]

    history = HistoryEntry(is_max_value_problem=problem.is_max_value_problem())
    history.add([p.copy() for p in population], list(fitness))

    for gen in range(generation):
        for i in range(population_size):
            r1, r2 = rng_wrapper.random(), rng_wrapper.random()
            
            # Construct swap sequence to move from current to personal best
            personal_swaps = get_swap_sequence(
                population[i], 
                personal_best_positions[i],
                rng_wrapper
            )
            
            # Construct swap sequence to move from current to global best
            global_swaps = get_swap_sequence(
                population[i],
                global_best_position,
                rng_wrapper
            )
            
            # Update velocity as combination of swap sequences
            # v = w*v + c1*r1*personal_swaps + c2*r2*global_swaps
            new_velocity: list[tuple[int, int]] = []
            
            # Keep some swaps from previous velocity (inertia)
            if velocities[i]:
                num_keep = int(len(velocities[i]) * inertia_weight)
                if num_keep > 0:
                    kept_swaps = rng_wrapper.rng.choice(
                        len(velocities[i]), 
                        size=min(num_keep, len(velocities[i])), 
                        replace=False
                    )
                    new_velocity.extend([velocities[i][idx] for idx in kept_swaps])
            
            # Add swaps toward personal best (cognitive component)
            if personal_swaps:
                num_personal = int(len(personal_swaps) * cognitive_coeff * r1)
                if num_personal > 0:
                    selected = rng_wrapper.rng.choice(
                        len(personal_swaps),
                        size=min(num_personal, len(personal_swaps)),
                        replace=False
                    )
                    new_velocity.extend([personal_swaps[idx] for idx in selected])
            
            # Add swaps toward global best (social component)
            if global_swaps:
                num_global = int(len(global_swaps) * social_coeff * r2)
                if num_global > 0:
                    selected = rng_wrapper.rng.choice(
                        len(global_swaps),
                        size=min(num_global, len(global_swaps)),
                        replace=False
                    )
                    new_velocity.extend([global_swaps[idx] for idx in selected])
            
            velocities[i] = new_velocity
            
            # Apply swap operators to update position
            new_solution = population[i].copy()
            for swap in velocities[i]:
                idx1, idx2 = swap
                new_solution[idx1], new_solution[idx2] = new_solution[idx2], new_solution[idx1]
            
            population[i] = new_solution
            
            # Evaluate new position
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

        # Track history
        avg_swaps = np.mean([len(v) for v in velocities])
        history.add(
            [p.copy() for p in population],
            list(fitness),
            f"gen={gen+1}, gbest={global_best_fitness:.4f}, avg_swaps={avg_swaps:.2f}"
        )

    total_time = timer.stop()
    best_x, best_value = history.get_best_value()

    return DiscreteResult(
        algorithm=f"Particle Swarm Optimization(NP={population_size}, gen={generation}, w={inertia_weight}, c1={cognitive_coeff}, c2={social_coeff})",
        short_name="PSO-TSP",
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


def pso_discrete_knapsack(
    problem: DiscreteProblem,
    population_size: int = 50,
    generation: int = 100,
    inertia_weight: float = 0.5,
    cognitive_coeff: float = 1.5,
    social_coeff: float = 1.5,
    rng_seed: int | None = None
) -> DiscreteResult:
    """
    Particle Swarm Optimization (PSO) for Knapsack using Binary PSO with Sigmoid.
    
    This implementation uses sigmoid function to convert continuous velocity into
    bit-flip probabilities for binary decision variables (item selected or not).
    
    Velocity update: v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
    Position update: x[i] = 1 if rand() < sigmoid(v[i]), else 0
    
    Args:
        problem: The Knapsack problem to solve.
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
    if not isinstance(problem, KnapsackFunction):
        raise ValueError("pso_discrete_knapsack requires a KnapsackFunction problem instance")

    rng_wrapper = RNGWrapper(rng_seed)
    timer = TimerWrapper()
    timer.start()

    # Initialize population
    population: list[FloatVector] = [problem.random_solution(rng_wrapper) for _ in range(population_size)]
    fitness = np.array([problem.evaluate(p) for p in population])

    # Initialize personal and global bests
    personal_best_positions = [p.copy() for p in population]
    personal_best_fitness = fitness.copy()
    
    global_best_idx = np.argmin(fitness)
    global_best_position = population[global_best_idx].copy()
    global_best_fitness = fitness[global_best_idx]

    # Binary PSO: velocity is real-valued vector, same dimension as solution
    velocities: list[FloatVector] = [
        rng_wrapper.rng.uniform(-4, 4, size=problem.dimension) 
        for _ in range(population_size)
    ]

    history = HistoryEntry(is_max_value_problem=problem.is_max_value_problem())
    history.add([p.copy() for p in population], list(fitness))

    for gen in range(generation):
        for i in range(population_size):
            r1, r2 = rng_wrapper.random(), rng_wrapper.random()
            
            # Binary PSO with Sigmoid Function
            # Update velocity: v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
            velocities[i] = (
                inertia_weight * velocities[i] +
                cognitive_coeff * r1 * (personal_best_positions[i] - population[i]) +
                social_coeff * r2 * (global_best_position - population[i])
            )
            
            # Limit velocity to prevent sigmoid saturation
            velocities[i] = np.clip(velocities[i], -6, 6)
            
            # Apply sigmoid function: S(v) = 1 / (1 + e^(-v))
            sigmoid = 1.0 / (1.0 + np.exp(-velocities[i]))
            
            # Update position using sigmoid as bit-flip probability
            new_solution = np.zeros(problem.dimension)
            for j in range(problem.dimension):
                if rng_wrapper.random() < sigmoid[j]:
                    new_solution[j] = 1.0
                else:
                    new_solution[j] = 0.0
            
            population[i] = new_solution
            
            # Evaluate new position
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

        # Track history
        avg_velocity = np.mean([np.mean(np.abs(v)) for v in velocities])
        history.add(
            [p.copy() for p in population], 
            list(fitness), 
            f"gen={gen+1}, gbest={-global_best_fitness:.4f}, avg_|v|={avg_velocity:.4f}"
        )

    total_time = timer.stop()
    best_x, best_value = history.get_best_value()

    return DiscreteResult(
        algorithm=f"Particle Swarm Optimization(NP={population_size}, gen={generation}, w={inertia_weight}, c1={cognitive_coeff}, c2={social_coeff})",
        short_name="PSO-KP",
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


def pso_discrete_graphcoloring(
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
    Particle Swarm Optimization (PSO) for Graph Coloring using Hybrid Approach.
    
    This implementation uses a hybrid approach combining:
    - Velocity-based probability to decide exploration vs exploitation
    - Solution blending to move toward personal/global best
    - Neighbor-based exploration for diversity
    
    Args:
        problem: The Graph Coloring problem to solve.
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
    if not isinstance(problem, GraphColoringFunction):
        raise ValueError("pso_discrete_graphcoloring requires a GraphColoringFunction problem instance")

    rng_wrapper = RNGWrapper(rng_seed)
    timer = TimerWrapper()
    timer.start()

    # Initialize population
    population: list[FloatVector] = [problem.random_solution(rng_wrapper) for _ in range(population_size)]
    fitness = np.array([problem.evaluate(p) for p in population])

    # Initialize personal and global bests
    personal_best_positions = [p.copy() for p in population]
    personal_best_fitness = fitness.copy()
    
    global_best_idx = np.argmin(fitness)
    global_best_position = population[global_best_idx].copy()
    global_best_fitness = fitness[global_best_idx]

    # General discrete: use simple scalar velocity
    velocities: list[Float] = [rng_wrapper.random() * 0.5 for _ in range(population_size)]

    history = HistoryEntry(is_max_value_problem=problem.is_max_value_problem())
    history.add([p.copy() for p in population], list(fitness))

    for gen in range(generation):
        for i in range(population_size):
            r1, r2 = rng_wrapper.random(), rng_wrapper.random()
            
            # Calculate distance to personal and global best
            personal_distance = np.sum(population[i] != personal_best_positions[i]) / problem.dimension
            global_distance = np.sum(population[i] != global_best_position) / problem.dimension
            
            # Update velocity based on distances
            velocities[i] = (
                inertia_weight * velocities[i] + 
                cognitive_coeff * r1 * personal_distance + 
                social_coeff * r2 * global_distance
            )
            
            velocities[i] = np.clip(velocities[i], 0.0, 1.0)
            
            # Decide movement strategy based on velocity
            rand_val = rng_wrapper.random()
            
            if rand_val < velocities[i] * 0.5:  # Move toward global best
                blend_prob = 0.3 + 0.4 * velocities[i]
                blend_mask = rng_wrapper.rng.random(problem.dimension) < blend_prob
                new_solution = np.where(blend_mask, global_best_position, population[i])
            elif rand_val < velocities[i]:  # Move toward personal best
                blend_prob = 0.3 + 0.4 * velocities[i]
                blend_mask = rng_wrapper.rng.random(problem.dimension) < blend_prob
                new_solution = np.where(blend_mask, personal_best_positions[i], population[i])
            else:  # Random exploration
                effective_step = max(1, int(step_size * (1 + velocities[i])))
                new_solution = problem.neighbor(population[i], effective_step, rng_wrapper)
            
            population[i] = new_solution
            
            # Evaluate new position
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

        # Track history
        avg_velocity = np.mean(velocities)
        num_colors = len(np.unique(global_best_position.astype(int)))
        history.add(
            [p.copy() for p in population],
            list(fitness),
            f"gen={gen+1}, gbest={global_best_fitness:.4f}, colors={num_colors}, avg_v={avg_velocity:.4f}"
        )

    total_time = timer.stop()
    best_x, best_value = history.get_best_value()

    return DiscreteResult(
        algorithm=f"Particle Swarm Optimization(NP={population_size}, gen={generation}, w={inertia_weight}, c1={cognitive_coeff}, c2={social_coeff})",
        short_name="PSO-GC",
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
    
    This is a dispatcher function that automatically selects the appropriate
    PSO variant based on the problem type:
    - TSPFunction -> pso_discrete_tsp (swap operators)
    - KnapsackFunction -> pso_discrete_knapsack (binary PSO with sigmoid)
    - GraphColoringFunction -> pso_discrete_graphcoloring (hybrid approach)
    
    Args:
        problem: The discrete optimization problem to solve.
        population_size: The number of particles in the swarm.
        generation: The number of iterations to run the algorithm.
        inertia_weight (w): Controls the momentum/exploration tendency.
        cognitive_coeff (c1): Weight for the particle's personal best influence.
        social_coeff (c2): Weight for the swarm's global best influence.
        step_size: Base step size for neighbor generation (used for graph coloring).
        rng_seed: Seed for the random number generator for reproducibility.
        
    Returns:
        DiscreteResult from the appropriate PSO variant.
    """
    # Dispatch to specialized PSO function based on problem type
    if isinstance(problem, TSPFunction):
        return pso_discrete_tsp(
            problem=problem,
            population_size=population_size,
            generation=generation,
            inertia_weight=inertia_weight,
            cognitive_coeff=cognitive_coeff,
            social_coeff=social_coeff,
            rng_seed=rng_seed
        )
    elif isinstance(problem, KnapsackFunction):
        return pso_discrete_knapsack(
            problem=problem,
            population_size=population_size,
            generation=generation,
            inertia_weight=inertia_weight,
            cognitive_coeff=cognitive_coeff,
            social_coeff=social_coeff,
            rng_seed=rng_seed
        )
    elif isinstance(problem, GraphColoringFunction):
        return pso_discrete_graphcoloring(
            problem=problem,
            population_size=population_size,
            generation=generation,
            inertia_weight=inertia_weight,
            cognitive_coeff=cognitive_coeff,
            social_coeff=social_coeff,
            step_size=step_size,
            rng_seed=rng_seed
        )
    else:
        # Fallback to graph coloring approach for unknown discrete problems
        return pso_discrete_graphcoloring(
            problem=problem,
            population_size=population_size,
            generation=generation,
            inertia_weight=inertia_weight,
            cognitive_coeff=cognitive_coeff,
            social_coeff=social_coeff,
            step_size=step_size,
            rng_seed=rng_seed
        )



def get_swap_sequence(
    current: FloatVector,
    target: FloatVector,
    rng_wrapper: RNGWrapper,
    max_swaps: int | None = None
) -> list[tuple[int, int]]:
    """
    Generate a sequence of swap operations to transform current solution toward target.
    
    This finds swaps that would move current closer to target in permutation space.
    
    Args:
        current: Current solution (permutation)
        target: Target solution (permutation)
        rng_wrapper: Random number generator wrapper
        max_swaps: Maximum number of swaps to return (default: no limit)
        
    Returns:
        List of swap operations as (index1, index2) tuples
    """
    swaps = []
    temp = current.copy()
    n = len(current)
    
    # Find positions where current differs from target
    for i in range(n):
        if temp[i] != target[i]:
            # Find where target[i] is located in temp
            target_value = target[i]
            j = i
            while j < n and temp[j] != target_value:
                j += 1
            
            if j < n and j != i:
                # Record swap that would move target_value to position i
                swaps.append((i, j))
                # Apply swap to temp to track progress
                temp[i], temp[j] = temp[j], temp[i]
    
    # Randomize and limit swap sequence
    if swaps and max_swaps is not None and len(swaps) > max_swaps:
        indices = rng_wrapper.rng.choice(len(swaps), size=max_swaps, replace=False)
        swaps = [swaps[idx] for idx in sorted(indices)]
    
    return swaps