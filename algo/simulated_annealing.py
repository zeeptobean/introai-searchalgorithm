from function.discrete_function import DiscreteProblem, TSPFunction, KnapsackFunction, GraphColoringFunction
from util.define import *
from util.util import *
from util.result import ContinuousResult
from util.result import DiscreteResult
from function.continuous_function import ContinuousProblem

def simulated_annealing_continuous(
    problem: ContinuousProblem, 
    temp: Float = 10.0, 
    cooling_rate: Float = 0.95, 
    min_temp: Float = 0.001, 
    step_bound: Float = 0.1, 
    max_iteration: int = 10000, 
    rng_seed: int | None = None
) -> ContinuousResult:
    rng = RNGWrapper(rng_seed)
    lower_bound = problem.lower_bound if problem.lower_bound is not None else -100
    upper_bound = problem.upper_bound if problem.upper_bound is not None else 100

    history = HistoryEntry()
    timer = TimerWrapper()
    timer.start()

    current_x = rng.uniform(lower_bound, upper_bound, size=problem.dimension)
    current_energy = problem.evaluate(current_x)

    history.add([current_x], [current_energy], f"temp: {temp:.4f}, delta: None")

    iteration = 1
    while temp > min_temp and iteration <= max_iteration:
        noise = rng.uniform(-step_bound, step_bound, size=problem.dimension)
        next_x = np.clip(current_x + noise, lower_bound, upper_bound)
        next_energy = problem.objective_function(next_x)
        delta = current_energy - next_energy #minimizing energy => delta > 0 is better 

        if delta > 0 or rng.random() < np.exp(delta / temp):
            current_x = next_x
            current_energy = next_energy
            history.add([next_x], [next_energy], f"temp: {temp:.4f}, delta: {delta:.4f} (accepted)")
        else:
            history.add([next_x], [current_energy], f"temp: {temp:.4f}, delta: {delta:.4f} (rejected)")

        temp *= cooling_rate
        iteration += 1

    total_time = timer.stop()

    best_x, best_value = history.get_best_value()

    return ContinuousResult(
        algorithm=f"Simulated Annealing (geometric-cooling, temp={temp:.4f}, min_temp={min_temp:.4f}, cooling_rate={cooling_rate:.4f})",
        short_name="SA",
        problem=problem,
        last_x=[current_x],
        last_value=[current_energy],
        best_x=best_x,
        best_value=best_value,
        time=total_time,
        iterations=iteration,
        rng_seed=rng.get_seed(),
        history=history
    )
"""
Temperature is scaled linearly with iteration count
"""
def simulated_annealing_linear_continuous(
    problem: ContinuousProblem, 
    max_temp: Float = 10.0, 
    step_bound: Float = 0.1, 
    max_iteration: int = 10000, 
    rng_seed: int | None = None
) -> ContinuousResult:
    rng = RNGWrapper(rng_seed)
    lower_bound = problem.lower_bound if problem.lower_bound is not None else -100
    upper_bound = problem.upper_bound if problem.upper_bound is not None else 100

    history = HistoryEntry()

    timer = TimerWrapper()
    timer.start()

    current_x = rng.uniform(lower_bound, upper_bound, size=problem.dimension)
    current_energy = problem.evaluate(current_x)

    history.add([current_x], [current_energy], f"temp: {max_temp:.4f}, delta: None")

    iteration = 0
    while iteration <= max_iteration:
        temp = max_temp * (1 - iteration/ max_iteration)
        temp = max(temp, 1e-12)     # Avoid divide by 0
        noise = rng.uniform(-step_bound, step_bound, size=problem.dimension)
        next_x = np.clip(current_x + noise, lower_bound, upper_bound)
        next_energy = problem.objective_function(next_x)
        delta = current_energy - next_energy

        if delta > 0 or rng.random() < np.exp(delta / temp):
            current_x = next_x
            current_energy = next_energy
            history.add([next_x], [next_energy], f"temp: {temp:.4f}, delta: {delta:.4f} (accepted)")
        else:
            history.add([next_x], [next_energy], f"temp: {temp:.4f}, delta: {delta:.4f} (rejected)")

        iteration += 1
    total_time = timer.stop()

    best_x, best_value = history.get_best_value()

    return ContinuousResult(
        algorithm=f"Simulated Annealing (linear-cooling, max_temp={max_temp:.4f}, max_iteration={max_iteration})",
        short_name="SA",
        problem=problem,
        last_x=[current_x],
        last_value=[current_energy],
        best_x=best_x,
        best_value=best_value,
        time=total_time,
        iterations=iteration,
        rng_seed=rng.get_seed(),
        history=history
    )

# discrete
def simulated_annealing_discrete_tsp(
    problem: TSPFunction, 
    temp: Float = 10.0, 
    cooling_rate: Float = 0.95, 
    min_temp: Float = 0.001, 
    max_iteration: int = 10000, 
    cooling_schedule: str = "geometric",
    rng_seed: int | None = None
) -> DiscreteResult:
    """
    Simulated Annealing for TSP problems.
    Uses 2-opt swap for neighbor generation.
    Accepts worse solutions with Metropolis probability: P = exp(-ΔE/T)
    
    Args:
        problem: The TSP problem to solve.
        temp: Initial temperature (for geometric cooling).
        cooling_rate: Cooling rate (for geometric cooling).
        min_temp: Minimum temperature (for geometric cooling).
        max_iteration: Maximum number of iterations.
        cooling_schedule: "geometric" or "linear".
        rng_seed: Seed for random number generator.
    """
    original_temp = temp
    rng = RNGWrapper(rng_seed)
    timer = TimerWrapper()
    timer.start()

    current_x = problem.random_solution(rng)
    current_energy = problem.evaluate(current_x)

    history = HistoryEntry(problem.is_max_value_problem())
    history.add([current_x], [current_energy], f"temp: {temp:.4f}, delta: None")

    iteration = 0
    while iteration < max_iteration:
        # Update temperature based on cooling schedule
        if cooling_schedule == "linear":
            temp = original_temp * (1 - iteration / max_iteration)
            temp = max(temp, 1e-12)
            if iteration >= max_iteration:
                break
        else:  # geometric
            if temp <= min_temp:
                break
        
        # Generate neighbor using 2-opt swap
        indices = sorted(rng.rng.choice(problem.dimension, size=2, replace=False))
        i_swap, j_swap = indices[0], indices[1]
        next_x = current_x.copy()
        next_x[i_swap:j_swap+1] = next_x[i_swap:j_swap+1][::-1]
        
        next_energy = problem.evaluate(next_x)
        delta = current_energy - next_energy  # minimizing energy => delta > 0 is better

        # Metropolis acceptance criterion
        if delta > 0 or rng.random() < np.exp(delta / temp):
            infostr = f"temp: {temp:.4f}, delta: {delta:.4f} (accepted)"
            current_x = next_x
            current_energy = next_energy
        else:
            infostr = f"temp: {temp:.4f}, delta: {delta:.4f} (rejected)"
        
        history.add([next_x], [next_energy], infostr)

        if cooling_schedule == "geometric":
            temp *= cooling_rate
        
        iteration += 1

    total_time = timer.stop()
    best_x, best_value = history.get_best_value()

    return DiscreteResult(
        algorithm=f"SA-TSP ({cooling_schedule}-cooling)",
        short_name="SA",
        problem=problem,
        last_x=[current_x],
        last_value=[current_energy],
        best_x=best_x,
        best_value=best_value,
        time=total_time,
        iterations=iteration,
        rng_seed=rng.get_seed(),
        history=history
    )


def simulated_annealing_discrete_knapsack(
    problem: KnapsackFunction, 
    temp: Float = 10.0, 
    cooling_rate: Float = 0.95, 
    min_temp: Float = 0.001, 
    max_iteration: int = 10000, 
    cooling_schedule: str = "geometric",
    rng_seed: int | None = None
) -> DiscreteResult:
    """
    Simulated Annealing for Knapsack problems.
    Uses bit flip for neighbor generation.
    Accepts worse solutions with Metropolis probability: P = exp(-ΔE/T)
    
    Args:
        problem: The Knapsack problem to solve.
        temp: Initial temperature (for geometric cooling).
        cooling_rate: Cooling rate (for geometric cooling).
        min_temp: Minimum temperature (for geometric cooling).
        max_iteration: Maximum number of iterations.
        cooling_schedule: "geometric" or "linear".
        rng_seed: Seed for random number generator.
    """
    original_temp = temp
    rng = RNGWrapper(rng_seed)
    timer = TimerWrapper()
    timer.start()

    current_x = problem.random_solution(rng)
    current_energy = problem.evaluate(current_x)

    history = HistoryEntry(problem.is_max_value_problem())
    history.add([current_x], [current_energy], f"temp: {temp:.4f}, delta: None")

    iteration = 0
    while iteration < max_iteration:
        # Update temperature based on cooling schedule
        if cooling_schedule == "linear":
            temp = original_temp * (1 - iteration / max_iteration)
            temp = max(temp, 1e-12)
            if iteration >= max_iteration:
                break
        else:  # geometric
            if temp <= min_temp:
                break
        
        # Generate neighbor using bit flip (1-3 random flips)
        next_x = current_x.copy()
        num_flips = rng.rng.integers(1, 4)  # 1 to 3 flips
        flip_indices = rng.rng.choice(problem.dimension, size=num_flips, replace=False)
        for idx in flip_indices:
            next_x[idx] = 1 - next_x[idx]
        
        next_energy = problem.evaluate(next_x)
        delta = current_energy - next_energy  # minimizing energy => delta > 0 is better

        # Metropolis acceptance criterion
        if delta > 0 or rng.random() < np.exp(delta / temp):
            infostr = f"temp: {temp:.4f}, delta: {delta:.4f} (accepted)"
            current_x = next_x
            current_energy = next_energy
        else:
            infostr = f"temp: {temp:.4f}, delta: {delta:.4f} (rejected)"
        
        history.add([next_x], [next_energy], infostr)

        if cooling_schedule == "geometric":
            temp *= cooling_rate
        
        iteration += 1

    total_time = timer.stop()
    best_x, best_value = history.get_best_value()

    return DiscreteResult(
        algorithm=f"SA-Knapsack ({cooling_schedule}-cooling)",
        problem=problem,
        last_x=[current_x],
        last_value=[current_energy],
        best_x=best_x,
        best_value=best_value,
        time=total_time,
        iterations=iteration,
        rng_seed=rng.get_seed(),
        history=history
    )


def simulated_annealing_discrete_graphcoloring(
    problem: GraphColoringFunction, 
    temp: Float = 10.0, 
    cooling_rate: Float = 0.95, 
    min_temp: Float = 0.001, 
    max_iteration: int = 10000, 
    cooling_schedule: str = "geometric",
    rng_seed: int | None = None
) -> DiscreteResult:
    """
    Simulated Annealing for Graph Coloring problems.
    Uses color change for neighbor generation.
    Accepts worse solutions with Metropolis probability: P = exp(-ΔE/T)
    
    Args:
        problem: The Graph Coloring problem to solve.
        temp: Initial temperature (for geometric cooling).
        cooling_rate: Cooling rate (for geometric cooling).
        min_temp: Minimum temperature (for geometric cooling).
        max_iteration: Maximum number of iterations.
        cooling_schedule: "geometric" or "linear".
        rng_seed: Seed for random number generator.
    """
    original_temp = temp
    rng = RNGWrapper(rng_seed)
    timer = TimerWrapper()
    timer.start()

    current_x = problem.random_solution(rng)
    current_energy = problem.evaluate(current_x)

    history = HistoryEntry(problem.is_max_value_problem())
    history.add([current_x], [current_energy], f"temp: {temp:.4f}, delta: None")

    iteration = 0
    while iteration < max_iteration:
        # Update temperature based on cooling schedule
        if cooling_schedule == "linear":
            temp = original_temp * (1 - iteration / max_iteration)
            temp = max(temp, 1e-12)
            if iteration >= max_iteration:
                break
        else:  # geometric
            if temp <= min_temp:
                break
        
        # Generate neighbor by changing color of 1-2 random vertices
        next_x = current_x.copy()
        num_changes = rng.rng.integers(1, 3)  # 1 to 2 changes
        vertices_to_change = rng.rng.choice(problem.dimension, size=num_changes, replace=False)
        
        max_color = int(np.max(current_x))
        for vertex in vertices_to_change:
            # Assign a random color
            next_x[vertex] = rng.rng.integers(0, max_color + 1)
        
        next_energy = problem.evaluate(next_x)
        delta = current_energy - next_energy  # minimizing energy => delta > 0 is better

        # Metropolis acceptance criterion
        if delta > 0 or rng.random() < np.exp(delta / temp):
            infostr = f"temp: {temp:.4f}, delta: {delta:.4f} (accepted)"
            current_x = next_x
            current_energy = next_energy
        else:
            infostr = f"temp: {temp:.4f}, delta: {delta:.4f} (rejected)"
        
        history.add([next_x], [next_energy], infostr)

        if cooling_schedule == "geometric":
            temp *= cooling_rate
        
        iteration += 1

    total_time = timer.stop()
    best_x, best_value = history.get_best_value()

    return DiscreteResult(
        algorithm=f"SA-GraphColoring ({cooling_schedule}-cooling)",
        short_name="SA",
        problem=problem,
        last_x=[current_x],
        last_value=[current_energy],
        best_x=best_x,
        best_value=best_value,
        time=total_time,
        iterations=iteration,
        rng_seed=rng.get_seed(),
        history=history
    )


def simulated_annealing_discrete(
    problem: DiscreteProblem, 
    temp: Float = 10.0, 
    cooling_rate: Float = 0.95, 
    min_temp: Float = 0.001, 
    step_bound: int = 2, 
    max_iteration: int = 10000, 
    cooling_schedule: str = "geometric",
    rng_seed: int | None = None
) -> DiscreteResult:
    """
    Simulated Annealing dispatcher for discrete optimization problems.
    Automatically selects the appropriate specialized implementation based on problem type.
    
    Uses Metropolis acceptance criterion: P = exp(-ΔE/T)
    Supports geometric and linear cooling schedules.
    
    Specialized implementations:
    - TSPFunction: Uses 2-opt swap for neighbor generation
    - KnapsackFunction: Uses bit flip for neighbor generation
    - GraphColoringFunction: Uses color change for neighbor generation
    - Generic: Uses problem.neighbor() function
    
    Args:
        problem: The discrete optimization problem to solve.
        temp: Initial temperature (for geometric cooling).
        cooling_rate: Cooling rate (for geometric cooling).
        min_temp: Minimum temperature (for geometric cooling).
        step_bound: Step size for neighbor generation (only for generic problems).
        max_iteration: Maximum number of iterations.
        cooling_schedule: "geometric" or "linear".
        rng_seed: Seed for random number generator.
    
    Returns:
        DiscreteResult containing the optimization results.
    """
    if isinstance(problem, TSPFunction):
        return simulated_annealing_discrete_tsp(problem, temp, cooling_rate, min_temp, max_iteration, cooling_schedule, rng_seed)
    elif isinstance(problem, KnapsackFunction):
        return simulated_annealing_discrete_knapsack(problem, temp, cooling_rate, min_temp, max_iteration, cooling_schedule, rng_seed)
    elif isinstance(problem, GraphColoringFunction):
        return simulated_annealing_discrete_graphcoloring(problem, temp, cooling_rate, min_temp, max_iteration, cooling_schedule, rng_seed)
    else:
        # Generic SA for other discrete problems
        original_temp = temp
        rng = RNGWrapper(rng_seed)
        timer = TimerWrapper()
        timer.start()

        current_x = problem.random_solution(rng)
        current_energy = problem.evaluate(current_x)

        history = HistoryEntry(problem.is_max_value_problem())
        history.add([current_x], [current_energy], f"temp: {temp:.4f}, delta: None")

        iteration = 0
        while iteration < max_iteration:
            # Update temperature based on cooling schedule
            if cooling_schedule == "linear":
                temp = original_temp * (1 - iteration / max_iteration)
                temp = max(temp, 1e-12)
                if iteration >= max_iteration:
                    break
            else:  # geometric
                if temp <= min_temp:
                    break
            
            noise = int(rng.rng.integers(1, step_bound, endpoint=True))
            next_x = problem.neighbor(current_x, noise, rng)
            next_energy = problem.evaluate(next_x)

            delta = current_energy - next_energy  # minimizing energy => delta > 0 is better

            if delta > 0 or rng.random() < np.exp(delta / temp):
                infostr = f"temp: {temp:.4f}, delta: {delta:.4f} (accepted)"
                current_x = next_x
                current_energy = next_energy
            else:
                infostr = f"temp: {temp:.4f}, delta: {delta:.4f} (rejected)"
            history.add([next_x], [next_energy], infostr)

            if cooling_schedule == "geometric":
                temp *= cooling_rate
            
            iteration += 1

        total_time = timer.stop()
        best_x, best_value = history.get_best_value()

        return DiscreteResult(
            algorithm=f"Simulated Annealing ({cooling_schedule}-cooling)",
            problem=problem,
            last_x=[current_x],
            last_value=[current_energy],
            best_x=best_x,
            best_value=best_value,
            time=total_time,
            iterations=iteration,
            rng_seed=rng.get_seed(),
            history=history
        )


# Legacy functions for backward compatibility
def simulated_annealing_discrete_geometric(
    problem: DiscreteProblem, 
    temp: Float = 10.0, 
    cooling_rate: Float = 0.95, 
    min_temp: Float = 0.001, 
    step_bound: int = 2, 
    max_iteration: int = 10000, 
    rng_seed: int | None = None
) -> DiscreteResult:
    """Legacy function - calls simulated_annealing_discrete with geometric cooling."""
    return simulated_annealing_discrete(
        problem, temp, cooling_rate, min_temp, step_bound, max_iteration, "geometric", rng_seed
    )


def simulated_annealing_linear_discrete(
    problem: DiscreteProblem, 
    max_temp: Float = 10.0, 
    step_bound: int = 2, 
    max_iteration: int = 10000, 
    rng_seed: int | None = None
) -> DiscreteResult:
    """Legacy function - calls simulated_annealing_discrete with linear cooling."""
    return simulated_annealing_discrete(
        problem, max_temp, 0.95, 0.001, step_bound, max_iteration, "linear", rng_seed
    )