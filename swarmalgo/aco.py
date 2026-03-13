from util.define import *
from util.util import *
import numpy as np
from util.result import DiscreteResult
from function.discrete_function import DiscreteProblem, TSPFunction, KnapsackFunction, GraphColoringFunction

def aco_discrete_tsp(
    problem: DiscreteProblem,
    num_ants: int = 20,
    generation: int = 100,
    alpha: float = 1.0,
    beta: float = 2.0,
    evaporation_rate: float = 0.5,
    pheromone_deposit_weight: float = 1.0,
    initial_pheromone: float = 1.0,
    rng_seed: int | None = None
) -> DiscreteResult:
    """
    Ant Colony Optimization (ACO) for TSP problem.
    
    This implementation uses edge-based pheromone trails where ants construct
    tours by selecting the next city using roulette wheel selection based on
    pheromone intensity (τ) and heuristic information (η = 1/distance).
    
    Args:
        problem: The TSP problem to solve.
        num_ants: Number of ants in the colony.
        generation: Number of iterations to run.
        alpha: Pheromone importance factor (τ^α).
        beta: Heuristic importance factor (η^β).
        evaporation_rate: Pheromone evaporation rate ρ ∈ [0, 1].
        pheromone_deposit_weight: Weight Q for pheromone deposit.
        initial_pheromone: Initial pheromone level on all edges.
        rng_seed: Seed for random number generator.
        
    Returns:
        DiscreteResult containing the optimization results.
    """
    if num_ants <= 0:
        raise ValueError("num_ants must be positive")
    if generation <= 0:
        raise ValueError("generation must be positive")
    if not (0 <= evaporation_rate <= 1):
        raise ValueError("evaporation_rate must be in [0, 1]")
    if alpha < 0 or beta < 0:
        raise ValueError("alpha and beta must be non-negative")

    rng_wrapper = RNGWrapper(rng_seed)
    timer = TimerWrapper()
    timer.start()

    # Get distance matrix from TSP problem
    from function.discrete_function import TSPFunction
    if not isinstance(problem, TSPFunction):
        raise ValueError("ACO for TSP requires a TSPFunction problem instance")
    
    distance_matrix = problem.distance_matrix
    num_cities = problem.dimension
    
    # Initialize pheromone matrix (edge-based: τ[i][j] = pheromone on edge from city i to city j)
    pheromone = np.full((num_cities, num_cities), initial_pheromone, dtype=float)
    
    # Calculate heuristic matrix (η[i][j] = 1/distance[i][j])
    heuristic = np.zeros((num_cities, num_cities), dtype=float)
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j and distance_matrix[i][j] > 0:
                heuristic[i][j] = 1.0 / distance_matrix[i][j]
            else:
                heuristic[i][j] = 0.0
    
    # Track best solution
    global_best_solution: FloatVector | None = None
    global_best_fitness = float('inf')
    
    # History tracking
    history = HistoryEntry(is_max_value_problem=problem.is_max_value_problem())
    
    for gen in range(generation):
        # Store all ant solutions in this generation
        ant_tours: list[FloatVector] = []
        ant_fitness: list[Float] = []
        
        # Each ant constructs a tour
        for ant_idx in range(num_ants):
            # Start from a random city
            current_city = rng_wrapper.rng.integers(0, num_cities)
            tour = [current_city]
            visited = {current_city}
            
            # Construct tour by selecting cities one by one
            while len(tour) < num_cities:
                # Get unvisited cities
                unvisited = [city for city in range(num_cities) if city not in visited]
                
                if not unvisited:
                    break
                
                # Calculate probabilities for each unvisited city using roulette wheel
                probabilities = []
                for next_city in unvisited:
                    # p_ij = (τ_ij^α * η_ij^β) / Σ(τ_ik^α * η_ik^β)
                    tau = pheromone[current_city][next_city]
                    eta = heuristic[current_city][next_city]
                    prob = (tau ** alpha) * (eta ** beta)
                    probabilities.append(prob)
                
                # Normalize probabilities
                prob_sum = sum(probabilities)
                if prob_sum > 0:
                    probabilities = [p / prob_sum for p in probabilities]
                else:
                    # If all probabilities are 0, use uniform distribution
                    probabilities = [1.0 / len(unvisited)] * len(unvisited)
                
                # Roulette wheel selection
                selected_idx = rng_wrapper.rng.choice(len(unvisited), p=probabilities)
                next_city = unvisited[selected_idx]
                
                # Add to tour
                tour.append(next_city)
                visited.add(next_city)
                current_city = next_city
            
            # Convert tour to numpy array
            tour_array = np.array(tour, dtype=float)
            ant_tours.append(tour_array)
            
            # Evaluate tour
            tour_fitness = problem.evaluate(tour_array)
            ant_fitness.append(tour_fitness)
            
            # Update global best
            if tour_fitness < global_best_fitness:
                global_best_fitness = tour_fitness
                global_best_solution = tour_array.copy()
        
        # Pheromone evaporation: τ_ij = (1 - ρ) * τ_ij
        pheromone *= (1 - evaporation_rate)
        
        # Pheromone deposit: Each ant deposits pheromone on edges it used
        for tour, fitness in zip(ant_tours, ant_fitness):
            if fitness == float('inf') or fitness == float('-inf'):
                continue  # Skip invalid tours
            
            # Deposit amount: Δτ = Q / L (where L is tour length)
            deposit = pheromone_deposit_weight / fitness
            
            # Add pheromone to all edges in the tour
            for i in range(len(tour)):
                from_city = int(tour[i])
                to_city = int(tour[(i + 1) % len(tour)])  # wrap around to start
                pheromone[from_city][to_city] += deposit
                pheromone[to_city][from_city] += deposit  # TSP is symmetric
        
        # Elitist strategy: Add extra pheromone for global best solution
        if global_best_solution is not None and global_best_fitness != float('inf'):
            elite_deposit = pheromone_deposit_weight / global_best_fitness
            for i in range(len(global_best_solution)):
                from_city = int(global_best_solution[i])
                to_city = int(global_best_solution[(i + 1) % len(global_best_solution)])
                pheromone[from_city][to_city] += elite_deposit
                pheromone[to_city][from_city] += elite_deposit
        
        # Track history
        avg_pheromone = np.mean(pheromone)
        history.add(
            [tour.copy() for tour in ant_tours],
            ant_fitness,
            f"gen={gen+1}, best={global_best_fitness:.4f}, avg_pheromone={avg_pheromone:.4f}"
        )
    
    total_time = timer.stop()
    
    # Get the best solution from history
    best_x, best_value = history.get_best_value()
    
    return DiscreteResult(
        algorithm="Ant Colony Optimization (TSP)",
        problem=problem,
        time=total_time,
        last_x=ant_tours,
        last_value=ant_fitness,
        best_x=best_x,
        best_value=best_value,
        iterations=generation,
        rng_seed=rng_wrapper.get_seed(),
        history=history
    )


def aco_discrete_knapsack(
    problem: DiscreteProblem,
    num_ants: int = 20,
    generation: int = 100,
    alpha: float = 1.0,
    beta: float = 2.0,
    evaporation_rate: float = 0.5,
    pheromone_deposit_weight: float = 1.0,
    initial_pheromone: float = 1.0,
    rng_seed: int | None = None
) -> DiscreteResult:
    """
    Ant Colony Optimization (ACO) for Knapsack problem.
    
    This implementation uses item-based pheromone trails where ants construct
    solutions by selecting items using roulette wheel selection based on
    pheromone intensity (τ) and heuristic information (η = value/weight ratio).
    
    Args:
        problem: The Knapsack problem to solve.
        num_ants: Number of ants in the colony.
        generation: Number of iterations to run.
        alpha: Pheromone importance factor (τ^α).
        beta: Heuristic importance factor (η^β).
        evaporation_rate: Pheromone evaporation rate ρ ∈ [0, 1].
        pheromone_deposit_weight: Weight Q for pheromone deposit.
        initial_pheromone: Initial pheromone level on all items.
        rng_seed: Seed for random number generator.
        
    Returns:
        DiscreteResult containing the optimization results.
    """
    if num_ants <= 0:
        raise ValueError("num_ants must be positive")
    if generation <= 0:
        raise ValueError("generation must be positive")
    if not (0 <= evaporation_rate <= 1):
        raise ValueError("evaporation_rate must be in [0, 1]")
    if alpha < 0 or beta < 0:
        raise ValueError("alpha and beta must be non-negative")

    rng_wrapper = RNGWrapper(rng_seed)
    timer = TimerWrapper()
    timer.start()

    # Get problem data
    if not isinstance(problem, KnapsackFunction):
        raise ValueError("ACO for Knapsack requires a KnapsackFunction problem instance")
    
    weights = problem.weights
    values = problem.values
    capacity = problem.capacity
    num_items = problem.dimension
    
    # Initialize pheromone vector (τ[i] = pheromone on item i)
    pheromone = np.full(num_items, initial_pheromone, dtype=float)
    
    # Calculate heuristic vector (η[i] = value[i]/weight[i], profit-to-weight ratio)
    heuristic = np.zeros(num_items, dtype=float)
    for i in range(num_items):
        if weights[i] > 0:
            heuristic[i] = values[i] / weights[i]
        else:
            heuristic[i] = 0.0
    
    # Track best solution
    global_best_solution: FloatVector | None = None
    global_best_fitness = float('inf')
    
    # History tracking
    history = HistoryEntry(is_max_value_problem=problem.is_max_value_problem())
    
    for gen in range(generation):
        # Store all ant solutions in this generation
        ant_solutions: list[FloatVector] = []
        ant_fitness: list[Float] = []
        
        # Each ant constructs a solution
        for ant_idx in range(num_ants):
            # Initialize empty knapsack
            solution = np.zeros(num_items, dtype=float)
            current_weight = 0.0
            available_items = list(range(num_items))
            
            # Construct solution by selecting items one by one
            while available_items:
                # Filter items that can still fit
                feasible_items = [
                    item for item in available_items 
                    if current_weight + weights[item] <= capacity
                ]
                
                if not feasible_items:
                    break  # No more items can fit
                
                # Calculate probabilities for each feasible item using roulette wheel
                probabilities = []
                for item in feasible_items:
                    # p_i = (τ_i^α * η_i^β) / Σ(τ_j^α * η_j^β)
                    tau = pheromone[item]
                    eta = heuristic[item]
                    prob = (tau ** alpha) * (eta ** beta)
                    probabilities.append(prob)
                
                # Normalize probabilities
                prob_sum = sum(probabilities)
                if prob_sum > 0:
                    probabilities = [p / prob_sum for p in probabilities]
                else:
                    # If all probabilities are 0, use uniform distribution
                    probabilities = [1.0 / len(feasible_items)] * len(feasible_items)
                
                # Roulette wheel selection
                selected_idx = rng_wrapper.rng.choice(len(feasible_items), p=probabilities)
                selected_item = feasible_items[selected_idx]
                
                # Add item to knapsack
                solution[selected_item] = 1.0
                current_weight += weights[selected_item]
                available_items.remove(selected_item)
            
            ant_solutions.append(solution)
            
            # Evaluate solution
            solution_fitness = problem.evaluate(solution)
            ant_fitness.append(solution_fitness)
            
            # Update global best (remember: knapsack returns negative value)
            if solution_fitness < global_best_fitness:
                global_best_fitness = solution_fitness
                global_best_solution = solution.copy()
        
        # Pheromone evaporation: τ_i = (1 - ρ) * τ_i
        pheromone *= (1 - evaporation_rate)
        
        # Pheromone deposit: Each ant deposits pheromone on selected items
        for solution, fitness in zip(ant_solutions, ant_fitness):
            if fitness == float('inf') or fitness == float('-inf'):
                continue  # Skip invalid solutions
            
            # Deposit amount: Δτ = Q / |f| (higher absolute value = more pheromone)
            # For knapsack, fitness is negative, so we use absolute value
            deposit = pheromone_deposit_weight / (abs(fitness) + 1e-9)
            
            # Add pheromone to selected items
            for i in range(num_items):
                if solution[i] > 0:  # Item is selected
                    pheromone[i] += deposit
        
        # Elitist strategy: Add extra pheromone for global best solution
        if global_best_solution is not None and global_best_fitness != float('inf'):
            elite_deposit = pheromone_deposit_weight / (abs(global_best_fitness) + 1e-9)
            for i in range(num_items):
                if global_best_solution[i] > 0:
                    pheromone[i] += elite_deposit
        
        # Track history
        avg_pheromone = np.mean(pheromone)
        history.add(
            [sol.copy() for sol in ant_solutions],
            ant_fitness,
            f"gen={gen+1}, best={-global_best_fitness:.4f}, avg_pheromone={avg_pheromone:.4f}"
        )
    
    total_time = timer.stop()
    
    # Get the best solution from history
    best_x, best_value = history.get_best_value()
    
    return DiscreteResult(
        algorithm="Ant Colony Optimization (Knapsack)",
        problem=problem,
        time=total_time,
        last_x=ant_solutions,
        last_value=ant_fitness,
        best_x=best_x,
        best_value=best_value,
        iterations=generation,
        rng_seed=rng_wrapper.get_seed(),
        history=history
    )


def aco_discrete_graph_coloring(
    problem: DiscreteProblem,
    num_ants: int = 20,
    generation: int = 100,
    alpha: float = 1.0,
    beta: float = 2.0,
    evaporation_rate: float = 0.5,
    pheromone_deposit_weight: float = 1.0,
    initial_pheromone: float = 1.0,
    rng_seed: int | None = None
) -> DiscreteResult:
    """
    Ant Colony Optimization (ACO) for Graph Coloring problem.
    
    This implementation uses vertex-color pheromone trails where ants construct
    colorings by assigning colors to vertices using roulette wheel selection based on
    pheromone intensity (τ) and heuristic information (η based on conflict avoidance).
    
    Args:
        problem: The Graph Coloring problem to solve.
        num_ants: Number of ants in the colony.
        generation: Number of iterations to run.
        alpha: Pheromone importance factor (τ^α).
        beta: Heuristic importance factor (η^β).
        evaporation_rate: Pheromone evaporation rate ρ ∈ [0, 1].
        pheromone_deposit_weight: Weight Q for pheromone deposit.
        initial_pheromone: Initial pheromone level on all vertex-color pairs.
        rng_seed: Seed for random number generator.
        
    Returns:
        DiscreteResult containing the optimization results.
    """
    if num_ants <= 0:
        raise ValueError("num_ants must be positive")
    if generation <= 0:
        raise ValueError("generation must be positive")
    if not (0 <= evaporation_rate <= 1):
        raise ValueError("evaporation_rate must be in [0, 1]")
    if alpha < 0 or beta < 0:
        raise ValueError("alpha and beta must be non-negative")

    rng_wrapper = RNGWrapper(rng_seed)
    timer = TimerWrapper()
    timer.start()

    # Get problem data
    if not isinstance(problem, GraphColoringFunction):
        raise ValueError("ACO for Graph Coloring requires a GraphColoringFunction problem instance")
    
    adjacency_matrix = problem.adjacency_matrix
    num_vertices = problem.dimension
    
    # Maximum colors to consider (upper bound: chromatic number ≤ n)
    max_colors = num_vertices
    
    # Initialize pheromone matrix (τ[vertex][color] = pheromone for assigning color to vertex)
    pheromone = np.full((num_vertices, max_colors), initial_pheromone, dtype=float)
    
    # Track best solution
    global_best_solution: FloatVector | None = None
    global_best_fitness = float('inf')
    
    # History tracking
    history = HistoryEntry(is_max_value_problem=problem.is_max_value_problem())
    
    for gen in range(generation):
        # Store all ant solutions in this generation
        ant_colorings: list[FloatVector] = []
        ant_fitness: list[Float] = []
        
        # Each ant constructs a coloring
        for ant_idx in range(num_ants):
            coloring = np.full(num_vertices, -1, dtype=float)
            
            # Color vertices in order (could be randomized)
            vertex_order = np.arange(num_vertices)
            rng_wrapper.rng.shuffle(vertex_order)
            
            for vertex in vertex_order:
                # Get colors used by adjacent vertices
                adjacent_colors = set()
                for neighbor in range(num_vertices):
                    if adjacency_matrix[vertex][neighbor] > 0 and coloring[neighbor] >= 0:
                        adjacent_colors.add(int(coloring[neighbor]))
                
                # Determine maximum color to consider
                # Start with colors used so far, plus one new color
                colors_used = set(int(c) for c in coloring if c >= 0)
                max_color_to_try = len(colors_used) if colors_used else 0
                
                # Available colors (not used by neighbors)
                available_colors = []
                for color in range(min(max_color_to_try + 1, max_colors)):
                    if color not in adjacent_colors:
                        available_colors.append(color)
                
                if not available_colors:
                    # Fallback: assign smallest unused color (even if conflicts)
                    color = 0
                    while color in adjacent_colors and color < max_colors:
                        color += 1
                    coloring[vertex] = float(color)
                    continue
                
                # Calculate probabilities for each available color using roulette wheel
                probabilities = []
                for color in available_colors:
                    # τ[vertex][color]^α
                    tau = pheromone[vertex][color]
                    
                    # Heuristic η: prefer colors already used (to minimize total colors)
                    # Higher value if color is already used in the partial solution
                    if color in colors_used:
                        eta = 2.0  # Prefer reusing colors
                    else:
                        eta = 1.0  # New color
                    
                    # p = (τ^α * η^β) / Σ(τ^α * η^β)
                    prob = (tau ** alpha) * (eta ** beta)
                    probabilities.append(prob)
                
                # Normalize probabilities
                prob_sum = sum(probabilities)
                if prob_sum > 0:
                    probabilities = [p / prob_sum for p in probabilities]
                else:
                    # If all probabilities are 0, use uniform distribution
                    probabilities = [1.0 / len(available_colors)] * len(available_colors)
                
                # Roulette wheel selection
                selected_idx = rng_wrapper.rng.choice(len(available_colors), p=probabilities)
                selected_color = available_colors[selected_idx]
                
                # Assign color to vertex
                coloring[vertex] = float(selected_color)
            
            ant_colorings.append(coloring)
            
            # Evaluate coloring
            coloring_fitness = problem.evaluate(coloring)
            ant_fitness.append(coloring_fitness)
            
            # Update global best
            if coloring_fitness < global_best_fitness:
                global_best_fitness = coloring_fitness
                global_best_solution = coloring.copy()
        
        # Pheromone evaporation: τ[v][c] = (1 - ρ) * τ[v][c]
        pheromone *= (1 - evaporation_rate)
        
        # Pheromone deposit: Each ant deposits pheromone on vertex-color pairs it used
        for coloring, fitness in zip(ant_colorings, ant_fitness):
            if fitness == float('inf') or fitness == float('-inf'):
                continue  # Skip invalid colorings
            
            # Deposit amount: Δτ = Q / fitness (better solutions deposit more)
            deposit = pheromone_deposit_weight / (fitness + 1e-9)
            
            # Add pheromone to vertex-color assignments
            for vertex in range(num_vertices):
                color = int(coloring[vertex])
                if 0 <= color < max_colors:
                    pheromone[vertex][color] += deposit
        
        # Elitist strategy: Add extra pheromone for global best solution
        if global_best_solution is not None and global_best_fitness != float('inf'):
            elite_deposit = pheromone_deposit_weight / (global_best_fitness + 1e-9)
            for vertex in range(num_vertices):
                color = int(global_best_solution[vertex])
                if 0 <= color < max_colors:
                    pheromone[vertex][color] += elite_deposit
        
        # Track history
        avg_pheromone = np.mean(pheromone)
        num_colors_best = len(np.unique(global_best_solution.astype(int))) if global_best_solution is not None else 0
        history.add(
            [col.copy() for col in ant_colorings],
            ant_fitness,
            f"gen={gen+1}, best_fitness={global_best_fitness:.4f}, colors={num_colors_best}, avg_pheromone={avg_pheromone:.4f}"
        )
    
    total_time = timer.stop()
    
    # Get the best solution from history
    best_x, best_value = history.get_best_value()
    
    return DiscreteResult(
        algorithm="Ant Colony Optimization (Graph Coloring)",
        problem=problem,
        time=total_time,
        last_x=ant_colorings,
        last_value=ant_fitness,
        best_x=best_x,
        best_value=best_value,
        iterations=generation,
        rng_seed=rng_wrapper.get_seed(),
        history=history
    )


def aco_discrete(
    problem: DiscreteProblem,
    num_ants: int = 20,
    generation: int = 100,
    alpha: float = 1.0,
    beta: float = 2.0,
    evaporation_rate: float = 0.5,
    pheromone_deposit_weight: float = 1.0,
    initial_pheromone: float = 1.0,
    rng_seed: int | None = None
) -> DiscreteResult:
    """
    Ant Colony Optimization (ACO) for discrete optimization problems.
    
    This is a dispatcher function that automatically selects the appropriate
    ACO variant based on the problem type:
    - TSPFunction -> aco_discrete_tsp (edge-based pheromone, roulette wheel)
    - KnapsackFunction -> aco_discrete_knapsack (item-based pheromone)
    - GraphColoringFunction -> aco_discrete_graph_coloring (vertex-color pheromone)
    
    Args:
        problem: The discrete optimization problem to solve.
        num_ants: Number of ants in the colony.
        generation: Number of iterations to run.
        alpha: Pheromone importance factor (τ^α).
        beta: Heuristic importance factor (η^β).
        evaporation_rate: Pheromone evaporation rate ρ ∈ [0, 1].
        pheromone_deposit_weight: Weight Q for pheromone deposit.
        initial_pheromone: Initial pheromone level.
        rng_seed: Seed for random number generator.
        
    Returns:
        DiscreteResult from the appropriate ACO variant.
    """
    # Dispatch to specialized ACO function based on problem type
    if isinstance(problem, TSPFunction):
        return aco_discrete_tsp(
            problem=problem,
            num_ants=num_ants,
            generation=generation,
            alpha=alpha,
            beta=beta,
            evaporation_rate=evaporation_rate,
            pheromone_deposit_weight=pheromone_deposit_weight,
            initial_pheromone=initial_pheromone,
            rng_seed=rng_seed
        )
    elif isinstance(problem, KnapsackFunction):
        return aco_discrete_knapsack(
            problem=problem,
            num_ants=num_ants,
            generation=generation,
            alpha=alpha,
            beta=beta,
            evaporation_rate=evaporation_rate,
            pheromone_deposit_weight=pheromone_deposit_weight,
            initial_pheromone=initial_pheromone,
            rng_seed=rng_seed
        )
    elif isinstance(problem, GraphColoringFunction):
        return aco_discrete_graph_coloring(
            problem=problem,
            num_ants=num_ants,
            generation=generation,
            alpha=alpha,
            beta=beta,
            evaporation_rate=evaporation_rate,
            pheromone_deposit_weight=pheromone_deposit_weight,
            initial_pheromone=initial_pheromone,
            rng_seed=rng_seed
        )
    else:
        # Fallback to graph coloring approach for unknown discrete problems
        return aco_discrete_graph_coloring(
            problem=problem,
            num_ants=num_ants,
            generation=generation,
            alpha=alpha,
            beta=beta,
            evaporation_rate=evaporation_rate,
            pheromone_deposit_weight=pheromone_deposit_weight,
            initial_pheromone=initial_pheromone,
            rng_seed=rng_seed
        )
