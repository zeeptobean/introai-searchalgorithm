from util.define import *
from typing import Callable, override

import numpy as np
import numpy.typing as npt

DiscreteNeighborFunction = Callable[[FloatVector, Float, RNGWrapper], FloatVector]
DiscreteRandomSolutionFunction = Callable[[RNGWrapper], FloatVector]

class DiscreteProblem:
    "Abstract class for discrete optimization problem"
    def __init__(self, objective_function: DiscreteFunction, dimension: int, neighbor_function: DiscreteNeighborFunction | None = None, random_solution_function: DiscreteRandomSolutionFunction | None = None):
        if(dimension <= 0):
            raise ValueError("Dimension must be greater than 0.")
        self.objective_function = objective_function
        self.dimension = dimension
        self.neighbor_function = neighbor_function
        self.random_solution_function = random_solution_function

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dimension={self.dimension})"
    
    def to_json(self) -> dict:
        return {
            "objective_function": self.__class__.__name__,
            "dimension": self.dimension,
        }
    
    @staticmethod
    def from_json(json_data: dict) -> 'DiscreteProblem':
        """Reconstruct a DiscreteProblem subclass from JSON metadata"""
        function_type = json_data.get("objective_function")
        dimension = json_data["dimension"]
        
        constructors = {
            "TSPFunction": lambda: TSPFunction(
                dimension,
                json_data.get("distance_matrix")
            ),
            "KnapsackFunction": lambda: KnapsackFunction(
                dimension,
                json_data.get("weights"),
                json_data.get("values"),
                json_data.get("capacity")
            ),
        }
        
        if function_type not in constructors:
            raise ValueError(f"Unknown function type: {function_type}")
        
        return constructors[function_type]()
    
    def random_solution(self, rng: RNGWrapper) -> FloatVector:
        """Generate a random solution within the bounds"""
        return self.random_solution_function(rng)

    def evaluate(self, x: FloatVector) -> Float:
        if(len(x) != self.dimension):
            raise ValueError(f"Input vector length ({len(x)}) != dimension ({self.dimension})")
        return self.objective_function(x)
    
    def neighbor(self, x: FloatVector, step_size: Float, rng: RNGWrapper) -> FloatVector:
        """Generate a neighboring solution by randomly perturbing one dimension"""
        if(len(x) != self.dimension):
            raise ValueError(f"Input vector length ({len(x)}) != dimension ({self.dimension})")
        
        return self.neighbor_function(x, step_size, rng)


class TSPFunction(DiscreteProblem):
    def __init__(self, distance_matrix: npt.NDArray[np.float64]):
        self.distance_matrix = distance_matrix
        dimension = distance_matrix.shape[0]
        super().__init__(objective_function=self.tsp_objective, neighbor_function=self.tsp_neighbor, random_solution_function=self.tsp_random_solution, dimension=dimension)

    def tsp_objective(self, tour: FloatVector) -> Float:
        """Calculate the total distance of the given tour"""
        total_distance = 0.0
        for i in range(len(tour)):
            from_city = int(tour[i])
            to_city = int(tour[(i + 1) % len(tour)])  # wrap around to the start
            total_distance += self.distance_matrix[from_city, to_city]
        return total_distance

    def tsp_neighbor(self, x: FloatVector, step_size: Float, rng: RNGWrapper) -> FloatVector:
        """Generate a neighboring solution by swapping two cities in the tour"""
        neighbor = x.copy()
        # Use step_size to determine number of swaps (at least 1)
        num_swaps = max(1, int(step_size))
        for _ in range(num_swaps):
            idx1, idx2 = rng.rng.choice(self.dimension, size=2, replace=False)
            neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
        return neighbor

    def tsp_random_solution(self, rng: RNGWrapper) -> FloatVector:
        """Generate a random tour (permutation of cities)"""
        tour = np.arange(self.dimension)
        rng.rng.shuffle(tour)
        return tour.astype(float)
    
    def __repr__(self) -> str:
        return f"TSPFunction(cities={self.dimension})"

# Return -value because we want to maximize value, but our algorithms minimize the objective
class KnapsackFunction(DiscreteProblem):
    def __init__(self, weights: npt.NDArray[np.float64], values: npt.NDArray[np.float64], capacity: Float):
        self.weights = weights
        self.values = values
        self.capacity = capacity
        dimension = weights.shape[0]
        super().__init__(objective_function=self.knapsack_objective, neighbor_function=self.knapsack_neighbor, random_solution_function=self.knapsack_random_solution, dimension=dimension)

    def knapsack_objective(self, selection: FloatVector) -> Float:
        """Calculate the total value of the given selection"""
        current_weight = np.dot(selection, self.weights)
        if current_weight > self.capacity:
            return float('inf')  # Penalize selections that exceed capacity
        return -np.dot(selection, self.values) # We negate because we want to maximize value, but our algorithms minimize the objective

    def knapsack_neighbor(self, x: FloatVector, step_size: Float, rng: RNGWrapper) -> FloatVector:
        """Generate a neighboring solution by swapping two items in the selection"""
        neighbor = x.copy()
        num_swaps = max(1, int(step_size))
        for _ in range(num_swaps):
            idx = rng.rng.integers(0, self.dimension)
            neighbor[idx] = 1 - neighbor[idx] 
        return neighbor

    def knapsack_random_solution(self, rng: RNGWrapper) -> FloatVector:
        """Generate a random selection (binary vector indicating item inclusion)"""
        return rng.rng.integers(0, 2, size=self.dimension).astype(float)
    
    def __repr__(self) -> str:
        return f"KnapsackFunction(items={self.dimension}, capacity={self.capacity})"

class GraphColoringFunction(DiscreteProblem):
    def __init__(self, adjacency_matrix: npt.NDArray[np.int_]):
        """
        Initialize Graph Coloring problem to find minimum number of colors needed.
        
        The algorithm will search for the minimum chromatic number by trying
        colorings with up to n colors (where n = number of vertices).
        This is always sufficient since any graph with n vertices can be colored
        with at most n colors.
        
        Args:
            adjacency_matrix: Adjacency matrix representing the graph
        """
        self.adjacency_matrix = adjacency_matrix
        dimension = adjacency_matrix.shape[0]
        super().__init__(objective_function=self.graph_coloring_objective, neighbor_function=self.graph_coloring_neighbor, random_solution_function=self.graph_coloring_random_solution, dimension=dimension)

    def graph_coloring_objective(self, colors: FloatVector) -> Float:
        """
        Calculate objective: number of colors used + penalty for conflicts.
        Goal is to minimize both the number of colors and conflicts.
        """
        # Count conflicts
        conflicts = 0
        for i in range(self.dimension):
            for j in range(i + 1, self.dimension):
                if self.adjacency_matrix[i, j] > 0 and int(colors[i]) == int(colors[j]):
                    conflicts += 1
        
        # Count unique colors used
        num_colors_used = len(np.unique(colors.astype(int)))
        
        # Objective: minimize colors + heavily penalize conflicts
        # Large penalty ensures we prioritize valid coloring over minimizing colors
        conflict_penalty = 10000
        return float(num_colors_used + conflict_penalty * conflicts)

    def graph_coloring_neighbor(self, x: FloatVector, step_size: Float, rng: RNGWrapper) -> FloatVector:
        """Generate a neighboring solution by changing the color of random vertices"""
        neighbor = x.copy()
        
        # Determine current number of colors used
        num_colors_used = len(np.unique(neighbor.astype(int)))
        # Allow colors in range [0, num_colors_used] to enable:
        # - Reducing colors (reuse existing colors)
        # - Adding at most one new color (explore new possibilities)
        max_color = min(num_colors_used, self.dimension - 1)
        
        num_changes = max(1, int(step_size))
        for _ in range(num_changes):
            vertex = rng.rng.integers(0, self.dimension)
            # Random color from currently used colors (or one new color)
            neighbor[vertex] = rng.rng.integers(0, max_color + 1)
        return neighbor

    def graph_coloring_random_solution(self, rng: RNGWrapper) -> FloatVector:
        """
        Generate a random coloring starting with fewer colors.
        Uses greedy approach to create initial solution with reasonable number of colors.
        """
        # Start with a greedy coloring to get a reasonable initial solution
        colors = np.full(self.dimension, -1, dtype=float)
        
        for vertex in range(self.dimension):
            # Get colors of adjacent vertices
            adjacent_colors = set()
            for neighbor in range(self.dimension):
                if self.adjacency_matrix[vertex, neighbor] > 0 and colors[neighbor] >= 0:
                    adjacent_colors.add(int(colors[neighbor]))
            
            # Find the smallest color not used by adjacent vertices
            color = 0
            while color in adjacent_colors:
                color += 1
            colors[vertex] = float(color)
        
        return colors
    
    def __repr__(self) -> str:
        return f"GraphColoringFunction(nodes={self.dimension})"