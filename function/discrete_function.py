from util.define import *
from typing import Callable, override, Union, Protocol

import numpy as np
import numpy.typing as npt

DiscreteNeighborFunction = Callable[[FloatVector, int, RNGWrapper], FloatVector]
DiscreteRandomSolutionFunction = Callable[[RNGWrapper], FloatVector]

class DiscreteProblem:
    "Abstract class for discrete optimization problem"
    def __init__(self, objective_function: DiscreteFunction, dimension: int, neighbor_function: DiscreteNeighborFunction, random_solution_function: DiscreteRandomSolutionFunction):
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
    
    def is_max_value_problem(self) -> bool:
        """
        Flag to indicate whether this is a min cost problem (like TSP) or max value problem (like Knapsack).
        By default, we assume it's a min cost problem and return False.
        Max value problems will return its best value as negative, this flag is used to flip to sign back on result
        """
        return False
    
    def random_solution(self, rng: RNGWrapper) -> FloatVector:
        """Generate a random solution within the bounds"""
        return self.random_solution_function(rng)

    def evaluate(self, x: FloatVector) -> Float:
        if(len(x) != self.dimension):
            raise ValueError(f"Input vector length ({len(x)}) != dimension ({self.dimension})")
        return self.objective_function(x)
    
    def neighbor(self, x: FloatVector, step_size: int, rng: RNGWrapper) -> FloatVector:
        """Generate a neighboring solution by randomly perturbing one dimension"""
        if(len(x) != self.dimension):
            raise ValueError(f"Input vector length ({len(x)}) != dimension ({self.dimension})")
        
        return self.neighbor_function(x, step_size, rng)


class TSPFunction(DiscreteProblem):
    def __init__(self, dimension: int, distance_matrix: npt.NDArray[np.float64]):
        self.distance_matrix = distance_matrix
        if distance_matrix.shape[0] != dimension or distance_matrix.shape[1] != dimension:
            raise ValueError(f"Distance matrix shape ({distance_matrix.shape}) do not match dimension size ({dimension})")
        super().__init__(objective_function=self.tsp_objective, neighbor_function=self.tsp_neighbor, random_solution_function=self.tsp_random_solution, dimension=dimension)

    @staticmethod
    def from_json(json_data: dict) -> 'TSPFunction':
        dimension = json_data["node_count"]
        distance_matrix = np.array(json_data["distance_matrix"])
        return TSPFunction(dimension=dimension, distance_matrix=distance_matrix)
    
    @override
    def to_json(self) -> dict:
        return super().to_json() | {
            "distance_matrix": self.distance_matrix.tolist()
        }

    def tsp_objective(self, tour: FloatVector) -> Float:
        """Calculate the total distance of the given tour"""
        total_distance = 0.0
        for i in range(len(tour)):
            from_city = int(tour[i])
            to_city = int(tour[(i + 1) % len(tour)])  # wrap around to the start
            total_distance += self.distance_matrix[from_city, to_city]
        return total_distance

    def tsp_neighbor(self, x: FloatVector, step_size: int, rng: RNGWrapper) -> FloatVector:
        """Generate a neighboring solution by swapping two cities in the tour"""
        neighbor = x.copy()
        # Use step_size to determine number of swaps (at least 1)
        num_swaps = max(1, step_size)
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
    def __init__(self, dimension: int, weights: npt.NDArray[np.float64], values: npt.NDArray[np.float64], capacity: Float):
        self.weights = weights
        self.values = values
        self.capacity = capacity
        if weights.shape[0] != dimension:
            raise ValueError(f"Weights length ({weights.shape[0]}) != dimension ({dimension})")
        if values.shape[0] != dimension:
            raise ValueError(f"Values length ({values.shape[0]}) != dimension ({dimension})")
        super().__init__(objective_function=self.knapsack_objective, neighbor_function=self.knapsack_neighbor, random_solution_function=self.knapsack_random_solution, dimension=dimension)

    @staticmethod
    def from_json(json_data: dict) -> 'KnapsackFunction':
        length = json_data["length"]
        weights = np.array(json_data["weights"])
        values = np.array(json_data["values"])
        capacity = json_data["capacity"]
        return KnapsackFunction(weights=weights, values=values, capacity=capacity, dimension=length)
    
    @override
    def to_json(self) -> dict:
        return super().to_json() | {
            "weights": self.weights.tolist(),
            "values": self.values.tolist(),
            "capacity": self.capacity
        }

    @override
    def is_max_value_problem(self) -> bool:
        """
        Knapsack is max value problem
        """
        return True

    def knapsack_objective(self, selection: FloatVector) -> Float:
        """Calculate the total value of the given selection"""
        current_weight = np.dot(selection, self.weights)
        if current_weight > self.capacity:
            return float('inf')  # Penalize selections that exceed capacity
        return -np.dot(selection, self.values) # We negate because we want to maximize value, but our algorithms minimize the objective

    def knapsack_neighbor(self, x: FloatVector, step_size: int, rng: RNGWrapper) -> FloatVector:
        """Generate a neighboring solution by swapping two items in the selection"""
        neighbor = x.copy()
        num_swaps = max(1, step_size)
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
    def __init__(self, dimension: int, adjacency_matrix: FloatVector):
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
        self.dimension = dimension
        if adjacency_matrix.shape[0] != dimension or adjacency_matrix.shape[1] != dimension:
            raise ValueError(f"Adjacency matrix shape ({adjacency_matrix.shape}) do not match dimension size ({dimension})")
        super().__init__(objective_function=self.graph_coloring_objective, neighbor_function=self.graph_coloring_neighbor, random_solution_function=self.graph_coloring_random_solution, dimension=dimension)

    @override
    def to_json(self) -> dict: 
        return super().to_json() | {
            "adjacency_matrix": self.adjacency_matrix.tolist()
        }

    @staticmethod
    def from_adjlist(adjacency_list: dict[int, list[int]]) -> 'GraphColoringFunction':
        """
        Node index start at 1 
        """
        dimension = 0
        for i, neighbors in adjacency_list.items():
            if i <= 0:
                raise ValueError(f"Node index must be positive integer, got {i}")
            dimension = max(dimension, i)
            for neighbor in neighbors:
                if neighbor <= 0:
                    raise ValueError(f"Node index must be positive integer, got {neighbor}")
                dimension = max(dimension, neighbor)
        dimension = dimension
        adjacency_matrix = np.zeros((dimension, dimension), dtype=int)
        for i, neighbors in adjacency_list.items():
            for neighbor in neighbors:
                adjacency_matrix[i-1, neighbor-1] = 1
                adjacency_matrix[neighbor-1, i-1] = 1
        return GraphColoringFunction(dimension=dimension, adjacency_matrix=adjacency_matrix)

    @staticmethod
    def from_json(json_data: dict) -> 'GraphColoringFunction':
        dimension = json_data["node_count"]
        adjacency_matrix = np.array(json_data["adjacency_matrix"])
        return GraphColoringFunction(dimension=dimension, adjacency_matrix=adjacency_matrix)
    
    @staticmethod
    def from_json_adjlist(json_data: dict) -> 'GraphColoringFunction':
        """
        Example JSON format:
        {
            "adjacency_list": {
                "1": [2, 3],
                "2": [1, 3],
            }
        }

        Index start at 1 
        """
        adjacency_list_str = json_data["adjacency_list"]
        adjacency_list: dict[int, list[int]] = {}
        
        for key in adjacency_list_str.keys():
            if not key.isdigit() or int(key) <= 0:
                raise ValueError(f"Invalid vertex index: {key}. Vertex indices must be integers starting from 1.")
            
            adjacency_list[int(key)] = adjacency_list_str[key]

        return GraphColoringFunction.from_adjlist(adjacency_list)

    def graph_coloring_objective(self, colors: FloatVector) -> int:
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
        return num_colors_used + conflict_penalty * conflicts

    def graph_coloring_neighbor(self, x: FloatVector, step_size: int, rng: RNGWrapper) -> FloatVector:
        """Generate a neighboring solution by changing the color of random vertices"""
        neighbor = x.copy()
        
        # Determine current number of colors used
        num_colors_used = len(np.unique(neighbor.astype(int)))
        # Allow colors in range [0, num_colors_used] to enable:
        # - Reducing colors (reuse existing colors)
        # - Adding at most one new color (explore new possibilities)
        max_color = min(num_colors_used, self.dimension - 1)
        
        num_changes = max(1, step_size)
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