from util.define import *
from typing import Callable, override

import numpy as np
import numpy.typing as npt

DiscreteNeighborFunction = Callable[[FloatVector, Float], FloatVector]
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
        return f"Unknown discrete problem: objective_function={self.objective_function}, dimension={self.dimension}"
    
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
    
    def neighbor(self, x: FloatVector, step_size: Float) -> FloatVector:
        """Generate a neighboring solution by randomly perturbing one dimension"""
        if(len(x) != self.dimension):
            raise ValueError(f"Input vector length ({len(x)}) != dimension ({self.dimension})")
        
        return self.neighbor_function(x, step_size)


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

    def tsp_neighbor(self, x: FloatVector, step_size: Float) -> FloatVector:
        """Generate a neighboring solution by swapping two cities in the tour"""
        neighbor = x.copy()
        idx1, idx2 = np.random.choice(self.dimension, size=2, replace=False)
        neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
        return neighbor

    def tsp_random_solution(self, rng: RNGWrapper) -> FloatVector:
        """Generate a random tour (permutation of cities)"""
        tour = np.arange(self.dimension)
        rng.rng.shuffle(tour)
        return tour.astype(float)
    
