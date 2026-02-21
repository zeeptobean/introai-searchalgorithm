from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from typing import Callable

Float = float
FloatVector = npt.NDArray[np.float64]
ContinuousFunction = Callable[[FloatVector], float]

class ContinuousProblem:
    def __init__(self, objective_function: ContinuousFunction, dimension: int, lower_bound: Float | None = None, upper_bound: Float | None = None):
        if(dimension <= 0):
            raise ValueError("Dimension must be greater than 0.")
        self.objective_function = objective_function
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __repr__(self) -> str:
        return f"Unknown continuous problem: objective_function={self.objective_function}, dimension={self.dimension}, bound=[{self.lower_bound}, {self.upper_bound}]"
    
    def evaluate(self, x: FloatVector) -> Float:
        if(len(x) != self.dimension):
            raise ValueError(f"Input vector length ({len(x)}) != dimension ({self.dimension})")
        if self.lower_bound is not None and self.upper_bound is not None:
            np.clip(x, self.lower_bound, self.upper_bound, out=x)
        return self.objective_function(x)
    
@dataclass
class Result:
    type: str
    algorithm: str
    objective_function: str
    iterations: int
    best_x: FloatVector
    best_value: Float
    history_x: list[FloatVector]
    history_value: list[Float]
    history_info: list[str]
    rng_seed: int
