from dataclasses import dataclass, field

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
    
class RNGWrapper:
    def __init__(self, seed: int | None = None):
        if seed is None:
            seed = np.random.SeedSequence().entropy
        self.rng = np.random.default_rng(seed)
        self._seed = seed

    def uniform(self, low: Float, high: Float, size: int) -> FloatVector:
        return self.rng.uniform(low, high, size)

    def random(self) -> Float:
        return self.rng.random()
    
    def get_seed(self) -> int:
        return self._seed
    
@dataclass
class _Result:
    type: str
    algorithm: str
    objective_function: str
    iterations: int
    history_info: list[str]
    rng_seed: int

@dataclass 
class ContinuousResult(_Result):
    type: str = field(init=False, default="continuous")
    last_x: FloatVector
    last_value: Float
    history_x: list[FloatVector]
    history_value: list[Float]