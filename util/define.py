from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from typing import Callable
import time

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
    
class TimerWrapper:
    def start(self):
        self.start_time = time.perf_counter()

    # return time in millsecond, round to 1 decimal place
    def stop(self) -> float:
        end_time = time.perf_counter()
        return round((end_time - self.start_time) * 1000, 1)
    
@dataclass
class _Result:
    type: str
    algorithm: str
    objective_function: str
    iterations: int
    history_info: list[str]     # additional info for each iteration
    rng_seed: int
    time: float

@dataclass 
class ContinuousResult(_Result):
    type: str = field(init=False, default="continuous")
    last_x: list[FloatVector]   # x state of the last iteration
    last_value: list[Float]
    best_x: FloatVector         # the best
    best_value: Float
    history_x: list[list[FloatVector]]  # history for each x for each iteration
    history_value: list[list[Float]]

    def _format_float(self, v: Float) -> str:
        return f"{float(v):.9f}"
    def _format_npfloat_list(self, fl: list[Float]):
        arr = [self._format_float(v) for v in fl]
        return "[" + ", ".join(arr) + "]"
    def _format_floatvector(self, fv: FloatVector) -> str:
        return "(" + ", ".join(f"{float(x):.9f}" for x in fv) + ")"
    def _format_floatvector_list(self, fl: list[FloatVector]) -> str:
        result: list[str] = []
        for v in fl:
            result.append(self._format_floatvector(v))
        return "[" + " , ".join(result) + "]"
    
    def __repr__(self) -> str:
        format_history_value = []
        for ele in self.history_value:
            format_history_value.append(self._format_npfloat_list(ele))
        format_history_x = []
        for ele in self.history_x:
            format_history_x.append(self._format_floatvector_list(ele))

        history_str = ""
        for i, (x, val, info) in enumerate(zip(format_history_x, format_history_value, self.history_info)):
            history_str += f"Ite {i}: x = {x}; value = {val}; info = {info}\n"

        format_str = f"""
Algorithm: {self.algorithm}
Objective Function: {self.objective_function}
Running time: {self.time} ms
Iterations: {self.iterations}
RNG Seed: {self.rng_seed}
Best solution: x = {self._format_floatvector(self.best_x)}, value = {self._format_float(self.best_value)}
Last solution: x = {self._format_floatvector_list(self.last_x)}, value = {self._format_npfloat_list(self.last_value)}
History:
{history_str}
"""
        return format_str