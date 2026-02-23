import numpy as np
import numpy.typing as npt
from typing import Callable
import time

Float = float
FloatVector = npt.NDArray[np.float64]
ContinuousFunction = Callable[[FloatVector], float]
    
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
    