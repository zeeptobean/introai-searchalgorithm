from util.define import *
from typing import Callable, override

import numpy as np
import numpy.typing as npt

class ContinuousProblem:
    "Abstract class for continuous optimization problem"
    def __init__(self, objective_function: ContinuousFunction, dimension: int, lower_bound: Float | None = None, upper_bound: Float | None = None):
        if(dimension <= 0):
            raise ValueError("Dimension must be greater than 0.")
        self.objective_function = objective_function
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __repr__(self) -> str:
        return f"Unknown continuous problem: objective_function={self.objective_function}, dimension={self.dimension}, bound=[{self.lower_bound}, {self.upper_bound}]"
    
    def to_json(self) -> dict:
        return {
            "objective_function": self.__class__.__name__,
            "dimension": self.dimension,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound
        }
    
    @staticmethod
    def from_json(json_data: dict) -> 'ContinuousProblem':
        """Reconstruct a ContinuousProblem subclass from JSON metadata"""
        function_type = json_data.get("objective_function")
        dimension = json_data["dimension"]
        
        constructors = {
            "RastriginFunction": lambda: RastriginFunction(
                dimension, 
                json_data.get("lower_bound", -5.12), 
                json_data.get("upper_bound", 5.12)
            ),
            "RosenbrockFunction": lambda: RosenbrockFunction(dimension),
            "SphereFunction": lambda: SphereFunction(dimension),
            "MichalewiczFunction": lambda: MichalewiczFunction(
                dimension, 
                json_data.get("lower_bound", 0.0),
                json_data.get("upper_bound", np.pi)
            ),
            "StyblinskiTangFunction": lambda: StyblinskiTangFunction(
                dimension,
                json_data.get("lower_bound", -5.0),
                json_data.get("upper_bound", 5.0)
            ),
            "GriewankFunction": lambda: GriewankFunction(
                dimension,
                json_data.get("lower_bound", -600.0),
                json_data.get("upper_bound", 600.0)
            ),
            "AckleyFunction": lambda: AckleyFunction(
                dimension,
                json_data.get("lower_bound", -32.768),
                json_data.get("upper_bound", 32.768)
            ),
        }
        
        if function_type not in constructors:
            raise ValueError(f"Unknown function type: {function_type}")
        
        return constructors[function_type]()
    
    def evaluate(self, x: FloatVector) -> Float:
        if(len(x) != self.dimension):
            raise ValueError(f"Input vector length ({len(x)}) != dimension ({self.dimension})")
        if self.lower_bound is not None and self.upper_bound is not None:
            np.clip(x, self.lower_bound, self.upper_bound, out=x)
        return self.objective_function(x)

"""
Global min: f(0) = 0
x_i within [-5.12, 5.12]
"""
class RastriginFunction(ContinuousProblem):
    def _rastrigin_function(self, x: FloatVector) -> Float:
        A = 10.0 
        return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

    def __init__(self, dimension: int, lower_bound: Float = -5.12, upper_bound: Float = 5.12):
        super().__init__(objective_function=self._rastrigin_function, dimension=dimension, lower_bound=lower_bound, upper_bound=upper_bound)

    @override
    def __repr__(self) -> str:
        return f"Rastrigin function: dimension={self.dimension}, bound=[{self.lower_bound}, {self.upper_bound}]"

"""
Global min: f(a, a^2, a^3, ..., a^n) = 0
"""
class RosenbrockFunction(ContinuousProblem):
    def _rosenbrock_function_generalized(self, x: FloatVector, a: Float = 1.0, b: Float = 100.0) -> Float:
        self.a = a
        self.b = b

        # term1 = (a - x_i)^2
        term1 = (a - x[:-1])**2

        # term2 = b * (x_{i+1} - x_i^2)^2
        term2 = b * (x[1:] - x[:-1]**2)**2
        return np.sum(term1 + term2)
    
    def _rosenbrock_function(self, x: FloatVector) -> Float:
        return self._rosenbrock_function_generalized(x, a=1.0, b=100.0)

    def __init__(self, dimension: int):
        super().__init__(objective_function=self._rosenbrock_function, dimension=dimension)

    @override
    def __repr__(self) -> str:
        return f"Rosenbrock function: dimension={self.dimension}, bound=[{self.lower_bound}, {self.upper_bound}], a={self.a}, b={self.b}"
    
"""
Global min: f(0, 0, ..., 0) = 0
"""
class SphereFunction(ContinuousProblem):
    def _sphere_function(self, x: FloatVector) -> Float:
        return np.sum(x**2)

    def __init__(self, dimension: int):
        super().__init__(objective_function=self._sphere_function, dimension=dimension)

    @override
    def __repr__(self) -> str:
        return f"Sphere function: dimension={self.dimension}"

"""
Michalewicz Function[1]
m: steepness parameter
x_i within [0, PI]

Global min With dimension d and m = 10: 
d = 1: f(2.20) = -0.8013 [1]
d = 2: f(2.20, 1.57) ≈ -1.8013  [2]
d = 5: min = -4.687658  [2]
d = 10: min = -9.66015  [2][3][4]
d = 15: min = -14.66015  [3]
d = 20: min = -19.6370  [4]
d = 25: min = -24.633194  [3]
d = 30: min = -29.6309  [4]
d = 50: min = -49.6248  [4]
d = 75: min = -74.6248  [3]

[1] Michalewicz, Z. (1992). Genetic Algorithms + Data Structures = Evolution Programs. Springer-Verlag
[2] https://www.sfu.ca/~ssurjano/michal.html
[3] Charlie Vanaret, Jean-Baptiste Gotteland, Nicolas Durand, Jean-Marc Alliot. Certified Global Minima for a Benchmark of Difficult Optimization Problems. 2014. ffhal-00996713f
[4] https://doi.org/10.48550/arXiv.2001.11465
"""
class MichalewiczFunction(ContinuousProblem):
    def _michalewicz_function(self, x: FloatVector) -> Float:
        m = 10.0
        i = np.arange(1, x.size + 1)
        term1 = np.sin(x) * (np.sin(i * x**2 / np.pi))**(2 * m)
        return -np.sum(term1)

    def __init__(self, dimension: int, lower_bound: Float = 0.0, upper_bound: Float = np.pi):
        super().__init__(objective_function=self._michalewicz_function, dimension=dimension, lower_bound=lower_bound, upper_bound=upper_bound)

    @override
    def __repr__(self) -> str:
        return f"Michalewicz function: dimension={self.dimension}, bound=[{self.lower_bound}, {self.upper_bound}]"
    
"""
Global Minimum with d dimension: f(-2.903534, -2.903534, ..., -2.903534) ≈ -39.16599*d
x_i within [-5, 5]
"""
class StyblinskiTangFunction(ContinuousProblem):
    def _styblinski_tang_function(self, x: FloatVector) -> Float:
        return 0.5 * np.sum(x**4 - 16*x**2 + 5*x)

    def __init__(self, dimension: int, lower_bound: Float = -5.0, upper_bound: Float = 5.0):
        super().__init__(objective_function=self._styblinski_tang_function, dimension=dimension, lower_bound=lower_bound, upper_bound=upper_bound)

    @override
    def __repr__(self) -> str:
        return f"Styblinski-Tang function: dimension={self.dimension}, bound=[{self.lower_bound}, {self.upper_bound}]"

"""
Global Minimum with d dimension: f(0, 0, ..., 0) = 0
x_i within [-600, 600]
"""
class GriewankFunction(ContinuousProblem):
    def _griewank_function(self, x: FloatVector) -> Float:
        i = np.arange(1, x.size + 1)
        sum_term = np.sum(x**2) / 4000
        prod_term = np.prod(np.cos(x / np.sqrt(i)))
        return sum_term - prod_term + 1.0

    def __init__(self, dimension: int, lower_bound: Float = -600.0, upper_bound: Float = 600.0):
        super().__init__(objective_function=self._griewank_function, dimension=dimension, lower_bound=lower_bound, upper_bound=upper_bound)

    @override
    def __repr__(self) -> str:
        return f"Griewank function: dimension={self.dimension}, bound=[{self.lower_bound}, {self.upper_bound}]"

"""
x_i within [-32.768, 32.768]
Global min: f(0, 0, ..., 0) = 0
config: a = 20, b = 0.2, c = 2 * pi
"""
class AckleyFunction(ContinuousProblem):
    def _ackley_function_generalized(self, x: FloatVector, a: Float = 20, b: Float = 0.2, c: Float = 2 * np.pi) -> Float:
        self.a = a
        self.b = b
        self.c = c
        d = x.size

        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c * x))

        return -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.exp(1)
    
    def _ackley_function(self, x: FloatVector) -> Float:
        return self._ackley_function_generalized(x, a=20, b=0.2, c=2 * np.pi)

    def __init__(self, dimension: int, lower_bound: Float = -32.768, upper_bound: Float = 32.768):
        super().__init__(objective_function=self._ackley_function, dimension=dimension, lower_bound=lower_bound, upper_bound=upper_bound)

    @override
    def __repr__(self) -> str:
        return f"Ackley function: dimension={self.dimension}, bound=[{self.lower_bound}, {self.upper_bound}], a={self.a}, b={self.b}, c={self.c}"