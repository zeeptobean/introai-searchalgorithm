from dataclasses import dataclass, field
from typing import List, Self, Tuple
from util.define import *
from util.util import HistoryEntry
import pickle
import function.continuous_function as contfunc
import function.discrete_function as discfunc
import function.graph_problem as graphfunc

@dataclass
class _BaseResult():
    type: str
    algorithm: str
    short_name: str
    iterations: int
    rng_seed: int
    time: float
    last_x: list[FloatVector]   # x state of the last iteration
    last_value: list[Float]
    best_x: FloatVector         # the best
    best_value: Float
    history: HistoryEntry
    
    def _format_element(self, v: Float | int) -> str:
        if isinstance(v, int):
            return str(v)
        return f"{float(v):.9f}"
    def _format_list(self, fl: list[Float] | list[int]) -> str:
        arr = [self._format_element(v) for v in fl]
        return "[" + ", ".join(arr) + "]"
    def _format_vector_element(self, fv: FloatVector | IntVector) -> str:
        return "(" + ", ".join(self._format_element(x) for x in fv) + ")"
    def _format_vector_element_list(self, fl: list[FloatVector] | list[IntVector]) -> str:
        result: list[str] = []
        for v in fl:
            result.append(self._format_vector_element(v))
        return "[" + " , ".join(result) + "]"

    def __post_init__(self):
        if not (len(self.history.history_x) == len(self.history.history_value) and len(self.history.history_x) == len(self.history.history_info)):
            raise ValueError("len(history_x) == len(history_value) == len(history_info) is false")
        
    def to_binary(self) -> bytes:
        return pickle.dumps(self)
    
    @classmethod
    def from_binary(cls, binary_data: bytes) -> Self:
        return pickle.loads(binary_data)
        
    def __repr__(self) -> str:
        format_history_value = []
        for ele in self.history.history_value:
            format_history_value.append(self._format_list(ele))
        format_history_x = []
        for ele in self.history.history_x:
            format_history_x.append(self._format_vector_element_list(ele))

        history_str = ""
        for i, (x, val, info) in enumerate(zip(format_history_x, format_history_value, self.history.history_info)):
            info_str = f"info = {info};" if info else ""
            history_str += f"Ite {i}: x = {x}; value = {val}; {info_str}\n"

        format_str = f"""
Algorithm: {self.algorithm}
Short name: {self.short_name}
Running time: {self.time} ms
Iterations: {self.iterations}
RNG Seed: {self.rng_seed}
Best solution: x = {self._format_vector_element(self.best_x)}, value = {self._format_element(self.best_value)}
Last solution: x = {self._format_vector_element_list(self.last_x)}, value = {self._format_list(self.last_value)}
History:
{history_str}
"""
        return format_str
    
    def to_json_simple(self) -> dict:
        """A simplified version of to_json, only include the most important information."""
        return {
            "algorithm": self.algorithm,
            "iterations": self.iterations,
            "time": self.time,
            "rng_seed": self.rng_seed,
            "best_x": self.best_x.tolist(),
            "best_value": float(self.best_value)
        }
    
    def to_json(self) -> dict:
        remdict = {
            "last_x": [x.tolist() for x in self.last_x],
            "last_value": [float(v) for v in self.last_value],
            "history": self.history.to_json()
        }
        return self.to_json_simple() | remdict
    
    @classmethod
    def from_json(cls, json_dict: dict) -> dict:
        return {
            "algorithm": json_dict["algorithm"],
            "short_name": json_dict["short_name"],
            "iterations": json_dict["iterations"],
            "time": json_dict["time"],
            "rng_seed": json_dict["rng_seed"],
            "best_x": np.array(json_dict["best_x"]),
            "best_value": json_dict["best_value"],
            "last_x": [np.array(x) for x in json_dict["last_x"]],
            "last_value": [float(v) for v in json_dict["last_value"]],
            "history": HistoryEntry.from_json(json_dict["history"]),
        }


@dataclass 
class ContinuousResult(_BaseResult):
    type: str = field(init=False, default="continuous")
    problem: contfunc.ContinuousProblem

    def __repr__(self) -> str:
        basestr = super().__repr__()
        format_str = f"""
{basestr}
Problem: {self.problem}
"""
        return format_str
    
    def to_json_simple(self) -> dict:
        remdict = {
            "problem": self.problem.to_json()
        }
        return remdict | super().to_json_simple()
    
    @classmethod
    def from_json(cls, json_dict: dict) -> 'ContinuousResult':
        shared_data = super().from_json(json_dict)
        problem_json = json_dict["problem"]
        problem = contfunc.ContinuousProblem.from_json(problem_json)

        return cls(problem=problem, **shared_data)
    

@dataclass
class DiscreteResult(_BaseResult):
    type: str = field(init=False, default="discrete")
    problem: discfunc.DiscreteProblem
    
    def __repr__(self) -> str:
        basestr = super().__repr__()
        format_str = f"""
{basestr}
Problem: {self.problem}
"""
        return format_str
    
    def to_json_simple(self) -> dict:
        remdict = {
            "problem": self.problem.to_json()
        }
        return remdict | super().to_json_simple()


@dataclass
class SearchResult():
    """
    Specialized result class for pathfinding algorithms (BFS, DFS, A*).
    """
    type: str = field(init=False, default="pathfinding")
    algorithm: str
    short_name: str
    nodes_expanded: int
    time: float
    problem: graphfunc.GridWorldProblem
    path: List[Tuple[int, int]]
    cost: float
    
    @property
    def path_length(self) -> int:
        """Number of nodes in the path"""
        return len(self.path)
    
    @property
    def success(self) -> bool:
        """Whether a path was found"""
        return self.cost != float('inf')
    
    def _format_path(self) -> str:
        """Format path for display"""
        if len(self.path) <= 10:
            return " -> ".join(str(p) for p in self.path)
        else:
            start_part = " -> ".join(str(p) for p in self.path[:3])
            end_part = " -> ".join(str(p) for p in self.path[-3:])
            return f"{start_part} -> ... ({len(self.path)-6} nodes) ... -> {end_part}"
    
    def __repr__(self) -> str:
        format_str = f"""
Algorithm: {self.algorithm}
Problem: {self.problem}
Running time: {self.time} ms
Nodes explored: {self.nodes_expanded}
Is success: {self.success}
"""
        if self.success:
            format_str2 = f"""
{format_str}
Path length: {self.path_length}
Path cost: {self.cost:.6f}
Path: {self._format_path()}
"""
            return format_str2
        return format_str
    
    def to_json(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "short_name": self.short_name,
            "problem": self.problem.to_json(),
            "path": [list(p) for p in self.path] if self.path else None,
            "path_length": self.path_length,
            "path_cost": float(self.cost),
            "nodes_expanded": self.nodes_expanded,
            "time": self.time,
            "success": self.success,
        }
    
    @staticmethod
    def from_json(json_data: dict) -> 'SearchResult':
        problem = graphfunc.GridWorldProblem.from_json(json_data["problem"])
        path = [tuple(p) for p in json_data["path"]]
        return SearchResult(
            algorithm=json_data["algorithm"],
            short_name=json_data["short_name"],
            problem=problem,
            path=path,
            cost=json_data["path_cost"],
            nodes_expanded=json_data["nodes_expanded"],
            time=json_data["time"]
        )
    
    def to_binary(self) -> bytes:
        return pickle.dumps(self)
    
    @staticmethod
    def from_binary(binary_data: bytes) -> 'SearchResult':
        return pickle.loads(binary_data)