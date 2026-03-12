from dataclasses import dataclass, field
from typing import List, Tuple, TYPE_CHECKING
from util.define import *
from util.util import HistoryEntry
import pickle
import function.continuous_function as contfunc
import function.discrete_function as discfunc

if TYPE_CHECKING:
    import function.graph_problem as graphfunc

@dataclass
class Result:
    type: str
    algorithm: str
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

@dataclass 
class ContinuousResult(Result):
    type: str = field(init=False, default="continuous")
    problem: contfunc.ContinuousProblem

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
        for ele in self.history.history_value:
            format_history_value.append(self._format_npfloat_list(ele))
        format_history_x = []
        for ele in self.history.history_x:
            format_history_x.append(self._format_floatvector_list(ele))

        history_str = ""
        for i, (x, val, info) in enumerate(zip(format_history_x, format_history_value, self.history.history_info)):
            info_str = f"info = {info};" if info else ""
            history_str += f"Ite {i}: x = {x}; value = {val}; {info_str}\n"

        format_str = f"""
Algorithm: {self.algorithm}
Problem: {self.problem}
Running time: {self.time} ms
Iterations: {self.iterations}
RNG Seed: {self.rng_seed}
Best solution: x = {self._format_floatvector(self.best_x)}, value = {self._format_float(self.best_value)}
Last solution: x = {self._format_floatvector_list(self.last_x)}, value = {self._format_npfloat_list(self.last_value)}
History:
{history_str}
"""
        return format_str
    
    def to_json_simple(self) -> dict:
        """A simplified version of to_json, only include the most important information."""
        return {
            "algorithm": self.algorithm,
            "problem": self.problem.to_json(),
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
    
    def to_binary(self) -> bytes:
        return pickle.dumps(self.to_json())
    
    @staticmethod
    def from_json(json_dict: dict) -> 'ContinuousResult':
        problem_json = json_dict["problem"]
        problem = contfunc.ContinuousProblem.from_json(problem_json)

        return ContinuousResult(
            algorithm=json_dict["algorithm"],
            problem=problem,
            iterations=json_dict["iterations"],
            time=json_dict["time"],
            rng_seed=json_dict["rng_seed"],
            best_x=np.array(json_dict["best_x"]),
            best_value=json_dict["best_value"],
            last_x=[np.array(x) for x in json_dict["last_x"]],
            last_value=[float(v) for v in json_dict["last_value"]],
            history=HistoryEntry.from_json(json_dict["history"]),
        )
    
    @staticmethod
    def from_binary(binary_data: bytes) -> 'ContinuousResult':
        json_dict = pickle.loads(binary_data)
        return ContinuousResult.from_json(json_dict)
    

@dataclass
class DiscreteResult(Result):
    type: str = field(init=False, default="discrete")
    problem: discfunc.DiscreteProblem
    
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
Problem: {self.problem}
Running time: {self.time} ms
Iterations: {self.iterations}
RNG Seed: {self.rng_seed}
Best solution: x = {self._format_vector_element(self.best_x)}, value = {self._format_element(self.best_value)}
Last solution: x = {self._format_vector_element_list(self.last_x)}, value = {self._format_list(self.last_value)}
History:
{history_str}
"""
        return format_str
    
    def to_binary(self) -> bytes:
        return pickle.dumps(self)
    
    @staticmethod
    def from_binary(binary_data: bytes) -> 'DiscreteResult':
        return pickle.loads(binary_data)


@dataclass
class SearchResult(Result):
    """
    Result class for pathfinding algorithms (BFS, DFS, A*).
    Inherits from Result for consistency with other algorithm results.
    
    Note: All parent fields (type, algorithm, iterations, rng_seed, time, last_x, last_value, best_x, best_value, history)
    must be provided. Additional pathfinding-specific field: problem, path.
    """
    problem: 'graphfunc.GraphProblem'
    path: List[Tuple[int, int]] | None
    
    def __post_init__(self):
        # Set type to "pathfinding"
        object.__setattr__(self, 'type', 'pathfinding')
        
        # Convert path to vector format for best_x (required by Result base class)
        if self.path:
            path_vector = self.problem.path_to_vector(self.path)
            object.__setattr__(self, 'best_x', path_vector)
            object.__setattr__(self, 'last_x', [path_vector])
        else:
            # No path found - use empty vector
            object.__setattr__(self, 'best_x', np.array([], dtype=int))
            object.__setattr__(self, 'last_x', [np.array([], dtype=int)])
        
        # last_value is same as best_value for pathfinding (deterministic)
        object.__setattr__(self, 'last_value', [self.best_value])
        
        # Ensure rng_seed is 0 for deterministic pathfinding
        if self.rng_seed != 0:
            object.__setattr__(self, 'rng_seed', 0)
        
        # Call parent's __post_init__ to validate history
        super().__post_init__()
    
    @property
    def path_length(self) -> int:
        """Number of nodes in the path"""
        return len(self.path) if self.path else 0
    
    @property
    def success(self) -> bool:
        """Whether a path was found"""
        return self.path is not None and self.best_value != float('inf')
    
    @property
    def path_cost(self) -> Float:
        """Alias for best_value (the cost of the found path)"""
        return self.best_value
    
    @property
    def nodes_explored(self) -> int:
        """Alias for iterations (number of nodes explored)"""
        return self.iterations
    
    def _format_path(self) -> str:
        """Format path for display"""
        if not self.path:
            return "No path found"
        
        if len(self.path) <= 10:
            return " -> ".join(str(p) for p in self.path)
        else:
            start_part = " -> ".join(str(p) for p in self.path[:3])
            end_part = " -> ".join(str(p) for p in self.path[-3:])
            return f"{start_part} -> ... ({len(self.path)-6} nodes) ... -> {end_part}"
    
    def __repr__(self) -> str:
        # Format history
        history_str = ""
        max_history_display = 10  # Only show first 10 entries in repr
        for i, info in enumerate(self.history.history_info[:max_history_display]):
            value = self.history.history_value[i][0] if self.history.history_value[i] else float('inf')
            history_str += f"Ite {i}: value = {value:.4f}; info = {info}\n"
        
        if len(self.history.history_info) > max_history_display:
            history_str += f"... ({len(self.history.history_info) - max_history_display} more entries)\n"
        
        format_str = f"""
Algorithm: {self.algorithm}
Problem: {self.problem}
Running time: {self.time} ms
Nodes explored: {self.iterations}
RNG Seed: {self.rng_seed}
Success: {self.success}
Path length: {self.path_length}
Path cost: {self.best_value:.6f}
Path: {self._format_path()}
History entries: {len(self.history.history_x)}
History (first {min(max_history_display, len(self.history.history_info))} entries):
{history_str}
"""
        return format_str
    
    def to_json(self) -> dict:
        """Serialize to JSON format"""
        return {
            "type": "pathfinding",
            "algorithm": self.algorithm,
            "problem": self.problem.to_json(),
            "path": [list(p) for p in self.path] if self.path else None,
            "path_length": self.path_length,
            "path_cost": float(self.best_value),
            "nodes_explored": self.iterations,
            "time": self.time,
            "rng_seed": self.rng_seed,
            "success": self.success,
            "history": self.history.to_json()
        }
    
    def to_json_simple(self) -> dict:
        """Simplified JSON without history"""
        return {
            "type": "pathfinding",
            "algorithm": self.algorithm,
            "problem": self.problem.to_json(),
            "path": [list(p) for p in self.path] if self.path else None,
            "path_length": self.path_length,
            "path_cost": float(self.best_value),
            "nodes_explored": self.iterations,
            "time": self.time,
            "rng_seed": self.rng_seed,
            "success": self.success
        }
    
    @staticmethod
    def from_json(json_data: dict) -> 'SearchResult':
        """Reconstruct SearchResult from JSON"""
        import function.graph_problem as graphfunc
        
        problem = graphfunc.GraphProblem.from_json(json_data["problem"])
        path = [tuple(p) for p in json_data["path"]] if json_data["path"] else None
        history = HistoryEntry.from_json(json_data["history"]) if "history" in json_data else HistoryEntry()
        
        return SearchResult(
            type="pathfinding",
            algorithm=json_data["algorithm"],
            iterations=json_data["nodes_explored"],
            rng_seed=json_data.get("rng_seed", 0),
            time=json_data["time"],
            last_x=[],  # Will be set in __post_init__
            last_value=[],  # Will be set in __post_init__
            best_x=np.array([]),  # Will be set in __post_init__
            best_value=json_data["path_cost"],
            history=history,
            problem=problem,
            path=path
        )
    
    def to_binary(self) -> bytes:
        return pickle.dumps(self.to_json())
    
    @staticmethod
    def from_binary(binary_data: bytes) -> 'SearchResult':
        json_data = pickle.loads(binary_data)
        return SearchResult.from_json(json_data)
    
    def visualize_path(self) -> str:
        """Visualize the path on the grid"""
        return self.problem.visualize_path(self.path)