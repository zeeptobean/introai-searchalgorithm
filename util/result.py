from dataclasses import dataclass, field
from util.define import *
import pickle
import function.continuous_function as contfunc
import function.discrete_function as discfunc

@dataclass
class _Result:
    type: str
    algorithm: str
    iterations: int
    history_info: list[str | None]     # additional info for each iteration
    rng_seed: int
    time: float

@dataclass 
class ContinuousResult(_Result):
    type: str = field(init=False, default="continuous")
    problem: contfunc.ContinuousProblem
    last_x: list[FloatVector]   # x state of the last iteration
    last_value: list[Float]
    best_x: FloatVector         # the best
    best_value: Float
    history_x: list[list[FloatVector]]  # history for each x for each iteration
    history_value: list[list[Float]]

    def __post_init__(self):
        if not (len(self.history_x) == len(self.history_value) and len(self.history_x) == len(self.history_info)):
            raise ValueError("len(history_x) == len(history_value) == len(history_info) is false")

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
            "history_x": [[x.tolist() for x in hist_x] for hist_x in self.history_x],
            "history_value": [[float(v) for v in hist_val] for hist_val in self.history_value],
            "history_info": self.history_info
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
            history_x=[[np.array(x) for x in hist_x] for hist_x in json_dict["history_x"]],
            history_value=[[float(v) for v in hist_val] for hist_val in json_dict["history_value"]],
            history_info=json_dict["history_info"]
        )
    
    @staticmethod
    def from_binary(binary_data: bytes) -> 'ContinuousResult':
        json_dict = pickle.loads(binary_data)
        return ContinuousResult.from_json(json_dict)
    

@dataclass
class DiscreteResult(_Result):
    type: str = field(init=False, default="discrete")
    problem: discfunc.DiscreteProblem
    last_x: list[FloatVector]   # x state of the last iteration
    last_value: list[Float]
    best_x: FloatVector         # the best
    best_value: Float
    history_x: list[list[FloatVector]]  # history for each x for each iteration
    history_value: list[list[Float]]

    def __post_init__(self):
        if not (len(self.history_x) == len(self.history_value) and len(self.history_x) == len(self.history_info)):
            raise ValueError("len(history_x) == len(history_value) == len(history_info) is false")

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
            "history_x": [[x.tolist() for x in hist_x] for hist_x in self.history_x],
            "history_value": [[float(v) for v in hist_val] for hist_val in self.history_value],
            "history_info": self.history_info
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
            history_x=[[np.array(x) for x in hist_x] for hist_x in json_dict["history_x"]],
            history_value=[[float(v) for v in hist_val] for hist_val in json_dict["history_value"]],
            history_info=json_dict["history_info"]
        )
    
    @staticmethod
    def from_binary(binary_data: bytes) -> 'ContinuousResult':
        json_dict = pickle.loads(binary_data)
        return ContinuousResult.from_json(json_dict)