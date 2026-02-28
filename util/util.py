import numpy as np
from util.define import *

class HistoryEntry:
    def __init__(self, is_max_value_problem: bool = False):
        self.is_max_value_problem = is_max_value_problem
        self.history_x: list[list[FloatVector]] = []
        self.history_value: list[list[Float]] = []
        self.history_info: list[str | None] = []

    def add(self, x: list[FloatVector], value: list[Float], info: str | None = None):
        self.history_x.append(x)
        if self.is_max_value_problem:
            value = [-v for v in value]
        self.history_value.append(value)
        self.history_info.append(info)

    def get_best_value(self) -> tuple[FloatVector, Float]:
        if self.is_max_value_problem:
            value_array = np.array(self.history_value)
            max_val = np.max(value_array)
            row, col = np.unravel_index(np.argmax(value_array), value_array.shape)
            max_x = self.history_x[row][col]
            return max_x, max_val
        else:
            return self._get_min_value()

    def _get_min_value(self) -> tuple[FloatVector, Float]:
        val_array = np.array(self.history_value)
        min_val = np.min(val_array)
        row, col = np.unravel_index(np.argmin(val_array), val_array.shape)
        min_x = self.history_x[row][col]
        return min_x, min_val
    
    def to_json(self) -> dict:
        return {
            "history_x": [[x.tolist() for x in hist_x] for hist_x in self.history_x],
            "history_value": [[float(v) for v in hist_val] for hist_val in self.history_value],
            "history_info": self.history_info
        }
    
    @staticmethod
    def from_json(json_dict: dict) -> 'HistoryEntry':
        history = HistoryEntry()
        history.history_x = [[np.array(x) for x in hist_x] for hist_x in json_dict["history_x"]]
        history.history_value = [[float(v) for v in hist_val] for hist_val in json_dict["history_value"]]
        history.history_info = json_dict["history_info"]
        return history

def get_min_2d_impl2(x_arr: list[list[FloatVector]], value_arr: list[list[Float]]) -> tuple[FloatVector, Float]:
    row, col = 0, 0
    min_val = Float('inf')
    for i in range(len(value_arr)):
        for j in range(len(value_arr[i])):
            if value_arr[i][j] < min_val:
                min_val = value_arr[i][j]
                row, col = i, j
    min_x = x_arr[row][col]
    return min_x, min_val


def get_min_2d(x_arr: list[list[FloatVector]], value_arr: list[list[Float]]) -> tuple[FloatVector, Float]:
    val_array = np.array(value_arr)
    min_val = np.min(val_array)
    row, col = np.unravel_index(np.argmin(val_array), val_array.shape)
    min_x = x_arr[row][col]
    return min_x, min_val