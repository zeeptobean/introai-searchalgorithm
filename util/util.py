import numpy as np
from util.define import *

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