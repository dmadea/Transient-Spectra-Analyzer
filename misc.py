import math
import numpy as np


def find_nearest_idx(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return idx - 1
    else:
        return idx


def find_nearest(array, value):
    idx = find_nearest_idx(array, value)
    return array[idx]


def crop_data(matrix, axis0_data, axis1_data, ax0_0=None, ax0_1=None, ax1_0=None, ax1_1=None):
    ax0_start = find_nearest_idx(axis0_data, ax0_0) if ax0_0 is not None else 0
    ax0_end = find_nearest_idx(axis0_data, ax0_1) + 1 if ax0_1 is not None else matrix.shape[0]

    ax1_start = find_nearest_idx(axis1_data, ax1_0) if ax1_0 is not None else 0
    ax1_end = find_nearest_idx(axis1_data, ax1_1) + 1 if ax1_1 is not None else matrix.shape[1]

    mat_crop = matrix[ax0_start:ax0_end, ax1_start:ax1_end]
    axis0_data_crop = axis0_data[ax0_start:ax0_end]
    axis1_data_crop = axis1_data[ax1_start:ax1_end]

    return mat_crop, axis0_data_crop, axis1_data_crop


def set_axes(plot_item, x_label='', y_label='', title='', grid_x=True, grid_y=True, grid_alpha=0.1):
    plot_item.showAxis('top', show=True)
    plot_item.showAxis('right', show=True)

    plot_item.setLabel('left', text=y_label)
    plot_item.setLabel('bottom', text=x_label)
    plot_item.setTitle(title)
    plot_item.showGrid(x=grid_x, y=grid_y, alpha=grid_alpha)




















