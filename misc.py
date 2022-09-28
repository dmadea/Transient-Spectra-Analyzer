import math
import numpy as np
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QWidget, QSizePolicy, QLabel, QCheckBox

import inspect


def _find_nearest_idx(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return idx - 1
    else:
        return idx


def find_nearest_idx(array, values):
    if not is_iterable(values):
        return _find_nearest_idx(array, values)

    result = []

    for val in values:
        result.append(_find_nearest_idx(array, val))

    return np.asarray(result)


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


def int_default_color(counter):
    colors = [
        (255, 0, 0, 255),  # red
        (0, 255, 0, 255),  # green
        (0, 0, 255, 255),  # blue
        (0, 0, 0, 255),  # black
        (255, 255, 0, 255),  # yellow
        (255, 0, 255, 255),  # magenta
        (0, 255, 255, 255),  # cyan
        (155, 155, 155, 255),  # gray
        (155, 0, 0, 255),  # dark red
        (0, 155, 0, 255),  # dark green
        (0, 0, 155, 255),  # dark blue
        (155, 155, 0, 255),  # dark yellow
        (155, 0, 155, 255),  # dark magenta
        (0, 155, 155, 255)  # dark cyan
    ]

    return QColor(*colors[counter % len(colors)])


def setup_size_policy(ui):
    for comp in inspect.getmembers(ui):
        if comp[0].startswith('__'):
            continue

        if not issubclass(comp[1].__class__, QWidget):
            continue

        c = comp[1]

        c.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

        if isinstance(c, QLabel):
            # print(comp[0])
            c.setWordWrap(True)


def is_iterable(obj):
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True


def str_is_integer(string):
    try:
        int(string)
    except ValueError:
        return False
    else:
        return True














