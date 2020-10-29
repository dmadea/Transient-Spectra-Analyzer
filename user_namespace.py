# from __future__ import division

from copy import deepcopy
from misc import find_nearest_idx
from PyQt5.QtWidgets import QApplication

import numpy as np
from gui_console import Console

from logger import Logger

from scipy import fftpack

from LFP_matrix import LFP_matrix
from Widgets.svd_widget import SVDWidget


_matrices = []


def setup_matrix(matrix):
    if UserNamespace.instance is None:
        return

    mw = UserNamespace.instance.main_widget
    mw.setup_matrix(matrix, center_lines=False)


def matrices():
    return _matrices


def clear():
    _matrices.clear()


def register_mat():

    if UserNamespace.instance is None:
        return

    m = UserNamespace.instance.main_widget.matrix.get_factored_matrix()
    #
    # if any(map(lambda item: all(item.Y == m.Y) and all(item.times == m.times) and all(item.wavelengths == m.wavelengths), matrices)):
    #     return

    _matrices.append(m)


def average_matrices(plot_matrix=True):
    if UserNamespace.instance is None:
        return

    if len(_matrices) == 0:
        return

    D_stack = np.stack([m.Y for m in _matrices if m is not None], axis=2)
    D_avrg = D_stack.mean(axis=2, keepdims=False)

    data_avrg = LFP_matrix.from_value_matrix(D_avrg, _matrices[0].times.copy(),
                                             _matrices[0].wavelengths.copy(),
                                             filename=f'{_matrices[0].filename}-avrg.txt',
                                             name=_matrices[0].name)

    if plot_matrix:
        setup_matrix(data_avrg)

    return data_avrg

# import traceback

#
# # variables active in console, but here are useless
# mainWidget = None

# adapted from http://scipy-lectures.org/intro/scipy/auto_examples/solutions/plot_fft_image_denoise.html
# FFT denoising of image, in our case, LFP matrix
def fft_filter(t_dim_fraction=1, w_dim_fraction=1):
    """Applies FFT filter to the data matrix, t_dim_fraction is the fraction of frequencies to preserve for time dimension,
    w_dim_fractions is the fraction of frequencies to preserve for wavelength dimension. Default is 1,
    which reloads the matrix unchanged."""

    if UserNamespace.instance is None:
        return

    A = UserNamespace.instance.main_widget.matrix.Y

    # Set r and c to be the number of rows and columns of the array.
    r, c = A.shape

    # perform a 2D FFT on the matrix
    fft_matrix = fftpack.fft2(A)

    # Set to zero all rows with indices between r*keep_fraction and
    # r*(1-keep_fraction):
    fft_matrix[int(r * t_dim_fraction):int(r * (1 - t_dim_fraction))] = 0
    # Similarly with the columns:
    fft_matrix[:, int(c * w_dim_fraction):int(c * (1 - w_dim_fraction))] = 0

    # reconstruct the matrix (inverse fourier transform) and take the real part
    new_matrix = fftpack.ifft2(fft_matrix).real

    # reload new data
    load_value_matrix(new_matrix)

# def save_fit_MCR(filepath):
#     mw = UserNamespace.instance.main_widget
#
#     save_fit(filepath, A=mw.matrix.A_MCR, C=mw.matrix.C_MCR)


def save_fit(filepath, ST=None, C=None):
    mw = UserNamespace.instance.main_widget

    mw.matrix.save_fit(filepath, ST, C)


def set_gradient(name='sym'):
    mw = UserNamespace.instance.main_widget

    positive_grad = {'ticks': [(0.0, (255, 255, 255, 255)), (1.0, (100, 0, 0, 255)), (0.33, (255, 200, 0, 255)),
                               (0.66, (255, 0, 0, 255))], 'mode': 'rgb'}

    dark = 100
    sym_grad = {'ticks': [(0.0, (0, 0, dark, 255)), (1.0, (dark, 0, 0, 255)), (0.25, (0, 0, 255, 255)),
                          (0.5, (255, 255, 255, 255)), (0.75, (255, 0, 0, 255))], 'mode': 'rgb'}

    mw.plot_widget.hist.gradient.restoreState(positive_grad if name == 'positive' else sym_grad)


def correct_to_time_zero():
    mw = UserNamespace.instance.main_widget
    mw.matrix.times -= mw.matrix.times[0]

    mw.plot_widget.plot_matrix(mw.matrix, center_lines=False)


def load_LFP_matrix(matrix):
    if UserNamespace.instance is None:
        return

    mw = UserNamespace.instance.main_widget
    mw.plot_widget.plot_matrix(matrix, center_lines=False)


def restore_original_data():
    if UserNamespace.instance is None:
        return
    mw = UserNamespace.instance.main_widget

    mw.matrix.restore_original_data()
    mw.plot_widget.plot_matrix(mw.matrix)


#
# def reconstruct_matrix(sing_values_num):
#     if UserNamespace.instance is None:
#         return
#
#     mw = UserNamespace.instance.main_widget
#
#     return mw.reconstruct_matrix(sing_values_num)
#


def load_reconstructed_matrix(sing_values_num):
    if UserNamespace.instance is None:
        return

    mw = UserNamespace.instance.main_widget
    mat = mw.matrix.reconstruct_matrix(sing_values_num)

    mw.plot_widget.plot_matrix(mat, center_lines=False)


def load_sing_value_matrix(singular_value):
    mw = UserNamespace.instance.main_widget
    mat = mw.matrix.reconstruct_matrix_from_sing_value(singular_value)

    mw.plot_widget.plot_matrix(mat, center_lines=False)


def load_value_matrix(value_matrix):
    if UserNamespace.instance is None:
        return

    mw = UserNamespace.instance.main_widget
    new_matrix = LFP_matrix.from_value_matrix(value_matrix, mw.matrix.times, mw.matrix.wavelengths)
    mw.plot_widget.plot_matrix(new_matrix, center_lines=False)


def center_heat_map_levels():
    if UserNamespace.instance is None:
        return

    plot_widget = UserNamespace.instance.main_widget.plot_widget

    z0, z1 = plot_widget.hist.getLevels()
    diff = z1 - z0
    plot_widget.hist.setLevels(-diff / 2, diff / 2)


def transpose_data():
    if UserNamespace.instance is None:
        return

    mw = UserNamespace.instance.main_widget

    if mw.matrix is None:
        return

    mw.matrix.transpose()

    mw.plot_widget.plot_matrix(mw.matrix)


def copy_plot_to_clipboard(plot='heat_map', type='img'):
    if UserNamespace.instance is None:
        return

    plot_widget = UserNamespace.instance.main_widget.plot_widget

    try:
        if type == 'img':
            if plot == 'heat_map':
                plot_widget.save_plot_to_clipboard_as_png(plot_widget.heat_map_layout)
            elif plot == 'spectrum':
                plot_widget.save_plot_to_clipboard_as_png(plot_widget.spectrum)
            elif plot == 'trace':
                plot_widget.save_plot_to_clipboard_as_png(plot_widget.trace)
        elif type == 'svg':
            if plot == 'heat_map':
                plot_widget.save_plot_to_clipboard_as_svg(plot_widget.heat_map_layout)
            elif plot == 'spectrum':
                plot_widget.save_plot_to_clipboard_as_svg(plot_widget.spectrum)
            elif plot == 'trace':
                plot_widget.save_plot_to_clipboard_as_svg(plot_widget.trace)
    except Exception as ex:
        Logger.message(ex.__str__())


def set_spectrum_range(x0=None, x1=None, y0=None, y1=None, padding=None):
    if UserNamespace.instance is None:
        return

    plot_widget = UserNamespace.instance.main_widget.plot_widget

    x_range, y_range = plot_widget.spectrum.getViewBox().viewRange()

    plot_widget.spectrum.getViewBox().setXRange(x_range[0] if x0 is None else x0,
                                                x_range[1] if x1 is None else x1,
                                                padding=padding)

    plot_widget.spectrum.getViewBox().setYRange(y_range[0] if y0 is None else y0,
                                                y_range[1] if y1 is None else y1,
                                                padding=padding)


def set_trace_range(x0=None, x1=None, y0=None, y1=None, padding=None):
    if UserNamespace.instance is None:
        return

    plot_widget = UserNamespace.instance.main_widget.plot_widget

    x_range, y_range = plot_widget.trace.getViewBox().viewRange()

    plot_widget.trace.getViewBox().setXRange(x_range[0] if x0 is None else x0,
                                             x_range[1] if x1 is None else x1,
                                             padding=padding)

    plot_widget.trace.getViewBox().setYRange(y_range[0] if y0 is None else y0,
                                             y_range[1] if y1 is None else y1,
                                             padding=padding)


def set_heat_map_range(x0=None, x1=None, y0=None, y1=None, padding=None):
    if UserNamespace.instance is None:
        return

    plot_widget = UserNamespace.instance.main_widget.plot_widget
    x_range, y_range = plot_widget.heat_map_plot.getViewBox().viewRange()

    plot_widget.heat_map_plot.getViewBox().setXRange(x_range[0] if x0 is None else x0,
                                                     x_range[1] if x1 is None else x1,
                                                     padding=padding)

    plot_widget.heat_map_plot.getViewBox().setYRange(y_range[0] if y0 is None else y0,
                                                     y_range[1] if y1 is None else y1,
                                                     padding=padding)


# def set_range(x0=None, x1=None, y0=None, y1=None, padding=None):
#     set_trace_range(x0, x1, y0, y1, padding)
#     set_spectrum_range(x0, x1, y0, y1, padding)
#     set_heat_map_range(x0, x1, y0, y1, padding)


def set_heat_map_z_range(z0, z1):
    """Sets the levels of the heat map, levels represents the z values of the transient spectra matrix,
    thus change in absorbance."""
    if UserNamespace.instance is None:
        return

    mw = UserNamespace.instance.main_widget

    if not (isinstance(z0, (int, float)) and isinstance(z1, (int, float))):
        raise ValueError("z0 and z1 must be type of int or float.")

    if not z0 < z1:
        raise ValueError("z1 cannot be higher than z0.")

    mw.plot_widget.hist.setLevels(z0, z1)


def set_grid(x=True, y=True, alpha=0.1):
    """Sets the grid of Trace and Spectrum views."""

    if UserNamespace.instance is None:
        return

    mw = UserNamespace.instance.main_widget
    mw.plot_widget.spectrum.showGrid(x=x, y=y, alpha=alpha)
    mw.plot_widget.trace.showGrid(x=x, y=y, alpha=alpha)


def reduce_time_dim(factor=2):
    mw = UserNamespace.instance.main_widget
    mw.matrix.reduce_time_dim(factor)
    load_LFP_matrix(mw.matrix)


def reduce_wavelength_dim(factor=2):
    mw = UserNamespace.instance.main_widget
    mw.matrix.reduce_wavelength_dim(factor)
    load_LFP_matrix(mw.matrix)

def cut_time_dim(t_start, t_end):
    """Cuts the time dimension of the matrix to region [t_start, t_end]"""

    if UserNamespace.instance is None:
        return

    mw = UserNamespace.instance.main_widget

    t_idx_start = find_nearest_idx(mw.matrix.times, t_start)
    t_idx_end = find_nearest_idx(mw.matrix.times, t_end) + 1

    mw.matrix.Y = mw.matrix.Y[t_idx_start:t_idx_end, :]
    mw.matrix.times = mw.matrix.times[t_idx_start:t_idx_end]

    mw.plot_widget.plot_matrix(mw.matrix)


def cut_wavelength_dim(wl_start, wl_end):
    """Cuts the wavelength dimension of the matrix to region [wl_start, wl_end]"""

    if UserNamespace.instance is None:
        return

    mw = UserNamespace.instance.main_widget

    wl_idx_start = find_nearest_idx(mw.matrix.wavelengths, wl_start)
    wl_idx_end = find_nearest_idx(mw.matrix.wavelengths, wl_end) + 1

    mw.matrix.Y = mw.matrix.Y[:, wl_idx_start:wl_idx_end]
    mw.matrix.wavelengths = mw.matrix.wavelengths[wl_idx_start:wl_idx_end]

    mw.plot_widget.plot_matrix(mw.matrix)


def set_spectrum_average_count(num=20):
    """Sets the number of spectra that will averaged and plotted. The spectra will be taken around
    user chosen time t, so num/2 spectra before and after time t will be taken and averaged. Default is 20."""

    if UserNamespace.instance is None:
        return
    mw = UserNamespace.instance.main_widget

    mw.plot_widget.smooth_count = int(num / 2)
    mw.plot_widget.update_trace_and_spectrum()


def SVD():
    if UserNamespace.instance is not None:
        UserNamespace.instance.main_widget.perform_SVD()


#
# def add_to_list(spectra):
#     """
#     Copies all spectra and import them to the treeWidget
#     :param spectra: input parameter can be single spectrum object, or hierarchic list of spectra
#     """
#
#     if UserNamespace.instance is not None:
#         UserNamespace.instance.add_items_to_list(spectra)


def copy_to_clipboard(array, delimiter='\t', decimal_sep='.'):
    if not isinstance(array, (np.ndarray, np.matrix, list, tuple)):
        raise ValueError("Cannot copy {} to clipboard.".format(type(array)))

    try:
        text = '\n'.join(delimiter.join(str(num).replace('.', decimal_sep) for num in row) for row in array)
    except:  # the second dimension is not iterable, we probably got only 1D array, so lets put into clipboard only this
        text = delimiter.join(str(num).replace('.', decimal_sep) for num in array)

    cb = QApplication.clipboard()
    cb.clear(mode=cb.Clipboard)
    cb.setText(text, mode=cb.Clipboard)

    # return "Copied to clipboard"


#
# def test():
#
#     UserNamespace.execute_console_command("display(Math('A\\times e^{-\\frac{t}{\\tau}} + C'))", False)


class UserNamespace:
    instance = None

    def __init__(self, main_widget):
        self.main_widget = main_widget
        UserNamespace.instance = self
        # UserNamespace.setup_variables()
