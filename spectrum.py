import numpy as np
import os

from copy import deepcopy
import math

from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

# import cythonmethods


from logger import Logger


class CalcMode:
    add_sub = 0
    mul_div = 1


class Spectrum(object):
    """
    Class that hold the spectrum object as a 2D array (n, 2) where n is a number of points in spectrum and includes some
    functions and calculation with it. Spectrum is stored as numpy array.
    """

    @classmethod
    def from_xy_values(cls, x_values, y_values, name='', group_name='', filepath=None, color=None, line_width=None,
                       line_type=None):
        try:
            if len(x_values) != len(y_values):
                raise ValueError("Length of x_values and y_values must match")

            x_data = np.asarray(x_values, dtype=np.float64)
            y_data = np.asarray(y_values, dtype=np.float64)
        except ValueError:
            raise
        except:
            raise ValueError(
                "Argument error, x_values and y_values must be type of (list, tuple, ndarray, range, ..., generally iterable) and contain numbers.")

        data = np.vstack((x_data, y_data)).T

        return cls(data, filepath=filepath, name=name, group_name=group_name, color=color, line_width=line_width,
                   line_type=line_type)

    def __init__(self, data, filepath=None, name='', group_name='', color=None, line_width=None, line_type=None):
        # sort according to first column,numpy matrix, 1. column wavelength, 2. column absorbance
        # order = 'C' - store data in memory row-vise
        self.data = np.asarray(data[data[:, 0].argsort()], dtype=np.float64)
        # self.data = data[data[:, 0].argsort()]
        # self.original_data = self.data.copy()  # this variable stores original data unchanged by any operation
        self.filepath = filepath
        self.name = name
        self.group_name = '' if group_name is None else group_name

        self.color = color
        self.line_width = line_width
        self.line_type = line_type

    def set_style(self, color=None, line_width=None, line_type=None):
        if color is not None:
            self.color = color
        if line_width is not None:
            self.line_width = line_width
        if line_type is not None:
            self.line_type = line_type

    def set_default_style(self):
        self.color = None
        self.line_width = None
        self.line_type = None

    def x_values(self):
        return self.data[:, 0]

    def y_values(self):
        return self.data[:, 1]

    def length(self):
        return self.data.shape[0]

    def x_min(self):
        return self.data[0, 0]

    def x_max(self):
        return self.data[-1, 0]

    def add_to_list(self):
        """
        Imports spectra to the Treewidget
        """
        from user_namespace import add_to_list
        add_to_list(self)

    # def restore_original_data(self):
    #     self.data = self.original_data.copy()

    def __str__(self, separator='\t', decimal_sep='.', new_line='\n', include_header=True):
        """To string method"""
        buffer = "Wavelength" + separator + self.name + new_line if include_header else ""
        buffer += new_line.join(
            separator.join(Spectrum.float_to_string(num, decimal_sep) for num in row) for row in self.data)
        return buffer

    def fit_curve(self, x0, x1, model='exp', print_data=True, add_spectra_to_list=True, bounds=None):

        # from user_namespace import UserNamespace

        start_idx = Spectrum.find_nearest_idx(self.data[:, 0], x0)
        end_idx = Spectrum.find_nearest_idx(self.data[:, 0], x1) + 1

        x_data = self.data[start_idx:end_idx, 0]
        y_data = self.data[start_idx:end_idx, 1]

        x_start = x_data[0]
        y_start = y_data[0]

        if model == 'A-B-tau':
            def func(x, A, tau, y0):
                return A * np.exp(-x / tau) + y0

            # initial parameters, A, tau, y0
            p0 = (y_start, 1, 0)
            if bounds is None:
                bounds = ([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf])

        elif model == 'A-B-k':
            def func(x, A, k, y0):
                return A * np.exp(-k * x) + y0

            # initial parameters, A, tau, y0
            p0 = (y_start, 1, 0)
            if bounds is None:
                bounds = ([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf])

        elif model == 'A-B-C(B)':
            def func(x, A, k1, k2, y0):
                return A * (k1 / (k2 - k1)) * (np.exp(-k1 * x) - np.exp(-k2 * x)) + y0

            # initial parameters A, k1, k2, y0
            p0 = (max(y_data), 1, 0.5, 0)
            if bounds is None:
                bounds = ([-np.inf, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf])

        elif model == 'A-B-C(A+B)':
            def func(x, A1, A2, k1, k2, y0):
                return A1 * np.exp(-k1 * x) + A2 * (k1 / (k2 - k1)) * (
                        np.exp(-k1 * x) - np.exp(-k2 * x)) + y0

            # initial parameters A1, A2, k1, k2, y0
            p0 = (0, max(y_data), 1, 0.5, 0)
            if bounds is None:
                bounds = ([-np.inf, -np.inf, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf])

        popt, pcov = curve_fit(func, x_data, y_data, p0=p0, bounds=bounds)

        y_fit_data = func(x_data, *popt)
        y_residuals = y_data - y_fit_data

        sp_fit = Spectrum.from_xy_values(x_data, y_fit_data, name="Fit of {}".format(self.name))
        sp_residuals = Spectrum.from_xy_values(x_data, y_residuals, name="Residuals of {}".format(self.name))

        if print_data:
            if model == 'A-B-tau':
                Logger.console_message("Initial params: A = {:.3g}, tau = {:.4g}, y0 = {:.3g}".format(*p0))
                Logger.console_message("Fit params:     A = {:.3g}, tau = {:.4g}, y0 = {:.3g}".format(*popt))
            elif model == 'A-B-k':
                Logger.console_message("Initial params: A = {:.3g}, k = {:.4g}, y0 = {:.3g}".format(*p0))
                Logger.console_message("Fit params:     A = {:.3g}, k = {:.4g}, y0 = {:.3g}".format(*popt))
            elif model == 'A-B-C(B)':
                # prinx("fitting equation")
                # UserNamespace.execute_console_command("Math(\"A\\times e^{-\\frac{t}{\\tau}} + C\")", False)
                Logger.console_message("Initial params: A = {:.3g}, k1 = {:.4g}, k2 = {:.4g}, y0 = {:.3g}".format(*p0))
                Logger.console_message(
                    "Fit params:     A = {:.3g}, k1 = {:.4g}, k2 = {:.4g}, y0 = {:.3g}".format(*popt))
            elif model == 'A-B-C(A+B)':
                Logger.console_message(
                    "Initial params: A1 = {:.3g}, A2 = {:.3g}, k1 = {:.4g}, k2 = {:.4g}, y0 = {:.3g}".format(*p0))
                Logger.console_message(
                    "Fit params:     A1 = {:.3g}, A2 = {:.3g}, k1 = {:.4g}, k2 = {:.4g}, y0 = {:.3g}".format(*popt))

        if add_spectra_to_list:
            sp_fit.add_to_list()
            sp_residuals.add_to_list()

        if print_data:
            return popt

        return popt, pcov, sp_fit, sp_residuals

    def savitzky_golay(self, window_length, poly_order):
        """Applies Savitysky-Golay filter to a spectrum."""
        window_length = int(window_length)
        poly_order = int(poly_order)

        if poly_order < 1:
            raise ValueError("Polynomial order must be > 0.")

        if poly_order >= window_length:
            raise ValueError("Polynomial order must be less than window_length.")

        # window_length must be odd number
        if window_length % 2 != 1:
            window_length += 1

        self.data[:, 1] = savgol_filter(self.data[:, 1], window_length, poly_order)

    def integrate(self, x0=None, x1=None):
        """Integrate current spectrum at x range [x0, x1] by trapezoidal integration method. If x0, x1 are None,
        all spectrum will be integrated. This will integrate also unevenly spaced data."""

        start_idx = 0
        end_idx = self.data.shape[0]  # not exactly end index, this index will not be included in the calculation

        if x0 is not None and x1 is not None:
            if x0 >= x1:
                raise ValueError("Argument error, x0 ({}) cannot be larger or equal than x1 ({}).".format(x0, x1))

            start_idx = Spectrum.find_nearest_idx(self.data[:, 0], x0)
            end_idx = Spectrum.find_nearest_idx(self.data[:, 0], x1) + 1

        # perform trapezodic integration, assuming the data are unevenly distributed,
        # so no simplification in computation, except the division by 2 at the end
        sum = 0.0
        for i in range(start_idx + 1, end_idx):
            dx = self.data[i, 0] - self.data[i - 1, 0]
            sum += (self.data[i, 1] + self.data[i - 1, 1]) * dx

        sum /= 2

        return sum

    def baseline_correct(self, x0, x1):
        """Subtracts the average of y data from the range of x values [x0, x1] from y values."""

        if x0 > x1:
            raise ValueError("Argument error, x0 ({}) cannot be larger than x1 ({}).".format(x0, x1))

        # get nearby indexes of user defined x values
        start_idx = Spectrum.find_nearest_idx(self.data[:, 0], x0)
        end_idx = Spectrum.find_nearest_idx(self.data[:, 0], x1) + 1

        # calculate the average of y values over the selected range
        avrg = np.average(self.data[start_idx:end_idx, 1])

        # subtract the average from y values
        self.data[:, 1] -= avrg

    def normalize(self, x0, x1):
        """Finds an y maximun at specified boundaries range [x0, x1] and divide the y values by this maximum."""

        if x0 > x1:
            raise ValueError("Argument error, x0 ({}) cannot be larger than x1 ({}).".format(x0, x1))

        # get nearby indexes of user defined x values
        start_idx = Spectrum.find_nearest_idx(self.data[:, 0], x0)
        end_idx = Spectrum.find_nearest_idx(self.data[:, 0], x1) + 1

        y_max = np.max(self.data[start_idx:end_idx, 1])

        # normalize y values
        self.data[:, 1] /= y_max

    def resample(self, spacing):
        """Resample the current spectrum by spacing value. Algorithm uses linear approximation.
         Eg. spectrum is unevenly distributed by strange x values. If spacing will be 1, the
         resulting spectrum will be spaced by 1 nm"""

        data_min = self.data[0, 0]
        data_max = self.data[-1, 0]

        if spacing > data_max - data_min:
            raise ValueError("Spacing ({}) cannot be larger that data itself. "
                             "Resample method, spectrum {}.".format(spacing, self.name))

        # TODO--> cythonmethods.resample function somehow does not work in some cases, cy_resample is used instead
        # try:
        #     # optimized compiled algorithm, 100 times faster than python version,
        #     out = cythonmethods.cy_resample(self.data, spacing)
        # except Exception as ex:
        #     raise ValueError(ex.__str__())
        #
        # self.data = out

        x_min = spacing * int(np.round(data_min / spacing, 0))
        x_max = spacing * int(np.round(data_max / spacing, 0))

        # length of array of new data
        n = int((x_max - x_min) / spacing + 1)

        lin_space = np.linspace(x_min, x_max, num=n)

        # define moovable indexes in original data array
        idx_p1 = 0
        idx_p2 = 1

        outputData = np.zeros((2, n))
        outputData[0] = lin_space

        d_max_idx = self.data.shape[0] - 1

        def y(y1, y2, x1, x2, x):
            return (y2 - y1) * (x - x2) / (x2 - x1) + y2

        for i, val in enumerate(lin_space):
            # move both indexes in data array so that it satisfies condition xdata[idx_p1] <= val < xdata[idx_p2]
            while idx_p1 < d_max_idx - 1 and self.data[idx_p1 + 1, 0] <= val:
                idx_p1 += 1

            while idx_p2 < d_max_idx and self.data[idx_p2, 0] <= val:
                idx_p2 += 1

            outputData[1, i] = y(self.data[idx_p1, 1], self.data[idx_p2, 1],
                                 self.data[idx_p1, 0], self.data[idx_p2, 0], val)

        self.data = outputData.T  # transpose

    def cut(self, x0, x1):
        """Cuts the spectrum to range [x0, x1]"""
        if x0 >= x1:
            raise ValueError("Argument error, x0 ({}) cannot be larger or equal than x1 ({}).".format(x0, x1))

        # get nearby indexes of user defined x values
        start_idx = Spectrum.find_nearest_idx(self.data[:, 0], x0)
        end_idx = Spectrum.find_nearest_idx(self.data[:, 0], x1) + 1

        if start_idx + 1 == end_idx:
            raise ValueError(
                "Oh... someone likes to cut a lot, but with these input parameters, resulting spectrum would have 1 point :-). Unfortunately, cannot perform cut operation.")

        self.data = self.data[start_idx:end_idx, :]

    def extend_by_zeros(self, x0, x1):
        """
        Enter the new x range [x0, x1]. Spectrum will be extended by zeros.
        """
        if x0 >= x1:
            raise ValueError("Argument error, x0 ({}) cannot be larger or equal than x1 ({}).".format(x0, x1))

        x_min = self.x_min()
        x_max = self.x_max()

        # get nearby indexes of user defined x values
        start_idx = Spectrum.find_nearest_idx(self.data[:, 0], x0)
        end_idx = Spectrum.find_nearest_idx(self.data[:, 0], x1) + 1

        if start_idx != 0 and end_idx != self.data.shape[0]:
            raise ValueError("Nothing to extend.")

        x_dif = x_max - x_min
        spacing = x_dif / (self.data.shape[0] - 1)

        new_x_min = spacing * int(np.round(x0 / spacing, 0))

        min_stack = None
        try:
            num_min = int((x_min - new_x_min) / spacing + 1)
            min_lin_space = np.linspace(new_x_min, x_min, num=num_min)

            min_stack = np.zeros((2, num_min - 1))
            min_stack[0] = min_lin_space[:-1]
            min_stack = min_stack.T
        except ValueError:
            pass

        max_stack = None
        try:
            new_x_max = spacing * int(np.round(x1 / spacing, 0))

            num_max = int((new_x_max - x_max) / spacing + 1)
            max_lin_space = np.linspace(x_max, new_x_max, num=num_max)

            max_stack = np.zeros((2, num_max - 1))
            max_stack[0] = max_lin_space[1:]
            max_stack = max_stack.T
        except ValueError:
            pass

        if min_stack is not None and max_stack is not None:
            result = np.vstack((min_stack, self.data, max_stack))
        elif min_stack is not None:
            result = np.vstack((min_stack, self.data))
        else:
            result = np.vstack((self.data, max_stack))

        self.data = result

    def _get_data(self, other, mode=CalcMode.add_sub):

        if not isinstance(other, Spectrum) and not (isinstance(other, float) or isinstance(other, int)):
            raise ValueError("Cannot perform calculation of Spectrum with {}".format(type(other)))

        shape = self.data.shape[0]

        # another spectrum
        if isinstance(other, Spectrum):
            if other.data.shape[0] != shape:
                raise ValueError(
                    "Spectra \'{}\' and \'{}\' have not the same length (dimension). "
                    "Unable to perform calculation.".format(self.name, other.name))

            y_data = other.data[:, 1]

        else:  # number
            y_data = np.full(shape, other, dtype=np.float64)

        x_data = np.zeros(shape, dtype=np.float64) if mode == CalcMode.add_sub else np.full(shape, 1, dtype=np.float64)
        return np.vstack((x_data, y_data)).T

    def __add__(self, other):
        if isinstance(other, SpectrumList):
            return other.__radd__(self)
        other_data = self._get_data(other, mode=CalcMode.add_sub)
        ret_data = self.data + other_data
        name = "{} + {}".format(self.name, other.name if isinstance(other, Spectrum) else other)
        return Spectrum(ret_data, name=name)

    def __sub__(self, other):
        if isinstance(other, SpectrumList):
            return other.__rsub__(self)
        other_data = self._get_data(other, mode=CalcMode.add_sub)
        ret_data = self.data - other_data
        name = "{} - {}".format(self.name, other.name if isinstance(other, Spectrum) else other)
        return Spectrum(ret_data, name=name)

    def __mul__(self, other):
        if isinstance(other, SpectrumList):
            return other.__rmul__(self)
        other_data = self._get_data(other, mode=CalcMode.mul_div)
        ret_data = self.data * other_data
        name = "{} * {}".format(self.name, other.name if isinstance(other, Spectrum) else other)
        return Spectrum(ret_data, name=name)

    def __truediv__(self, other):
        if isinstance(other, SpectrumList):
            return other.__rtruediv__(self)
        other_data = self._get_data(other, mode=CalcMode.mul_div)
        ret_data = self.data / other_data
        name = "{} / {}".format(self.name, other.name if isinstance(other, Spectrum) else other)
        return Spectrum(ret_data, name=name)

    def __radd__(self, other):
        other_data = self._get_data(other, mode=CalcMode.add_sub)
        ret_data = other_data + self.data
        name = "{} + {}".format(other.name if isinstance(other, Spectrum) else other, self.name)
        return Spectrum(ret_data, name=name)

    def __rsub__(self, other):
        if not (isinstance(other, float) or isinstance(other, int)):
            raise ValueError("Cannot perform calculation of Spectrum with {}".format(type(other)))
        ret_data = self.data.copy()
        ret_data[:, 1] = other - ret_data[:, 1]
        name = "{} - {}".format(other, self.name)
        return Spectrum(ret_data, name=name)

    def __rmul__(self, other):
        other_data = self._get_data(other, mode=CalcMode.mul_div)
        ret_data = other_data * self.data
        name = "{} * {}".format(other.name if isinstance(other, Spectrum) else other, self.name)
        return Spectrum(ret_data, name=name)

    def __rtruediv__(self, other):
        if not (isinstance(other, float) or isinstance(other, int)):
            raise ValueError("Cannot perform calculation of Spectrum with {}".format(type(other)))
        ret_data = self.data.copy()
        ret_data[:, 1] = other / ret_data[:, 1]
        name = "{} / {}".format(other, self.name)
        return Spectrum(ret_data, name=name)

    @staticmethod
    def get_name_of_file(filepath):
        if filepath is None:
            return ''

        tail = os.path.split(filepath)[1]
        return os.path.splitext(tail)[0]  # without extension

    @staticmethod
    def float_to_string(float_number, decimal_sep='.'):
        return str(float_number).replace('.', decimal_sep)

    # TODO ---> use csv module for csv file saving
    @staticmethod
    def list_to_string(list_of_spectra, include_group_name=True, include_header=True, delimiter='\t',
                       decimal_sep='.', new_line='\n', save_to_file=False, dir_path=None, extension=None, **kwargs):
        """
        Return a hierarchic structure of spectra as text according to specifications
        or save it to file/s to specified directory.
        """
        if not isinstance(list_of_spectra, list):
            if isinstance(list_of_spectra, Spectrum):
                return list_of_spectra.__str__(**kwargs)
            else:
                raise ValueError("Argument \'list_of_spectra\' must be type of list or spectrum.")

        ret_buffer = ""

        for i, node in enumerate(list_of_spectra):

            if isinstance(node, list):
                # export as group

                if len(node) == 0:
                    continue

                if include_group_name:
                    buffer = node[0].group_name + new_line if node[0].group_name != '' else ""
                else:
                    buffer = ""

                # add row of wavelength, then, we will transpose the matrix, otherwise, we would have to reshape
                matrix = node[0].data[:, 0]

                # add absorbance data to matrix from all exported spectra
                for sp in node:
                    if sp.length() != node[0].length():
                        raise ValueError(
                            "Spectra \'{}\' and \'{}\' in group \'{}\' have not the same length (dimension). "
                            "Unable to export.".format(node[0].name, sp.name, node[0].group_name))
                    matrix = np.vstack((matrix, sp.data[:, 1]))

                # add header if it is user defined
                buffer += "Wavelength" + delimiter + delimiter.join(
                    sp.name for sp in node) + new_line if include_header else ""

                matrix = matrix.T  # transpose

                # add a new line and print the matrix to string buffer
                buffer += new_line.join(delimiter.join(Spectrum.float_to_string(num, decimal_sep)
                                                       for num in row) for row in matrix)

                if save_to_file:
                    filepath = os.path.join(dir_path,
                                            (node[0].group_name if node[0].group_name != '' else 'Untitled{}'.format(i))
                                            + extension)
                    try:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(buffer)
                    except Exception as ex:
                        raise Exception("Exception occurred while "
                                        "attempting to save a file \'{}\'.\n{}".format(filepath, ex.__str__()))
                else:
                    ret_buffer += buffer + 2 * new_line

            if isinstance(node, Spectrum):

                buffer = node.__str__(delimiter, decimal_sep, new_line, include_header)

                if save_to_file:
                    filepath = os.path.join(dir_path,
                                            (node.name if node.name != '' else 'Untitled{}'.format(i))
                                            + extension)
                    try:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(buffer)
                    except Exception as ex:
                        raise Exception("Exception occurred while "
                                        "attempting to save a file \'{}\'.\n{}".format(filepath, ex.__str__()))
                else:
                    ret_buffer += buffer + 2 * new_line

        if ret_buffer != "":
            # return ret_buffer and remove the 2 new line characters that are at the end
            return ret_buffer[:-2 * len(new_line)]

    @staticmethod
    def find_nearest_idx(array, value):
        # length = len(array)
        # min = array[0]
        # max = array[len(array) - 1]
        #
        # if value < min or value > max:
        #     return -1

        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
            return idx - 1
        else:
            return idx

    @staticmethod
    def find_nearest(array, value):
        idx = Spectrum.find_nearest_idx(array, value)
        return array[idx]
        # if idx > -1:
        #     return array[idx]
        # else:
        #     return None

    def __copy__(self):
        return deepcopy(self)


class SpectrumList(list):
    """
    Class that holds list of spectra and enables simple arithmetic calculations among lists of spectra, number and list and spectrum and list.
    Also, it enables other calculations on the group of spectra, like baseline correction, cutting, basically all the operation provided in
    Spectrum class plus additional operations that can be only used for groups, like transposition of list, ....
    """

    @staticmethod
    def _append(items):
        # this only does '[items]' but this cannot be done because we are using ItemList and not regular list...
        ret = SpectrumList()
        ret.append(items)
        return ret

    def add_to_list(self):
        """
        Imports spectra to the Treewidget
        """
        from user_namespace import add_to_list
        add_to_list(self)

    def fit_curve(self, x0, x1, model='A-B', print_data=False):

        if not isinstance(self[0], Spectrum):
            raise ValueError("Objects in list have to be type of Spectrum.")

        from user_namespace import add_to_list

        fit_sps = SpectrumList()
        res_sps = SpectrumList()
        fitted_params = []

        for sp in self:
            fit_vars, cov, sp_fit, sp_res = sp.fit_curve(x0, x1, model, print_data, False)

            sp_fit.group_name = 'Fits of {}'.format(self[0].group_name)
            sp_res.group_name = 'Residuals of {}'.format(self[0].group_name)

            fit_sps.append(sp_fit)
            res_sps.append(sp_res)
            fitted_params.append(fit_vars)

        sp_to_view = SpectrumList()
        sp_to_view.append(fit_sps)
        sp_to_view.append(res_sps)

        add_to_list(sp_to_view)

        return np.asarray(fitted_params, dtype=np.float64)

    def _iterate_items(self):
        for node in self:
            if isinstance(node, SpectrumList):
                for sp in node:
                    if not isinstance(sp, Spectrum):
                        raise ValueError("Objects in list have to be type of Spectrum.")
                    yield sp
                continue
            if isinstance(node, Spectrum):
                yield node
                continue
            raise ValueError("Objects in list have to be type of Spectrum.")

    def integrate(self, x0=None, x1=None):
        """Integrates this group of spectra and return the result as ndarray."""

        # .format(Spectrum.integrate.__doc__)

        if not isinstance(self[0], Spectrum):
            raise ValueError("Objects in list have to be type of Spectrum.")

        results = []

        for sp in self._iterate_items():
            results.append(sp.integrate(x0, x1))

        return results

    def get_y_values_at_x(self, x):
        """Returns y values at particular x value as an ndarray in this group."""

        if not isinstance(self[0], Spectrum):
            raise ValueError("Objects in list have to be type of Spectrum.")

        ret_list = []

        for sp in self:
            min = sp.data[0, 0]
            max = sp.data[-1, 0]

            if x < min or x > max:
                raise ValueError("Entered x value '{}' is out of range of spectrum '{}' x boundaries [{}, {}],"
                                 " cannot select y value.".format(x, sp.name, min, max))

            idx = Spectrum.find_nearest_idx(sp.data[:, 0], x)

            ret_list.append(sp.data[idx, 1])

        return np.asarray(ret_list, dtype=np.float64)

    def get_names(self):
        """Returns names of all Spectrum objects as list."""
        if not isinstance(self[0], Spectrum):
            raise ValueError("Objects in list have to be type of Spectrum.")

        return [sp.name for sp in self]

    def transpose(self, max_items=1000):
        """
        Transpose group, names of spectra in group will be taken as x values for transposed data.
        So these values must be convertible to int or float numbers. No text is allowed in these cells, only values.
        All x values of spectra in the group will become names in new group.
        """

        if len(self) < 2:
            raise ValueError("At least 2 items have to in the group in order to perform transposition.")

        if not isinstance(self[0], Spectrum):
            raise ValueError("Objects in list have to be type of Spectrum.")

        x_vals = []

        for sp in self:
            try:
                x_vals.append(float(sp.name.replace(',', '.').strip()))
            except ValueError:
                raise ValueError("Names of spectra cannot be parsed to float.")

        #
        matrix = np.asarray(x_vals, dtype=np.float64)

        # add absorbance data to matrix from all exported spectra
        for sp in self:
            if sp.length() != self[0].length():
                raise ValueError(
                    "Spectra \'{}\' and \'{}\' have not the same length (dimension). "
                    "Unable to transpose.".format(self[0].name, sp.name))
            matrix = np.vstack((matrix, sp.data[:, 1]))

    def savitzky_golay(self, window_length, poly_order):
        """Applies Savitysky-Golay filter to a list of spectra"""
        if not isinstance(self[0], Spectrum):
            raise ValueError("Objects in list have to be type of Spectrum.")

        for sp in self:
            sp.savitzky_golay(window_length, poly_order)

    def baseline_correct(self, x0, x1):
        for sp in self._iterate_items():
            sp.baseline_correct(x0, x1)

    def cut(self, x0, x1):
        for sp in self._iterate_items():
            sp.cut(x0, x1)

    def resample(self, spacing):
        for sp in self._iterate_items():
            sp.resample(spacing)

    def normalize(self, x0, x1):
        for sp in self._iterate_items():
            sp.normalize(x0, x1)

    def extend_by_zeros(self, x0, x1):
        for sp in self._iterate_items():
            sp.extend_by_zeros(x0, x1)

    def _perform_arithmetic_operation(self, other, func_operation):
        """
        Perform an defined operation on groups of spectra (type Spectrum). func_operation is a pointer to function.
        This function takes 2 arguments that must be type of Spectrum.

        :param other:
        :param func_operation:
        :return: ItemList of Spectra
        """
        # operation with another list, group + group
        if isinstance(other, SpectrumList):
            if len(self) != len(other):
                raise ValueError("Cannot perform an operation on groups which contains different number of items.")
            if len(self) == 0:
                return SpectrumList()
            if not isinstance(self[0], Spectrum) or not isinstance(other[0], Spectrum):
                raise ValueError("Objects in list have to be type of Spectrum.")
            ret_list = SpectrumList()
            for i in range(len(self)):
                ret_list.append(func_operation(self[i], other[i]))
            return ret_list, other[0].group_name

        # operation with single spectrum, group + spectrum or with number, eg. group - 1
        if isinstance(other, Spectrum) or isinstance(other, float) or isinstance(other, int):
            if len(self) == 0:
                return SpectrumList()

            if not isinstance(self[0], Spectrum):
                raise ValueError("Objects in list have to be type of Spectrum.")
            ret_list = SpectrumList()
            for sp in self:
                ret_list.append(func_operation(sp, other))
            return ret_list, str(other) if isinstance(other, float) or isinstance(other, int) else other.name

        raise ValueError("Cannot perform calculation of SpectrumList with {}".format(type(other)))

    def __add__(self, other):
        list, other_str = self._perform_arithmetic_operation(other, lambda s1, s2: s1 + s2)
        for sp in list:
            sp.group_name = "{} + {}".format(self[0].group_name, other_str)
        return SpectrumList._append(list)

    def __sub__(self, other):
        list, other_str = self._perform_arithmetic_operation(other, lambda s1, s2: s1 - s2)
        for sp in list:
            sp.group_name = "{} - {}".format(self[0].group_name, other_str)
        return SpectrumList._append(list)

    def __mul__(self, other):
        list, other_str = self._perform_arithmetic_operation(other, lambda s1, s2: s1 * s2)
        for sp in list:
            sp.group_name = "{} * {}".format(self[0].group_name, other_str)
        return SpectrumList._append(list)

    def __truediv__(self, other):
        list, other_str = self._perform_arithmetic_operation(other, lambda s1, s2: s1 / s2)
        for sp in list:
            sp.group_name = "{} / {}".format(self[0].group_name, other_str)
        return SpectrumList._append(list)

    def __radd__(self, other):
        list, other_str = self._perform_arithmetic_operation(other, lambda s1, s2: s2 + s1)
        for sp in list:
            sp.group_name = "{} + {}".format(other_str, self[0].group_name)
        return SpectrumList._append(list)

    def __rsub__(self, other):
        list, other_str = self._perform_arithmetic_operation(other, lambda s1, s2: s2 - s1)
        for sp in list:
            sp.group_name = "{} - {}".format(other_str, self[0].group_name)
        return SpectrumList._append(list)

    def __rmul__(self, other):
        list, other_str = self._perform_arithmetic_operation(other, lambda s1, s2: s2 * s1)
        for sp in list:
            sp.group_name = "{} * {}".format(other_str, self[0].group_name)
        return SpectrumList._append(list)

    def __rtruediv__(self, other):
        list, other_str = self._perform_arithmetic_operation(other, lambda s1, s2: s2 / s1)
        for sp in list:
            sp.group_name = "{} / {}".format(other_str, self[0].group_name)
        return SpectrumList._append(list)

    # wrap-around about slicing, prevention of returning a regular list, we want to return a ItemList object instead
    def __getitem__(self, item):
        result = super(SpectrumList, self).__getitem__(item)
        if isinstance(result, list):
            return SpectrumList(result)
        return result

    # def __str__(self):
    #     return "TODO--->"
