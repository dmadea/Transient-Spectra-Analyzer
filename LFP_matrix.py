from scipy.linalg import svd, svdvals, diagsvd, pinv
import numpy as np

from scipy.optimize import fmin, minimize
from scipy.linalg import lstsq

from numba import njit, vectorize, jit
import os

from copy import deepcopy

import matplotlib.pyplot as plt

from gui_console import Console

from fitmodels import _Model

from lmfit import Parameters, fit_report, fit_report, ci_report
import lmfit

from gui_console import Console

from pymcr.mcr import McrAR
from pymcr.regressors import OLS, NNLS
from pymcr.constraints import ConstraintNonneg, ConstraintNorm

from logger import Logger
from matplotlib.ticker import Locator

from spectrum import Spectrum

from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm


def register_div_cmap(zmin, zmax):
    diff = zmax - zmin
    w = np.abs(zmin / diff)  # white color point set to zero z value

    _cdict = {'red': ((0.0, 0.0, 0.0),
                      (w / 2, 0.0, 0.0),
                      (w, 1.0, 1.0),
                      (w + (1 - w) / 3, 1.0, 1.0),
                      (w + (1 - w) * 2 / 3, 1.0, 1.0),
                      (1.0, 0.3, 0.3)),

              'green': ((0.0, 0, 0),
                        (w / 2, 0.0, 0.0),
                        (w, 1.0, 1.0),
                        (w + (1 - w) / 3, 1.0, 1.0),
                        (w + (1 - w) * 2 / 3, 0.0, 0.0),
                        (1.0, 0.0, 0.0)),

              'blue': ((0.0, 0.3, 0.3),
                       (w / 2, 1.0, 1.0),
                       (w, 1.0, 1.0),
                       (w + (1 - w) / 3, 0.0, 0.0),
                       (w + (1 - w) * 2 / 3, 0.0, 0.0),
                       (1.0, 0.0, 0.0))
              }

    custom_cmap = LinearSegmentedColormap('diverging', _cdict)
    cm.register_cmap('diverging', custom_cmap)


# cdict = {'red': ((0.0, 0.0, 0.0),
#                  (2 / 5, 0.0, 0.0),
#                  (1 / 2, 1.0, 1.0),
#                  (3 / 5, 1.0, 1.0),
#                  (4 / 5, 1.0, 1.0),
#                  (1.0, 0.3, 0.3)),
#
#          'green': ((0.0, 0, 0),
#                    (2 / 5, 0.0, 0.0),
#                    (1 / 2, 1.0, 1.0),
#                    (3 / 5, 1.0, 1.0),
#                    (4 / 5, 0.0, 0.0),
#                    (1.0, 0.0, 0.0)),
#
#          'blue': ((0.0, 0.3, 0.3),
#                   (2 / 5, 1.0, 1.0),
#                   (1 / 2, 1.0, 1.0),
#                   (3 / 5, 0.0, 0.0),
#                   (4 / 5, 0.0, 0.0),
#                   (1.0, 0.0, 0.0))
#          }
#
# _custom_cmap = LinearSegmentedColormap('diverging', cdict)
# cm.register_cmap('diverging', _custom_cmap)


# from fitmodels import *


#
# _ = LFP_matrix.construct_test_matrix()
# load_LFP_matrix(_)
# main_widget.LFP_matrix = _


def gauss(x, mu, sigma):
    return np.exp(- (x - mu) * (x - mu) / (2 * sigma * sigma))


class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """

    def __init__(self, linthresh, nints=10):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically. nints gives the number of
        intervals that will be bounded by the minor ticks.
        """
        self.linthresh = linthresh
        self.nintervals = nints

    def __call__(self):
        # Return the locations of the ticks
        majorlocs = self.axis.get_majorticklocs()

        if len(majorlocs) == 1:
            return self.raise_if_exceeds(np.array([]))

        # add temporary major tick locs at either end of the current range
        # to fill in minor tick gaps
        dmlower = majorlocs[1] - majorlocs[0]  # major tick difference at lower end
        dmupper = majorlocs[-1] - majorlocs[-2]  # major tick difference at upper end

        # add temporary major tick location at the lower end
        if majorlocs[0] != 0. and ((majorlocs[0] != self.linthresh and dmlower > self.linthresh) or (
                dmlower == self.linthresh and majorlocs[0] < 0)):
            majorlocs = np.insert(majorlocs, 0, majorlocs[0] * 10.)
        else:
            majorlocs = np.insert(majorlocs, 0, majorlocs[0] - self.linthresh)

        # add temporary major tick location at the upper end
        if majorlocs[-1] != 0. and ((np.abs(majorlocs[-1]) != self.linthresh and dmupper > self.linthresh) or (
                dmupper == self.linthresh and majorlocs[-1] > 0)):
            majorlocs = np.append(majorlocs, majorlocs[-1] * 10.)
        else:
            majorlocs = np.append(majorlocs, majorlocs[-1] + self.linthresh)

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i - 1]
            if abs(majorlocs[i - 1] + majorstep / 2) < self.linthresh:
                ndivs = self.nintervals
            else:
                ndivs = self.nintervals - 1.

            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i - 1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                                  '%s type.' % type(self))


# class MinorSymLogLocator(Locator):
#     """
#     Dynamically find minor tick positions based on the positions of
#     major ticks for a symlog scaling.
#     """
#     def __init__(self, linthresh):
#         """
#         Ticks will be placed between the major ticks.
#         The placement is linear for x between -linthresh and linthresh,
#         otherwise its logarithmically
#         """
#         self.linthresh = linthresh
#
#     def __call__(self):
#         'Return the locations of the ticks'
#         majorlocs = self.axis.get_majorticklocs()
#
#         # iterate through minor locs
#         minorlocs = []
#
#         # handle the lowest part
#         for i in range(1, len(majorlocs)):
#             majorstep = majorlocs[i] - majorlocs[i-1]
#             if abs(majorlocs[i-1] + majorstep/2) < self.linthresh:
#                 ndivs = 10
#             else:
#                 ndivs = 9
#             minorstep = majorstep / ndivs
#             locs = np.arange(majorlocs[i-1], majorlocs[i], minorstep)[1:]
#             minorlocs.extend(locs)
#
#         return self.raise_if_exceeds(np.array(minorlocs))
#
#     def tick_values(self, vmin, vmax):
#         raise NotImplementedError('Cannot get tick locations for a '
#                                   '%s type.' % type(self))


class LFP_matrix(object):

    @classmethod
    def from_value_matrix(cls, value_matrix, times, wavelengths):
        m = cls()
        m.Y = value_matrix
        m.times = times
        m.wavelengths = wavelengths
        m.SVD()
        return m

    @property
    def SVD_filter(self):
        return self._SVD_filter

    @SVD_filter.setter
    def SVD_filter(self, value):
        self.D = self.Yr if value else self.Y
        self._SVD_filter = value

    def __init__(self, data=None, filename=None, name=None):

        self.wavelengths = None  # dim = w
        self.times = None  # dim = t
        self.Y = None  # dim = (t, w)
        self.original_data_matrix = None

        self._SVD_filter = False

        self.D = None  # matrix to be plotted, can be raw data or after any filtering (eg. SVD filter)

        if data is not None:
            self.wavelengths = data[0, 1:]  # first row without first column
            self.times = data[1:, 0]  # first column without the first row

            self.Y = data[1:, 1:]  # slice of the original matrix
            self.original_data_matrix = data

            self.D = self.Y

        self.filename = filename
        self.name = name

        # svd matrices k = min(t, w)
        self.U = None  # dim = (t x k)
        self.S = None  # !! this is only 1D array of singular values, not diagonal matrix
        self.V_T = None  # dim = (k x w)

        # reduced svd matrices, nr is number of taken singular values, cca 1 to 10 max
        self.Ur = None  # dim = (t x nr)
        self.Sr = None  # !! this is diagonal matrix - dim = (nr x nr)
        self.V_Tr = None  # dim = (nr x w)

        # n is number of actual absorbing species our model is based on
        # we are looking for C and A so it satisfy condition Y = C @ A

        # concentration matrix
        self.C = None  # dim = (t, n)

        # coefficients matrix dim = (n x nr)
        self.K = None

        # Spectra matrix is linear combination of V_Tr:  A = K @ V_Tr
        self.A = None  # dim = (n, w)

        self.UrSr = None  # Ur @ Sr

        # reconstructed data matrix after data reduction
        self.Yr = self.Y

        self.Y_fit = None

        self.model = None

        self.last_result = None
        self.minimizer = None

        # concetration and spectra matrices estimated by MCR-ALS method
        self.C_MCR = None
        self.A_MCR = None

        self.C_fit = None
        self.ST_fit = None
        self.E = None  # residuals

        self.times_fine = None
        self.C_fine = None

        self.SVD()

        # self.A_list = []
        # self.C_list = []

    def reduce_time_dim(self, factor=2):
        self.Y = self.Y[::int(factor), :]
        self.times = self.times[::int(factor)]

    def reduce_wavelength_dim(self, factor=2):
        self.Y = self.Y[:, ::int(factor)]
        self.times = self.wavelengths[::int(factor)]

    @classmethod
    def construct_test_matrix(cls, noise_intensity=0.1):

        n_times = 300
        num = 3  # number of species
        n_wls = 200

        times = np.linspace(0, 100, num=n_times)
        wls = np.linspace(1, 300, num=n_wls)

        # create and fill spectra matrix
        ST = np.zeros((num, n_wls))

        ST[0] = gauss(wls, 100, 50)
        ST[1] = 1 * gauss(wls, 200, 50)
        ST[2] = 0.5 * gauss(wls, 50, 50)
        # A[3] = 0.6 * gauss(wls, 150, 30)

        import fitmodels

        # m = ABCDE_Model(times, visible=[True, True, False, True, True])
        m = fitmodels.ABC_Model(times)

        params = m.params

        params['c0'].value = 1
        params['k1'].value = 0.5
        params['k2'].value = 0.2
        # params['k3'].value = 0.05

        C = m.calc_C(params)

        # construct data
        D = C @ ST

        # add plane
        for i in range(n_times):
            D[i] += np.random.normal(scale=0.01)

        # Y += noise_intensity * np.random.rand(n_times, n_wls)

        matrix = cls.from_value_matrix(D, times, wls)

        from user_namespace import load_LFP_matrix, UserNamespace
        from Widgets.fit_widget import FitWidget

        load_LFP_matrix(matrix)
        UserNamespace.instance.main_widget.matrix = matrix
        Console.push_variables({'matrix': matrix, 'model': m, 'iA': ST, 'iC': C})
        FitWidget.instance.matrix = matrix

        from fitmodels import plot_figures

        # plot_figures(m)

    def SVD(self):

        if self.Y is None:
            return

        self.U, self.S, self.V_T = svd(self.Y, full_matrices=False, lapack_driver='gesdd')

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def calc_chi2(Y, C, A):
        # R = Y - np.dot(C, A)  # R = Y - C @ A = Y - C @ K @ V_Tr
        # return np.sum(R * R)
        return Y - np.dot(C, A)

    def residuals(self, params):

        C = self.model.calc_C(params)

        self.K = lstsq(C, self.UrSr)[0]

        self.A = np.dot(self.K, self.V_Tr)

        # #spectra cannot be negative
        # for i in range(self.A.shape[0]):
        #     for j in range(self.A.shape[1]):
        #         self.A[i, j] = self.A[i, j] if self.A[i, j] >= 0 else 0

        # self.A[0] = self.Y[0]

        # return self.Y - np.dot(C, self.A)

        # R = self.Y - self.C @ self.K @ self.V_Tr
        # chi2 = np.sum(R * R)
        # print('sum of residuals = {}'.format(chi2))
        return self.calc_chi2(self.Y, C, self.A)
        # return chi2

    def global_fit(self, model_cls, verbose=True, method='leastsq'):

        # reduction must have been perfomed before calling global fit
        if self.Ur is None or self.Sr is None or self.V_Tr is None:
            return

        # t = self.Y.shape[0]
        w = self.Y.shape[1]
        nr = self.Sr.shape[0]

        # Y = self.Y
        # times = self.times

        if isinstance(model_cls, _Model):
            self.model = model_cls
            self.model.init_times(self.times)
        else:
            self.model = model_cls(self.times)

        n = self.model.n

        # initialize matrices
        # self.C = np.zeros((t, n), dtype=np.float64)
        self.A = np.zeros((n, w), dtype=np.float64)
        self.K = np.zeros((n, nr), dtype=np.float64)

        self.UrSr = self.Ur @ self.Sr

        self.minimizer = lmfit.Minimizer(self.residuals, self.model.params)

        self.last_result = lmfit.minimize(self.residuals, self.model.params, method=method)

        # calculate final A matrix and C matrix
        # self.A = self.K @ self.V_Tr
        self.C = self.model.calc_C()

        self.Y_fit = self.C @ self.A

        if verbose:
            report = fit_report(self.last_result)
            Logger.console_message(report if report is not None else '')
            self.plot_figures_one()

        Console.push_variables({'K': self.K})

        return LFP_matrix.from_value_matrix(self.Y_fit, self.times, self.wavelengths)

    #
    # def MCR_ALS(self, n_components, from_original=False, max_iter=100, verbose=False):
    #
    #     # if not from_original and n_components > self.A.shape[0]:
    #     #     raise ValueError(f"Parameter n_components cannot be larger than number of taken vector for reconstruction.")
    #
    #     mcrals = McrAR(max_iter=max_iter, st_regr='NNLS', c_regr=OLS(),
    #                    c_constraints=[ConstraintNonneg()])
    #
    #     # if there was global fit performed, take result form it as an initial solution, the rest just noise
    #     iST = np.random.rand(n_components, self.wavelengths.shape[0])  # if self.A is None else self.A[:n_components]
    #
    #     if self.A is not None:
    #         for i in range(self.A.shape[0]):
    #             iST[i] = self.A[i]
    #
    #     mcrals.fit(self.Yr if not from_original else self.Y, ST=iST, verbose=verbose)
    #
    #     self.C_MCR = mcrals.C_opt_
    #     self.A_MCR = mcrals.ST_opt_
    #
    #     self.plot_figures_MCR()
    #
    #     self.Y_fit = self.C_MCR @ self.A_MCR
    #
    #     return LFP_matrix.from_value_matrix(self.Y_fit, self.times, self.wavelengths)

    def print_confidence_intervals(self, sigmas=(1, 2, 3)):

        if self.last_result is None:
            return

        ci = lmfit.conf_interval(self.minimizer, self.last_result, sigmas=sigmas,
                                 trace=False, verbose=False)

        Logger.console_message(ci_report(ci))

    def save_fit(self, filepath, ST=None, C=None):

        if self.C_fit is None:
            return

        D_fit = self.C_fit @ self.ST_fit

        # wavelengths = np.concatenate([[0], mw.matrix.wavelengths])

        mat = np.vstack((self.wavelengths, D_fit))

        t = np.concatenate([[0], self.times])
        mat = np.hstack((t.reshape(-1, 1), mat))

        buff_mat = LFP_matrix.to_string(mat, separator=',')

        with open(filepath + '.csv', 'w') as f:
            f.write(buff_mat)

        if ST is None and C is None:
            ST = self.ST_fit
            C = self.C_fit

        buff_A = 'Wavelength,' + ','.join([str(i + 1) for i in range(ST.shape[0])]) + '\n'

        ST = np.vstack((self.wavelengths, ST))
        buff_A += '\n'.join(','.join(str(num) for num in row) for row in ST.T)

        with open(filepath + '-A.csv', 'w') as f:
            f.write(buff_A)

        buff_C = 'Conc,' + ','.join([str(i + 1) for i in range(C.shape[1])]) + '\n'

        C = np.hstack((self.times.reshape(-1, 1), C))
        buff_C += '\n'.join(','.join(str(num) for num in row) for row in C)

        with open(filepath + '-C.csv', 'w') as f:
            f.write(buff_C)

    def plot_data(self, symlog=False, t_unit='s', z_unit='$\Delta A$', c_map='inferno_r', zmin=None, zmax=None,
                  w0=None, w1=None, t0=None, t1=None, fig_size=(6, 4), dpi=500, filepath=None, transparent=True,
                  linthreshy=10, linscaley=0.5):

        plt.rcParams['figure.figsize'] = fig_size
        # plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.23, hspace=0.26)

        plt.tight_layout()

        # cut data if necessary

        t_idx_start = Spectrum.find_nearest_idx(self.times, t0) if t0 is not None else 0
        t_idx_end = Spectrum.find_nearest_idx(self.times, t1) + 1 if t1 is not None else self.Y.shape[0]

        wl_idx_start = Spectrum.find_nearest_idx(self.wavelengths, w0) if w0 is not None else 0
        wl_idx_end = Spectrum.find_nearest_idx(self.wavelengths, w1) + 1 if w1 is not None else self.Y.shape[1]

        D = self.Y[t_idx_start:t_idx_end, wl_idx_start:wl_idx_end]
        times = self.times[t_idx_start:t_idx_end]
        wavelengths = self.wavelengths[wl_idx_start:wl_idx_end]

        zmin = np.min(D) if zmin is None else zmin
        zmax = np.max(D) if zmax is None else zmax

        register_div_cmap(zmin, zmax)

        x, y = np.meshgrid(wavelengths, times)  # needed for pcolormesh to correctly scale the image

        # plot data matrix D

        plt.pcolormesh(x, y, D, cmap=c_map, vmin=zmin, vmax=zmax)

        plt.colorbar(label=z_unit)
        plt.title("Data matrix $D$")
        plt.ylabel(f'Time ({t_unit})')
        plt.xlabel('Wavelength (nm)')

        plt.gca().invert_yaxis()

        if symlog:
            plt.yscale('symlog', subsy=[2, 3, 4, 5, 6, 7, 8, 9], linscaley=linscaley, linthreshy=linthreshy)
            yaxis = plt.gca().yaxis
            yaxis.set_minor_locator(MinorSymLogLocator(linthreshy))

        # save to file
        if filepath:
            ext = os.path.splitext(filepath)[1].lower()[1:]
            plt.savefig(fname=filepath, format=ext, transparent=transparent, dpi=dpi)
        else:
            plt.show()

    def plot_fit(self, symlog=True, wls=(520, 560, 600), times=(0, 20, 200, 2000), t_unit='s', z_unit='Absorbance $A$',
                 c_map='inferno_r', fpath=None, format='png', dpi=500, figsize=(18, 10), time_treshold=100,
                 time_linscale=0.5):

        # if self.C_fit is None:
        #     return

        # comp = list('ABCD')
        # comp = ['A', 'B+D', 'C']
        comp = ['D', 'M', 'C']

        # time_treshold = 100
        # time_linscale = 0.5

        D_fit = self.C_fit @ self.ST_fit

        self.E = self.Y - D_fit

        plt.rcParams['figure.figsize'] = figsize
        # plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.23, hspace=0.26)
        # plt.tight_layout()

        x, y = np.meshgrid(self.wavelengths, self.times)  # needed for pcolormesh to correctly scale the image

        # plot data matrix D

        plt.subplot(231)  # # of rows, # of columns, index counting form 1

        plt.pcolormesh(x, y, self.Y, cmap=c_map, vmin=np.min(self.Y), vmax=np.max(self.Y))
        plt.colorbar(label=z_unit)
        # plt.colorbar()
        plt.title("Data matrix $D$")
        plt.ylabel(f'Time ({t_unit})')
        plt.xlabel('Wavelength (nm)')

        plt.gca().invert_yaxis()

        if symlog:
            plt.yscale('symlog', subsy=[1, 2, 3, 4, 5, 6, 7, 8, 9], linscaley=time_linscale, linthreshy=time_treshold)
            yaxis = plt.gca().yaxis
            yaxis.set_minor_locator(MinorSymLogLocator(time_treshold))

        # concentration profile

        plt.subplot(232)  # # of rows, # of columns, index counting form 1

        for i in range(self.C_fit.shape[1]):
            plt.plot(self.times, self.C_fit[:, i], label=f"Species {comp[i]}", lw=1.5)

        plt.title("Concentration matrix $C$")
        plt.xlabel(f'Time ({t_unit})')
        plt.ylabel('Relative population')
        plt.legend()

        if symlog:
            plt.xscale('symlog', subsx=[1, 2, 3, 4, 5, 6, 7, 8, 9], linscalex=time_linscale, linthreshx=time_treshold)
            xaxis = plt.gca().xaxis
            xaxis.set_minor_locator(MinorSymLogLocator(time_treshold))

        # spectras

        plt.subplot(233)  # # of rows, # of columns, index counting form 1

        for i in range(self.ST_fit.shape[0]):
            plt.plot(self.wavelengths, self.ST_fit[i], label=f"Species {comp[i]}", lw=1.5)

        plt.title("Spectra matrix $S^T$")
        plt.xlabel('Wavelength (nm)')
        plt.ylabel(z_unit)
        plt.legend()

        # plot residuals

        plt.subplot(234)  # # of rows, # of columns, index counting form 1

        plt.pcolormesh(x, y, self.E, cmap='seismic', vmin=-np.abs(np.max(self.E)), vmax=np.abs(np.max(self.E)))

        R2 = 1 - (self.E * self.E).sum() / (self.Y * self.Y).sum()
        # E2 = (self.E * self.E).sum()
        title = "Residuals $E=CS^T - D$"
        title += f", $R^2$={R2:.5g}"

        # title += f", |E|$^2$={E2:.1E}"

        plt.colorbar(label=z_unit)
        plt.title(title)
        plt.ylabel(f'Time ({t_unit})')
        plt.xlabel('Wavelength (nm)')

        plt.gca().invert_yaxis()

        if symlog:
            plt.yscale('symlog', subsy=[1, 2, 3, 4, 5, 6, 7, 8, 9], linscaley=time_linscale, linthreshy=time_treshold)
            yaxis = plt.gca().yaxis
            yaxis.set_minor_locator(MinorSymLogLocator(time_treshold))

        # wavelength fits time trace

        wl_idxs = []
        for wl in wls:
            wl_idxs.append(Spectrum.find_nearest_idx(self.wavelengths, wl))

        plt.subplot(235)  # # of rows, # of columns, index counting form 1

        for i, idx in enumerate(wl_idxs):
            plt.plot(self.times, self.Y[:, idx], label=f"Data at {wls[i]} nm", lw=1)
        for i, idx in enumerate(wl_idxs):
            plt.plot(self.times, D_fit[:, idx], label=f"Fit at {wls[i]} nm", lw=1, color='black', ls='--')

        plt.title("Traces at various wavelengths")
        plt.xlabel(f'Time ({t_unit})')
        plt.ylabel(z_unit)
        plt.legend()

        if symlog:
            plt.xscale('symlog', subsx=[1, 2, 3, 4, 5, 6, 7, 8, 9], linscalex=time_linscale, linthreshx=time_treshold)
            xaxis = plt.gca().xaxis
            xaxis.set_minor_locator(MinorSymLogLocator(time_treshold))

        # time fits spectras

        t_idxs = []
        for t in times:
            t_idxs.append(Spectrum.find_nearest_idx(self.times, t))

        plt.subplot(236)  # # of rows, # of columns, index counting form 1

        for i, idx in enumerate(t_idxs):
            plt.plot(self.wavelengths, self.Y[idx], label=f"Data at {times[i]} {t_unit}", lw=1)
        for i, idx in enumerate(t_idxs):
            plt.plot(self.wavelengths, D_fit[idx], label=f"Fit at {times[i]} {t_unit}", lw=1, color='black', ls='--')

        plt.title("Spectra at various times")
        plt.xlabel('Wavelength (nm)')
        plt.ylabel(z_unit)
        plt.legend()

        plt.tight_layout()

        if fpath is None:
            plt.show()
        else:
            plt.savefig(fname=fpath, format=format, transparent=True, dpi=dpi)

        plt.cla()
        plt.clf()
        plt.close()

    def plot_fit_eq(self, symlog=True, wls=(520, 560, 600), times=(0, 20, 200, 2000), t_unit='mol dm$^{{-3}}$',
                    z_unit='Absorbance $A$',
                    c_map='inferno_r', fpath=None, format='png', dpi=500, figsize=(18, 10), time_treshold=100,
                    time_linscale=0.5,
                    font_size=8, loc='best'):

        comp = ['D', 'M', 'C']

        # time_treshold = 100
        # time_linscale = 0.5

        # D_fit = self.C_fit @ self.ST_fit

        D_fit = self.C_fine @ self.ST_fit  # fine

        self.E = self.Y - self.C_fit @ self.ST_fit

        plt.rcParams['figure.figsize'] = figsize
        # plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.23, hspace=0.26)
        # plt.tight_layout()

        x, y = np.meshgrid(self.wavelengths, self.times)  # needed for pcolormesh to correctly scale the image

        # plot data matrix D

        plt.subplot(231)  # # of rows, # of columns, index counting form 1
        c1 = np.asarray([0.12156863, 0.46666667, 0.70588235, 1])
        c2 = np.asarray([1, 0.49803922, 0.05490196, 1])

        n = self.Y.shape[0]

        for i in range(n):
            ind = i % n
            pos = ind / n
            ci = c1 * (1 - pos) + c2 * pos
            plt.plot(self.wavelengths, self.Y[i], label=f"$c_{{L,0}}$ = {self.times[i]:.2E} mol dm$^{{-3}}$", lw=1,
                     color=ci)

        # plt.title("Traces at various wavelengths")
        plt.xlabel('Wavelength (nm)')
        plt.ylabel(z_unit)
        plt.legend(loc=loc, prop={'size': font_size})

        #
        #
        # plt.pcolormesh(x, y, self.Y, cmap=c_map, vmin=np.min(self.Y), vmax=np.max(self.Y))
        # plt.colorbar(label=z_unit)
        # # plt.colorbar()
        # plt.title("Data matrix $D$")
        # plt.ylabel('Added pyridine (mol dm$^{-3}$)')
        # plt.xlabel('Wavelength (nm)')
        #
        # plt.gca().invert_yaxis()
        #
        # if symlog:
        #     plt.yscale('symlog', subsy=[1, 2, 3, 4, 5, 6, 7, 8, 9], linscaley=time_linscale, linthreshy=time_treshold)
        #     yaxis = plt.gca().yaxis
        #     yaxis.set_minor_locator(MinorSymLogLocator(time_treshold))

        # concentration profile

        plt.subplot(232)  # # of rows, # of columns, index counting form 1

        for i in range(self.C_fine.shape[1]):
            plt.plot(self.times_fine, self.C_fine[:, i], label=f"Species {comp[i]}", lw=1.5)

        plt.title("Concentration matrix $C$")
        plt.xlabel('Added pyridine (mol dm$^{-3}$)')
        plt.ylabel('Concentration (mol dm$^{-3}$)')
        plt.legend()

        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

        if symlog:
            plt.xscale('symlog', subsx=[1, 2, 3, 4, 5, 6, 7, 8, 9], linscalex=time_linscale, linthreshx=time_treshold)
            xaxis = plt.gca().xaxis
            xaxis.set_minor_locator(MinorSymLogLocator(time_treshold))

        # spectras

        plt.subplot(233)  # # of rows, # of columns, index counting form 1

        for i in range(self.ST_fit.shape[0]):
            plt.plot(self.wavelengths, self.ST_fit[i], label=f"Species {comp[i]}", lw=1.5)

        plt.title("Spectra matrix $S^T$")
        plt.xlabel('Wavelength (nm)')
        plt.ylabel(r'$\varepsilon$ (mol$^{-1}$ dm$^3$ cm$^{-1}$)')
        plt.legend()

        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        # plot residuals

        plt.subplot(234)  # # of rows, # of columns, index counting form 1

        plt.pcolormesh(x, y, self.E, cmap='seismic', vmin=-np.abs(np.max(self.E)), vmax=np.abs(np.max(self.E)))

        R2 = 1 - (self.E * self.E).sum() / (self.Y * self.Y).sum()
        # E2 = (self.E * self.E).sum()
        title = "Residuals $E=CS^T - D$"
        title += f", $R^2$={R2:.5g}"

        # title += f", |E|$^2$={E2:.1E}"

        plt.colorbar(label=z_unit)
        plt.title(title)
        plt.ylabel('Added pyridine (mol dm$^{-3}$)')
        plt.xlabel('Wavelength (nm)')

        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        plt.gca().invert_yaxis()

        if symlog:
            plt.yscale('symlog', subsy=[1, 2, 3, 4, 5, 6, 7, 8, 9], linscaley=time_linscale, linthreshy=time_treshold)
            yaxis = plt.gca().yaxis
            yaxis.set_minor_locator(MinorSymLogLocator(time_treshold))

        # wavelength fits time trace

        wl_idxs = []
        for wl in wls:
            wl_idxs.append(Spectrum.find_nearest_idx(self.wavelengths, wl))

        plt.subplot(235)  # # of rows, # of columns, index counting form 1

        for i, idx in enumerate(wl_idxs):
            plt.scatter(self.times, self.Y[:, idx], label=f"Data at {wls[i]} nm", lw=1)
        for i, idx in enumerate(wl_idxs):
            plt.plot(self.times_fine, D_fit[:, idx], label=f"Fit at {wls[i]} nm", lw=1, color='black', ls='--')

        plt.title("Traces at various wavelengths")
        plt.xlabel('Added pyridine (mol dm$^{-3}$)')
        plt.ylabel(z_unit)
        plt.xlim(-0.6e-5, 2.2e-4)
        plt.legend()

        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

        if symlog:
            plt.xscale('symlog', subsx=[1, 2, 3, 4, 5, 6, 7, 8, 9], linscalex=time_linscale, linthreshx=time_treshold)
            xaxis = plt.gca().xaxis
            xaxis.set_minor_locator(MinorSymLogLocator(time_treshold))

        # time fits spectras

        # t_idxs = []
        # for t in times:
        #     t_idxs.append(Spectrum.find_nearest_idx(self.times, t))

        # plt.subplot(236)  # # of rows, # of columns, index counting form 1
        #
        # for i, idx in enumerate(t_idxs):
        #     plt.plot(self.wavelengths, self.Y[idx], label=f"Data at {times[i]} {t_unit}", lw=1)
        # for i, idx in enumerate(t_idxs):
        #     plt.plot(self.wavelengths, D_fit[idx], label=f"Fit at {times[i]} {t_unit}", lw=1, color='black', ls='--')
        #
        # plt.title("Spectra at various py concentrations")
        # plt.xlabel('Wavelength (nm)')
        # plt.ylabel(z_unit)
        # plt.legend()

        plt.tight_layout()

        if fpath is None:
            plt.show()
        else:
            plt.savefig(fname=fpath, format=format, transparent=True, dpi=dpi)

        plt.cla()
        plt.clf()
        plt.close()

    def plot_first_n_vectors(self, n=4, symlog=False, t_unit='ms'):
        # import matplotlib.ticker as ticker
        S_single = []  # define a list of diagonal matrices with only one singular value
        for i in range(n):
            S_i = np.zeros((self.S.shape[0], self.S.shape[0]))  # recreate diagonal matrices
            S_i[i, i] = self.S[i]
            S_single.append(S_i)

        plt.rcParams['figure.figsize'] = [15, 8]
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.26, hspace=0.41)

        x, y = np.meshgrid(self.wavelengths, self.times)  # needed for pcolormesh to correctly scale the image
        for i in range(n):
            plt.subplot(4, n, i + 1)
            plt.plot(self.wavelengths, self.V_T[i], color='black', lw=1.5)
            plt.title("{}. $V^T$ vector, $\Sigma_{{{}}}$ = {:.3g}".format(i + 1, str(i + 1) * 2, self.S[i]))
            plt.xlabel('Wavelength (nm)')
            # plt.xticks(np.arange(self.wavelengths[0], self.wavelengths[-1], step=50))
            # plt.xlim(self.wavelengths[0], self.wavelengths[-1])
            # plt.set_major_locator(ticker.MultipleLocator(100))
        for i in range(n):
            plt.subplot(4, n, i + n + 1)

            plt.plot(self.times, self.U[:, i], color='black', lw=1.5)
            plt.title("{}. $U$ vector".format(i + 1))
            plt.xlabel(f'Time ({t_unit})')

            if symlog:
                plt.xscale('symlog', subsx=[1, 2, 3, 4, 5, 6, 7, 8, 9], linscalex=1, linthreshx=100)
                xaxis = plt.gca().xaxis
                xaxis.set_minor_locator(MinorSymLogLocator(100))

        for i in range(n):
            plt.subplot(4, n, i + 2 * n + 1)

            A_rec = self.U @ S_single[i] @ self.V_T  # reconstruct the A matrix
            # setup z range so that white color corresponds to 0
            plt.pcolormesh(x, y, A_rec, cmap='seismic', vmin=-np.abs(np.max(A_rec)), vmax=np.abs(np.max(A_rec)))
            # plt.colorbar(label='Absorbance')
            plt.colorbar()
            plt.title("Component matrix {}".format(i + 1))
            plt.xlabel('Wavelength (nm)')
            plt.ylabel(f'Time ({t_unit})')

            if symlog:
                plt.xscale('symlog', subsx=[1, 2, 3, 4, 5, 6, 7, 8, 9], linscalex=1, linthreshx=100)
                xaxis = plt.gca().xaxis
                xaxis.set_minor_locator(MinorSymLogLocator(100))

            plt.gca().invert_yaxis()

            plt.subplot(4, n, i + 3 * n + 1)

            S = np.diag(self.S)

            Ur = self.U[:, :i + 1]
            Sr = S[:i + 1, :i + 1]
            V_Tr = self.V_T[:i + 1, :]
            Yr = Ur @ Sr @ V_Tr

            A_diff = Yr - self.Y

            # E2 = (A_diff * A_diff).sum()
            R2 = (1 - (A_diff * A_diff).sum() / (self.Y * self.Y).sum())

            title = "Residuals (E=D$_{{rec}}$({}) - D)".format(i + 1)
            title += f", $R^2$={R2:.4g}"

            plt.pcolormesh(x, y, A_diff, cmap='seismic', vmin=-np.abs(np.max(A_diff)), vmax=np.abs(np.max(A_diff)))
            plt.colorbar()
            plt.title(title)
            plt.xlabel('Wavelength (nm)')
            plt.ylabel(f'Time ({t_unit})')

            if symlog:
                plt.xscale('symlog', subsx=[1, 2, 3, 4, 5, 6, 7, 8, 9], linscalex=1, linthreshx=100)
                xaxis = plt.gca().xaxis
                xaxis.set_minor_locator(MinorSymLogLocator(100))

            plt.gca().invert_yaxis()

        # plt.tight_layout()

        plt.show()

    def EFA(self, forward=True, backward=False, sing_values_num=10, points=100):
        """Performs forward and/or backward Evolving factor analysis over time domain on the current matrix."""

        t = np.linspace(int(self.times.shape[0] / points), self.times.shape[0] - 1, num=points).astype(int)
        sing_values = np.zeros((points, sing_values_num), dtype=np.float64)

        for i in range(points):
            U, S, V_T = svd(self.Y[:t[i], :], full_matrices=False, lapack_driver='gesdd')
            sing_values[i] = S[:sing_values_num]

        times = self.times[t]

        for i in range(sing_values_num):
            plt.plot(times, sing_values[:, i], label='{}'.format(i + 1))
        plt.xlabel('Time / s')
        plt.ylabel('Singular value')
        plt.title('Evolving factor analysis')
        plt.yscale('log')
        plt.legend()

        plt.show()

    def plot_log_of_S(self, n=10):

        # log_S = np.log(self.S[:n])
        x_data = range(1, n + 1)

        plt.rcParams['figure.figsize'] = [10, 6]

        plt.scatter(x_data, self.S[:n])
        plt.yscale('log')
        # plt.xlabel('Significant value number')
        plt.ylabel('Magnitude')
        plt.xlabel('Singular value index')
        min, max = np.min(self.S[:n]), np.max(self.S[:n])
        # dif = max - min

        plt.ylim(min / 2, 2 * max)
        plt.title('First {} sing. values'.format(n))

        fig = plt.gcf()
        fig.canvas.set_window_title('SVD Analysis')

        plt.show()

    def plot_figures_MCR(self):

        if self.C_MCR is None:
            return

        n = self.C_MCR.shape[1]
        plt.rcParams['figure.figsize'] = [12, 6]
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, wspace=0.33, hspace=0.33)

        plt.subplot(2, 2, 1)
        for i in range(n):
            plt.plot(self.wavelengths, self.A_MCR[i], label='Species {}'.format(i + 1))
        plt.xlabel('Wavelength / nm')
        plt.title('Spectra')
        plt.legend()

        plt.subplot(2, 2, 2)
        for i in range(n):
            plt.plot(self.times, self.C_MCR[:, i], label='Species {}'.format(i + 1))
        plt.xlabel('Time')
        plt.title('Concentrations')
        plt.legend()

        # plot residual matrix
        plt.subplot(2, 1, 2)
        A_dif = self.C_MCR @ self.A_MCR - self.Y  # the difference matrix between MCR fit and original
        x, y = np.meshgrid(self.times, self.wavelengths)  # needed for pcolormesh to correctly scale the image
        plt.pcolormesh(x, y, A_dif.T, cmap='seismic', vmin=-np.abs(np.max(A_dif)), vmax=np.abs(np.max(A_dif)))
        plt.colorbar().set_label("$\Delta$A")
        plt.title("Residual matrix A_fit - A")
        plt.ylabel('Wavelength / nm')
        plt.xlabel('Time')

        plt.tight_layout()

        fig = plt.gcf()
        # fig.canvas.set_window_title('Global fit - {}'.format(self.model.__class__.__name__))

        plt.show()

    def plot_figures_one(self):

        n = self.model.n
        plt.rcParams['figure.figsize'] = [12, 6]
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, wspace=0.33, hspace=0.33)

        plt.subplot(2, 2, 1)
        for i in range(n):
            plt.plot(self.wavelengths, self.A[i], label='Species {}'.format(self.model.get_species_name(i)))
        plt.xlabel('Wavelength / nm')
        plt.title('Spectra')
        plt.legend()

        plt.subplot(2, 2, 2)
        for i in range(n):
            plt.plot(self.times, self.C[:, i], label='Species {}'.format(self.model.get_species_name(i)))
        plt.xlabel('Time')
        plt.title('Concentrations')
        plt.legend()

        # plot residual matrix
        plt.subplot(2, 1, 2)
        A_dif = self.Y_fit - self.Y  # the difference matrix between fit and original
        x, y = np.meshgrid(self.times, self.wavelengths)  # needed for pcolormesh to correctly scale the image
        plt.pcolormesh(x, y, A_dif.T, cmap='seismic', vmin=-np.abs(np.max(A_dif)), vmax=np.abs(np.max(A_dif)))
        plt.colorbar().set_label("$\Delta$A")
        plt.title("Residual matrix A_fit - A")
        plt.ylabel('Wavelength / nm')
        plt.xlabel('Time')

        fig = plt.gcf()
        fig.canvas.set_window_title('Global fit - {}'.format(self.model.__class__.__name__))

        plt.show()

    def plot_figures_multiple(self):

        # plt.figure(1)

        n = self.model.n
        plt.rcParams['figure.figsize'] = [12, 6]
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.33, hspace=0.33)

        for i in range(n):
            plt.subplot(2, n, i + 1)
            plt.plot(self.wavelengths, self.A[i])
            plt.title('Spectrum {}'.format(self.model.get_species_name(i)))

        for i in range(n):
            plt.subplot(2, n, i + n + 1)
            plt.plot(self.times, self.C[:, i])
            plt.title('Conc. {}'.format(self.model.get_species_name(i)))

        plt.show()

    # def reconstruct_matrix_from_sing_value(self, sing_val_index):
    #     """Reconstruct the matrix only for specified singular value. 0, for
    #     1. sinular value, 1 for 2. singular value and so on..."""
    #
    #     if self.U is None or self.S is None or self.V_T is None:
    #         return
    #
    #     S = np.diag(self.S)
    #
    #     # fill zeros after our singular value to singular value matrix
    #     S[sing_val_index + 1:, sing_val_index + 1:] = 0
    #
    #     # fill zeros before our singular value to singular value matrix
    #     if sing_val_index > 0:
    #         S[:sing_val_index, :sing_val_index] = 0
    #
    #     Yr = self.U @ S @ self.V_T
    #
    #     return LFP_matrix.from_value_matrix(Yr, self.times, self.wavelengths)

    def set_SVD_filter(self, l_vectors=(0,)):
        """l_vector - list of singular vector to include into the filter, numbering from 0,
        eg. [0, 1, 2, 3, 5, 6], [0], [1], etc.
        """

        Sr_plain = self.S.copy()

        # calculate the difference of sets, from https://stackoverflow.com/questions/3462143/get-difference-between-two-lists
        l_diff = list(set([i for i in range(self.S.shape[0])]) - set(l_vectors))

        # put all other singular vectors different from chosen vectors to zero
        Sr_plain[l_diff] = 0

        Sr = np.diag(Sr_plain)

        # reconstruct the data matrix, @ is a dot product
        self.Yr = self.U @ Sr @ self.V_T

        # update D
        self.SVD_filter = self.SVD_filter

    # def reconstruct_matrix(self, nr):
    #     """Reduces the U, S and V_T matrices and constructs and loads the new matrix. nr is number of
    #     taken singular values. """
    #
    #     if self.U is None or self.S is None or self.V_T is None:
    #         return
    #
    #     # # create m x n Sigma matrix
    #     # Sigma = np.zeros((A.shape[0], A.shape[1]))
    #     # # populate Sigma with n x n diagonal matrix
    #     # Sigma[:A.shape[1], :A.shape[1]] = np.diag(self.S)
    #
    #     # construct
    #     S = np.diag(self.S)
    #
    #     # reduce the U matrix
    #     self.Ur = self.U[:, :nr]
    #
    #     # reduce the Sigma matrix
    #     self.Sr = S[:nr, :nr]
    #     # reduce the V_T matrix
    #     self.V_Tr = self.V_T[:nr, :]
    #
    #     # reconstruct the data matrix, @ is a dot product
    #     self.Yr = self.Ur @ self.Sr @ self.V_Tr
    #
    #     return LFP_matrix.from_value_matrix(self.Yr, self.times, self.wavelengths)

    def crop_data(self, t0=None, t1=None, w0=None, w1=None):
        t_idx_start = Spectrum.find_nearest_idx(self.times, t0) if t0 is not None else 0
        t_idx_end = Spectrum.find_nearest_idx(self.times, t1) + 1 if t1 is not None else self.Y.shape[0]

        wl_idx_start = Spectrum.find_nearest_idx(self.wavelengths, w0) if w0 is not None else 0
        wl_idx_end = Spectrum.find_nearest_idx(self.wavelengths, w1) + 1 if w1 is not None else self.Y.shape[1]

        self.Y = self.Y[t_idx_start:t_idx_end, wl_idx_start:wl_idx_end]
        self.times = self.times[t_idx_start:t_idx_end]
        self.wavelengths = self.wavelengths[wl_idx_start:wl_idx_end]

        self.SVD()

        # update matrix D
        self.SVD_filter = self.SVD_filter

        return self

    def reduce(self, t_dim=None, w_dim=None):
        """Reduces the time and wavelength dimension by t_dim and w_dim, respectively.
        eg. for t_dim=10, every 10-th row of original matrix will contain reduced matrix."""

        t_factor = int(t_dim) if t_dim is not None else 1
        w_factor = int(w_dim) if w_dim is not None else 1

        self.Y = self.Y[::t_factor, :]
        self.times = self.times[::t_factor]

        self.Y = self.Y[:, ::w_factor]
        self.wavelengths = self.wavelengths[::w_factor]

        self.SVD()

        # update matrix D
        self.SVD_filter = self.SVD_filter

        return self

    def restore_original_data(self):
        if self.original_data_matrix is not None:
            self.wavelengths = self.original_data_matrix[0, 1:]
            self.times = self.original_data_matrix[1:, 0]
            self.Y = self.original_data_matrix[1:, 1:]

    # time_slice and wavelength_slice are np.s_ slice objects
    def slice(self, time_slice, wavelength_slice):
        self.Y = self.Y[time_slice, wavelength_slice]
        self.wavelengths = self.wavelengths[wavelength_slice]
        self.times = self.times[time_slice]

    def get_time_dimension(self):
        return self.times.shape[0]

    def get_wavelength_dimension(self):
        return self.wavelengths.shape[0]

    def transpose(self):

        wavelengths = self.times
        times = self.wavelengths
        data = self.Y

        self.__init__(name=self.name, filename=self.filename)

        self.Y = data.T
        self.times = times
        self.wavelengths = wavelengths

        self.SVD()

        # update matrix D
        self.SVD_filter = self.SVD_filter

    @staticmethod
    def to_string(array, separator='\t', decimal_sep='.', new_line='\n'):
        list_array = array.tolist()

        list_array[0][0] = 'Wavelength'

        buffer = new_line.join(separator.join("{}".format(num) for num in row) for row in list_array)

        return buffer
