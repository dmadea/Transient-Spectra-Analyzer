import numpy as np
from scipy.integrate import odeint
from lmfit import Parameters
from abc import abstractmethod
# from gui_console import Console
# from scipy.stats import multivariate_normal
# from multiprocessing import Pool
# import math
# from scipy.interpolate import interp2d, interp1d
# from scipy.special import erfc
# from scipy.linalg import lstsq

from numba import njit, prange, vectorize

from scipy.linalg import svd
from math import erfc as math_erfc
from misc import find_nearest_idx

import matplotlib.pyplot as plt
from settings import Settings

from genericinputdialog import GenericInputDialog
from PyQt6.QtWidgets import QPushButton, QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox
from target_model import TargetModel
import glob, os
import scipy.constants as sc


# from concurrent.futures import ProcessPoolExecutor



## inspiration from https://github.com/Tillsten/skultrafast/blob/9544c3cc3c3c3fa46b728156198807e2b21ba24b/skultrafast/base_funcs/pytorch_fitter.py
def blstsq(A, B, alpha=0.001):
    """
    Batched linear least-squares by numpy with direct solve method with optional Tikhonov regularization
    to prevent errors in case of singular matrices.
    Minimizes sum ||A_i x_i - B_i||_2^2 + alpha||Ix||_2^2 for x, where A is a tensor (L, M, N), B is matrix (L, M)

    Parameters
    ----------
        A : shape(L, M, N)
        B : shape(L, M)
        alpha: float

    Returns
    -------
        tuple of (coefficients.T, fit.T)
    """

    # https://en.wikipedia.org/wiki/Tikhonov_regularization
    # (A^T A + alpha*I) X = A^T B, solve for X

    AT = np.transpose(A, (0, 2, 1))  # transpose of A
    ATA = np.matmul(AT, A)  # A.T @ A
    ATB = np.matmul(AT, B[..., None])  # A.T @ B

    if alpha != 0:
        I = alpha * np.eye(ATA.shape[-1])  # alpha * identity matrix
        ATA += I[None, ...]  # add to ATA

    X = np.linalg.solve(ATA, ATB)  # solve batched linear system of equations

    fit = np.matmul(A, X).squeeze().T

    return X.squeeze().T, fit

# copied from https://github.com/Tillsten/skultrafast/blob/23572ba9ea32238f34a8a15390fb572ecd8bc6fa/skultrafast/base_funcs/backend_tester.py
# © Till Stensitzki
# @vectorize(nopython=True, fastmath=False)
def fast_erfc(x):
    """
    Calculates the erfc near zero faster than
    the libary function, but has a bigger error, which
    is not a problem for us.
    Parameters
    ----------
    x: float
        The array
    Returns
    -------
    ret: float
        The erfc of x.
    """
    a1 = 0.278393
    a2 = 0.230389
    a3 = 0.000972
    a4 = 0.078108
    smaller = x < 0
    if smaller:
        x = x * -1.
    bot = 1 + a1 * x + a2 * x * x + a3 * x * x * x + a4 * x * x * x * x
    ret = 1. / (bot * bot * bot * bot)

    if smaller:
        ret = -ret + 2.

    return ret


@vectorize(nopython=True, fastmath=False)
def fold_exp(t, k, fwhm):

    w = fwhm / (2 * np.sqrt(np.log(2)))  # width

    if w > 0:
        return 0.5 * np.exp(k * (k * w * w / 4.0 - t)) * math_erfc(w * k / 2.0 - t / w)
    else:
        return np.exp(-t * k) if t >= 0 else 0

#
# def fold_exp_numpy(t, k, fwhm):
#     w = fwhm / (2 * np.sqrt(np.log(2)))  # width
#
#     return np.where(w > 0,
#             0.5 * np.exp(k * (k * w * w / 4.0 - t)) * erfc(w * k / 2.0 - t / w),
#             np.exp(-t * k) * np.heaviside(t, 1))


@vectorize(nopython=True, fastmath=False)
def photokin_factor(A):
    ln10 = np.log(10)
    ll2 = ln10 ** 2 / 2

    if A < 1e-3:
        return ln10 - A * ll2  # approximation with first two taylor series terms
    else:
        return (1 - np.exp(-A * ln10)) / A  # exact photokinetic factor


@njit(fastmath=True)
def _dc_dt_nb(c, t, I0, K, eps, V, t0):  # eps = spectra * l
    pA = c.reshape(-1, 1) * eps  # hadamard product - partial absorbances
    A = pA.sum(axis=0)  # total absorbance

    F = photokin_factor(A)  # (1-10^-A) / A

    product = K * (F * I0 * eps).sum(axis=-1)  # K @ diag(sum(F * I0 * eps * l))

    irr_on = 1 if t >= t0 else 0

    return irr_on * product.dot(c) / V  # final dot product / V


def get_target_C_profile(times, K, j):
    """K matrix, times to compute and j is initial population/concentration vector"""
    L, Q = np.linalg.eig(K)
    Q_inv = np.linalg.inv(Q)

    A2_T = Q * Q_inv.dot(j)  # Q @ np.diag(Q_inv.dot(j))

    t = times[:, None]
    C = np.exp(t * L[None, :]) * np.heaviside(t, 1)

    return C.dot(A2_T.T)


# virtual class that every model must inherits
class _Model(object):
    # n = 0  # number of visible species in model
    n = 0  # number of all possible species

    species_names = None
    # connectivity = None

    name = 'AB Model'
    description = "..."
    _class = '-class-'

    _err = 1e-8

    def __init__(self, times=None, connectivity=(0, 1, 2), wavelengths=None, C=None):
        self.times = times
        self.wavelengths = wavelengths
        self.C = C
        self._connectivity = connectivity
        self.method = 'MCR-ALS'

        self.init_times(times)
        self.target_model = None
        self.j = None  # j vector for target analysis
        self.params = None
        self.weights = []  # (wl_start, wl_end, weight) default weight 1

        # self.init_params()
        self.species_names = np.array(list('ABCDEFGHIJKL'), dtype=np.str)
        self.model_settigs_dialog = None

    @property
    def connectivity(self):
        return self._connectivity

    @connectivity.setter
    def connectivity(self, value):
        if not isinstance(value, (list, tuple)):
            raise ValueError("Connectivity array must be type of list or tuple.")
        self._connectivity = value

    def init_times(self, times):
        if times is not None:
            self.C = np.zeros((times.shape[0], self.n))
            self.times = times

    def calc_C(self, params=None, C_out=None):
        if params is not None:
            self.params = params

    def setup_target_models(self, widgets):
        models = []
        cbModels = QComboBox()
        for fpath in glob.glob(os.path.join(Settings.target_models_dir, '*.json'), recursive=True):
            fname = os.path.splitext(os.path.split(fpath)[1])[0]
            models.append(fpath)
            cbModels.addItem(fname)

        if self.target_model is not None:
            cbModels.setCurrentIndex(models.index(self.target_model.fpath))

        btnPlotModel = QPushButton('Plot Target Model')

        def _plot_model():
            t_model = TargetModel.load(models[cbModels.currentIndex()])
            t_model.plot_model()
        btnPlotModel.clicked.connect(_plot_model)

        widgets.append(['Target model:', cbModels])
        widgets.append([None, btnPlotModel])

        return models, cbModels

    def open_model_settings(self, show_target_model=False):
        pass
        # if GenericInputDialog.if_opened_activate():
        #     return
        #
        # self.model_settigs_dialog = GenericInputDialog()
        # self.model_settigs_dialog.show()
        # self.model_settigs_dialog.exec()

    def init_params(self):
        """Calls internally init_model_params and then transfers values from old params to new ones."""

        _params = self.init_model_params()

        if self.params is not None and _params is not None:
            for key, par in self.params.items():
                if key in _params:
                    _params[key].value = par.value
                    _params[key].vary = par.vary
                    _params[key].min = par.min
                    _params[key].max = par.max
                    _params[key].stderr = par.stderr

        self.params = _params

    def init_model_params(self):
        params = Parameters()
        return params

    def simulate_mod(self, D):
        pass

    def update_n(self, new_n=None):
        self.n = new_n if new_n is not None else self.n
        self.init_params()
        self.init_times(self.times)

    def update_T(self, new_T):
        pass

    def get_conc_matrix(self, C_out, connectivity=(1, 2, 3)):
        """Replaces the values in C_out according to calculated values based on conectivity"""
        if C_out is None:
            return self.C
        else:
            assert C_out.shape[-1] == len(connectivity)

            for i in range(len(connectivity)):
                # for values > 0 - 0 means MCR fit, replace the values, then 1 - A, 2 - B, 3 - C, etc.
                if connectivity[i] > 0:
                    C_out[..., i] = self.C[..., connectivity[i] - 1]

        return C_out

    def get_species_name(self, i):
        # i is index in range(n)
        return self.species_names[i]

    def cA(self, c0, k, n):
        if n == 1:  # first order
            return np.heaviside(self.times, 1) * c0 * np.exp(-self.times * k)
        elif n == 2:  # second order
            return np.heaviside(self.times, 1) * c0 / (1 + c0 * k * self.times)
        else:  # n-th order, this wont work for negative times c(t) = 1-n root of (c^(1-n) + k*(n-1)*t)
            expr_in_root = np.power(float(c0), 1 - n) + k * (n - 1) * self.times
            expr_in_root = expr_in_root.clip(min=0)  # set to 0 all the negative values
            return np.heaviside(self.times, 1) * np.power(expr_in_root, 1.0 / (1 - n))

    def cB(self, t, c0, k1, k2, k0=0):
        if np.abs(k1 - k2) < self._err:
            return np.heaviside(t, 1) * c0 * k1 * t * np.exp(-(k1 + k0) * t) / (k1 + k0)
        else:
            return np.heaviside(t, 1) * (k1 * c0 / (k2 - k1 - k0)) * (np.exp(-(k1 + k0) * t) - np.exp(-k2 * t))

    def get_weights(self):
        weights = np.ones((self.times.shape[0], self.wavelengths.shape[0]))
        for *rng, w in self.weights:
            idx0, idx1 = find_nearest_idx(self.wavelengths, rng)
            weights[:, idx0:idx1+1] *= w

        return weights

class _Femto(_Model):

    n_poly_chirp = 5  # order of polynomial for chirp_type: 'poly'
    n_exp_chirp = 2  # number of exponentials to describe chirp for mu_type = 'exp'
    n_partau = 2
    spectra = 'EADS'  # or 'EADS'

    def __init__(self, times=None, connectivity=(0, 1, 2), wavelengths=None,  method='femto'):
        super(_Femto, self).__init__(times, connectivity, wavelengths)
        self.method = method
        self.description = ""

        self.times = times
        self.C = None
        self._connectivity = connectivity
        self.init_times(times)
        self.species_names = np.array(list('ABCDEFGHIJ'), dtype=np.str)

        self.coh_spec = True
        self.coh_spec_order = 2

        self.partau = True

        # self.zero_coh_spec_range = [(460, 750)]  # zero coherent artifact in that wavelength range
        # self.weights = [(456, 475, 0.05)]  # (wl_start, wl_end, weight) default weight 1

        self.zero_coh_spec_range = []  # zero coherent artifact in that wavelength range

        self.weight_chirp = False
        self.w_of_chirp = 0.1
        self.t_radius_chirp = 0.2  # time radius around chirp / in ps

        self.chirp_type = 'exp'  # poly, exp chirp type
        self.spectra_choises = ['EADS', 'DADS']
        self.chirp_type_choices = ['poly', 'exp']

        self.update_n()
        self.ridge_alpha = 0.0001

        self.C_COH = None
        self.ST_COH = None

    def open_model_settings(self, show_target_model=False):
        if GenericInputDialog.if_opened_activate():
            return

        sbChripOrder = QSpinBox()
        sbChripOrder.setMinimum(0)
        sbChripOrder.setMaximum(10)
        sbChripOrder.setValue(self.n_poly_chirp)

        sbChripExp = QSpinBox()
        sbChripExp.setMinimum(0)
        sbChripExp.setMaximum(10)
        sbChripExp.setValue(self.n_exp_chirp)

        cbParTau = QCheckBox('Include Variable IRF-FWHM (partau)')
        cbParTau.setChecked(self.partau)
        sbParTau = QSpinBox()
        sbParTau.setMinimum(0)
        sbParTau.setMaximum(10)
        sbParTau.setValue(self.n_partau)

        def cbParTau_toggled():
            sbParTau.setEnabled(cbParTau.isChecked())
            btnPlotTau.setEnabled(cbParTau.isChecked())

        cbParTau.toggled.connect(cbParTau_toggled)

        cbCohSpec = QCheckBox('Include Coherent Artifacts')
        cbCohSpec.setChecked(self.coh_spec)
        sbCohSpecOrder = QSpinBox()
        sbCohSpecOrder.setMinimum(0)
        sbCohSpecOrder.setMaximum(4)
        sbCohSpecOrder.setValue(self.coh_spec_order)

        cbCohSpec.toggled.connect(lambda: sbCohSpecOrder.setEnabled(cbCohSpec.isChecked()))

        btnPlotTau = QPushButton('Plot FWHM')
        btnPlotTau.clicked.connect(self.plot_tau)

        cbSpectra = QComboBox()
        cbSpectra.addItems(self.spectra_choises)
        cbSpectra.setCurrentIndex(self.spectra_choises.index(self.spectra))

        cbChirp = QComboBox()
        cbChirp.addItems(self.chirp_type_choices)
        cbChirp.setCurrentIndex(self.chirp_type_choices.index(self.chirp_type))

        dsbAlpha = QDoubleSpinBox()
        dsbAlpha.setDecimals(5)
        dsbAlpha.setMinimum(0)
        dsbAlpha.setMaximum(10000)
        dsbAlpha.setSingleStep(0.1)
        dsbAlpha.setValue(self.ridge_alpha)

        widgets = [['Model of chirp:', cbChirp],
                   ['Chirp polynomial order:', sbChripOrder],
                   ['Number of exponentials (chirp):', sbChripExp],
                   [cbParTau, None],
                   ['Variable IRF-FWHM polynomial order:', sbParTau],
                   [btnPlotTau, None],
                   [cbCohSpec, None],
                   ["Number of Gaussian derivatives:", sbCohSpecOrder],
                   ["Used kinetic model:", cbSpectra],
                   ["Optional Ridge Regularization alpha:", dsbAlpha]
                   ]

        if show_target_model:
            models, cbModel = self.setup_target_models(widgets)

        def set_result():
            self.n_poly_chirp = int(sbChripOrder.value())
            self.n_exp_chirp = int(sbChripExp.value())
            self.partau = cbParTau.isChecked()
            self.n_partau = int(sbParTau.value())
            self.coh_spec = cbCohSpec.isChecked()
            self.coh_spec_order = int(sbCohSpecOrder.value())
            self.spectra = self.spectra_choises[cbSpectra.currentIndex()]
            self.chirp_type = self.chirp_type_choices[cbChirp.currentIndex()]
            self.ridge_alpha = float(dsbAlpha.value())
            if show_target_model:
                self.target_model = TargetModel.load(models[cbModel.currentIndex()])
                self.species_names = self.target_model.get_compartments()
            self.init_params()

        self.model_settigs_dialog = GenericInputDialog(widget_list=widgets, label_text="",
                                                       title=f'{self.name} settings',
                                                       set_result=set_result)
        self.model_settigs_dialog.show()
        self.model_settigs_dialog.exec()

    # @staticmethod
    # def conv_exp(t, k, fwhm):
    #
    #     w = fwhm / (2 * np.sqrt(np.log(2)))  # width
    #
    #     # if isinstance(w, np.ndarray):
    #     return np.where(w > 0,
    #             0.5 * np.exp(k * (k * w * w / 4.0 - t)) * erfc(w * k / 2.0 - t / w),
    #             np.exp(-t * k) * np.heaviside(t, 1))
    #     # else:
    #     #     if w > 0:
    #     #         return 0.5 * np.exp(k * (k * w * w / 4.0 - t)) * erfc(w * k / 2.0 - t / w)
    #     #     else:
    #     #         return np.exp(-t * k) * np.heaviside(t, 1)

    @staticmethod
    def simulate_model(t, K, j, mu=None, fwhm=0):
        # based on Ivo H.M. van Stokkum equation in doi:10.1016/j.bbabio.2004.04.011
        L, Q = np.linalg.eig(K)
        Q_inv = np.linalg.inv(Q)

        A2_T = Q * Q_inv.dot(j)  # Q @ np.diag(Q_inv.dot(j))

        _tau = fwhm[:, None, None] if isinstance(fwhm, np.ndarray) else fwhm

        if mu is not None:  # TODO !!! pořešit, ať je to obecne
            # C = _Femto.conv_exp(t[None, :, None] - mu[:, None, None], -L[None, None, :], _tau)
            C = fold_exp(t[None, :, None] - mu[:, None, None], -L[None, None, :], _tau)

        else:
            # C = _Femto.conv_exp(t[:, None], -L[None, :], fwhm)
            C = fold_exp(t[:, None], -L[None, :], fwhm)

        return C.dot(A2_T.T)

    def simulate_coh_gaussian(self, params=None, zero_coh_range=None):

        order = self.coh_spec_order

        fwhm = self.get_tau(params)
        mu = self.get_mu(params)

        s = fwhm / (2 * np.sqrt(2 * np.log(2)))  # sigma
        s = s[:, None, None] if isinstance(s, np.ndarray) else s

        tt = self.times[None, :, None] - mu[:, None, None]

        y = np.where(s > 0,
                     np.exp(-0.5 * tt * tt / (s * s)),
                     np.zeros((mu.shape[0], self.times.shape[0], 1)))

        y = np.tile(y, (1, 1, order + 1))

        if order > 0:  # first derivative
            y[..., 1] *= -tt.squeeze()

        if order > 1:  # second derivative
            y[..., 2] *= (tt * tt - s * s).squeeze()

        if order > 2:  # third derivative
            y[..., 3] *= (-tt * (tt * tt - 3 * s * s)).squeeze()

        if order > 3:  # fourth derivative
            y[..., 4] *= (tt ** 4 - 6 * tt * tt * s * s + 3 * s ** 4).squeeze()

        y_max = np.max(y, axis=1, keepdims=True)  # find maxima over time axis
        y_max[np.isclose(y_max, 0)] = 1  # values close to zero force to 1 to not divide by zero
        y /= y_max

        self.C_COH = y

        if zero_coh_range is not None:
            self.C_COH *= zero_coh_range[:, None, None]

        return self.C_COH

    def update_n(self, new_n=None, n_poly_chirp=None, n_exp_chirp=None):
        self.n_poly_chirp = n_poly_chirp if n_poly_chirp is not None else self.n_poly_chirp
        self.n_exp_chirp = n_exp_chirp if n_exp_chirp is not None else self.n_exp_chirp
        super(_Femto, self).update_n(new_n)

    def get_kin_pars(self, params=None):
        params = params if params is not None else self.params

        pars = [par.value for key, par in params.items()]

        chirp_params = 2 * self.n_exp_chirp if self.chirp_type is 'exp' else self.n_poly_chirp
        partau_params = self.n_partau if self.partau else 0

        fwhm = self.params['FWHM_lambda_c'].value  # fwhm at lambda_c
        taus = pars[3 + chirp_params + partau_params:]

        ks = 1 / np.asarray(taus)
        return fwhm, ks

    def get_lambda_c(self):
        return self.params['lambda_c'].value

    def get_tau(self, params=None):
        if self.wavelengths is None:
            return

        params = self.params if params is None else params

        FWHM_lambda_c = params['FWHM_lambda_c'].value

        if self.partau is False:
            return FWHM_lambda_c

        lambda_c = self.get_lambda_c()
        partaus = [params[f'partau_{i+1}'] for i in range(self.n_partau)]

        tau = np.ones(self.wavelengths.shape[0], dtype=np.float64) * FWHM_lambda_c

        for i in range(self.n_partau):
            tau += partaus[i] * ((self.wavelengths - lambda_c) / 100) ** (i + 1)

        return tau

    def plot_tau(self):
        plt.plot(self.wavelengths, self.get_tau())
        plt.xlabel('Wavelength / nm')
        plt.ylabel('IRF_FWHM / ps')
        plt.show()

    def get_parmu(self, params=None):
        params = self.params if params is None else params
        pars = [par[1].value for par in params.items()]

        lambda_c, mu_lambda_c = pars[:2]
        pars = pars[2:]

        n_pars = self.n_exp_chirp * 2 if self.chirp_type == 'exp' else self.n_poly_chirp

        return lambda_c, mu_lambda_c, pars[:n_pars]

    def get_mu(self, params=None):
        """Return the curve that defines chirp."""

        if self.wavelengths is None:
            return

        lambda_c, mu_lambda_c, pars = self.get_parmu(params)
        mu = np.ones(self.wavelengths.shape[0], dtype=np.float64) * mu_lambda_c
        x = self.wavelengths - lambda_c

        if self.chirp_type is 'exp':
            for i in range(self.n_exp_chirp):
                # mu += pars[2*i] * (1 - np.exp((lambda_c - self.wavelengths) / pars[2*i+1]))
                mu += pars[2*i] * np.exp(x * pars[2*i+1])
        else:
            for i in range(0, self.n_poly_chirp):
                mu += pars[i] * (x / 100) ** (i + 1)

        return mu

    def set_parmu(self, coefs, type='poly'):
        self.params['parmu_lambda_c'].value = coefs[0]

        if type == 'poly':
            for i in range(self.n_poly_chirp):
                self.params[f'parmu_{i+1}'].value = coefs[i+1]

        else:  # exponential coeffitients
            # coefs consist of - 1. is lambda_c, then are pairs of multiplier and lambdas

            for i in range(self.n_exp_chirp):
                self.params[f'mul_{i+1}'].value = coefs[i*2 + 1]
                self.params[f'lam_{i+1}'].value = coefs[i*2 + 2]

    def init_model_params(self):
        params = super(_Femto, self).init_model_params()

        params.add('lambda_c', value=433, min=0, max=1000, vary=False)
        params.add('parmu_lambda_c', value=1.539, min=-np.inf, max=np.inf, vary=True)

        if self.chirp_type is 'exp':
            for i in range(self.n_exp_chirp):
                params.add(f'mul_{i+1}', value=0.5, min=-np.inf, max=np.inf, vary=True)
                params.add(f'lam_{i+1}', value=0.01, min=-np.inf, max=np.inf, vary=True)

        else:  # polynomial by Ivo van Stokkum
            for i in range(self.n_poly_chirp):
                params.add(f'parmu_{i+1}', value=0.5, min=-np.inf, max=np.inf, vary=True)

        params.add('FWHM_lambda_c', value=0.1135, min=0, max=np.inf, vary=True)
        if self.partau:
            for i in range(self.n_partau):
                params.add(f'partau_{i+1}', value=0.01, min=-np.inf, max=np.inf, vary=True)

        return params

    def get_weights(self, params=None):
        weights = super(_Femto, self).get_weights()
        if self.weight_chirp:
            mu = self.get_mu(params)

            for i, t0 in enumerate(mu):
                idx0, idx1 = find_nearest_idx(self.times, [t0 - self.t_radius_chirp / 2, t0 + self.t_radius_chirp / 2])
                weights[idx0:idx1+1, i] *= self.w_of_chirp

        return weights


    def simulate_C_tensor(self, params=None):
        if params is None:
            params = self.params

        _C_tensor = self.calc_C(params)
        # n = _C_tensor.shape[-1]
        # ST = np.zeros((n + (self.coh_spec_order + 1 if self.coh_spec else 0), self.wavelengths.shape[0]))

        zero_coh_range = np.ones_like(self.wavelengths)
        for rng in self.zero_coh_spec_range:
            idx0, idx1 = find_nearest_idx(self.wavelengths, rng)
            zero_coh_range[idx0:idx1+1] = 0

        if self.coh_spec:
            _C_COH = self.simulate_coh_gaussian(zero_coh_range=zero_coh_range)
            _C_tensor = np.concatenate((_C_tensor, _C_COH), axis=-1) if _C_tensor is not None else _C_COH

        _C_tensor = np.nan_to_num(_C_tensor)

        return _C_tensor


    def simulate_mod(self, D, params=None):
        if D is None:
            raise ValueError('param D cannot be None')

        if params is None:
            params = self.params

        _C_tensor = self.simulate_C_tensor(params)

        ST, D_fit = blstsq(_C_tensor, D.T, self.ridge_alpha)  # solve batched least squares problem

        if self.coh_spec:
            self.ST_COH = ST[-self.coh_spec_order - 1:]

        # D_fit = np.matmul(_C_tensor, ST.T[..., None]).squeeze().T

        C = _C_tensor[0, :, :-self.coh_spec_order - 1] if self.coh_spec else _C_tensor[0]
        ST = ST[:-self.coh_spec_order - 1] if self.coh_spec else ST

        D_fit = np.nan_to_num(D_fit)
        return D_fit, C, ST


class _Photokinetic_Model(_Model):

    def __init__(self, times=None, connectivity=(0, 1, 2), ST=None, wavelengths=None, aug_matrix=None, rot_mat=True,
                 C=None, method='RFA'):
        super(_Photokinetic_Model, self).__init__(times, connectivity, wavelengths)

        self.fit_methods = [
            {'name': 'Resolving Factor Analysis', 'abbr': 'RFA'},
            {'name': 'HS-MCR-ALS (fit C profiles)', 'abbr': 'HS-MCR-ALS'},
            {'name': 'Variable Projection (fit full matrix)', 'abbr': 'VarPro'}
        ]

        self.method = self.fit_methods[0]['abbr']

        self.times = times
        self.C = C
        self._connectivity = connectivity
        self.init_times(times)
        self.species_names = np.array(list('ABCDEFGHIJ'), dtype=np.str)

        self.ST = ST
        self.interp_kind = 'quadratic'

        self.D = None  # original data

        self.aug_matrix = aug_matrix

        self.T = np.zeros((self.n, self.n), dtype=np.float64) if rot_mat else None
        self.U, self.Sigma, self.VT = None, None, None  # truncated SVD based on actual n components

        self.params = None

        self.update_n()

        self.description = ""

        _data_LEDs = np.genfromtxt(Settings.LED_sources_filepath, delimiter='\t', names=True, replace_space=False, deletechars='')

        self.LED_names = list(_data_LEDs.dtype.names[1:])
        data_mat = _data_LEDs.copy()
        data_mat.dtype = np.float64
        dat_mat = np.reshape(data_mat, (_data_LEDs.shape[0], len(self.LED_names) + 1))
        self.LEDs_wls = dat_mat[:, 0]
        self.LED_sources_norm = dat_mat[:, 1:]

        _data = np.genfromtxt(Settings.Photodiode_responsivity_filepath, delimiter='\t', skip_header=1)

        self.R_photodiode_values = _data[:, 1]
        self.R_photodiode_wls = _data[:, 0]

        self.selected_LED = 0

    def get_LED_source(self, index=None):
        index = self.selected_LED if index is None else index
        idx1, idx2 = find_nearest_idx(self.LEDs_wls, [self.wavelengths[0], self.wavelengths[-1]])
        return self.LED_sources_norm[idx1:idx2+1, index]

    def get_Photodiode_resposivity(self):
        idx1, idx2 = find_nearest_idx(self.R_photodiode_wls, [self.wavelengths[0], self.wavelengths[-1]])
        return self.R_photodiode_values[idx1:idx2+1]

    def get_q_rel(self):
        # q_rel = lambda / (N_A * h * c * R)
        return self.wavelengths * 1e-9 / (sc.N_A * sc.h * sc.c * self.get_Photodiode_resposivity())

    def plot_LED_source(self, index=None):
        index = self.selected_LED if index is None else index
        plt.plot(self.LEDs_wls, self.LED_sources_norm[:, index], label=f'LED {self.LED_names[index]}')
        plt.xlabel("Wavelength / nm")
        plt.ylabel("PDF / nm$^{-1}$")
        plt.legend()
        plt.show()

    def open_model_settings(self, show_target_model=False):
        if GenericInputDialog.if_opened_activate():
            return

        led_comboBox = QComboBox()
        led_comboBox.addItems(self.LED_names)
        led_comboBox.setCurrentIndex(self.selected_LED)
        btn_show_LED = QPushButton('Show Spectrum')

        btn_show_LED.clicked.connect(lambda: self.plot_LED_source(int(led_comboBox.currentIndex())))

        method_comboBox = QComboBox()
        method_comboBox.addItems(map(lambda m: m['name'], self.fit_methods))

        curr_idx_method = 0
        for i, m in enumerate(self.fit_methods):
            if m['abbr'] is self.method:
                curr_idx_method = i
                break

        method_comboBox.setCurrentIndex(curr_idx_method)

        widgets = [['LED for irradiation (maximum):', led_comboBox],
                   ['', btn_show_LED],
                   ['Fitting method:', method_comboBox]]

        def set_result():
            self.selected_LED = int(led_comboBox.currentIndex())
            self.method = self.fit_methods[int(method_comboBox.currentIndex())]['abbr']
            self.init_params()

        self.model_settigs_dialog = GenericInputDialog(widget_list=widgets, label_text="",
                                                       title=f'{self.name} settings',
                                                       set_result=set_result)
        self.model_settigs_dialog.show()
        self.model_settigs_dialog.exec()

    def update_n(self, new_n=None):
        super(_Photokinetic_Model, self).update_n(new_n)

        self.T = np.zeros((self.n, self.n + 1), dtype=np.float64) if self.T is not None else None

        if self.aug_matrix:
            U, S, VT = svd(self.aug_matrix.aug_mat, full_matrices=False)
            self.U, self.Sigma, self.VT = U[:, :self.n + 1], np.diag(S[:self.n + 1]), VT[:self.n + 1, :]

    def get_T(self, params=None):
        if self.T is None:
            return

        n = self.n
        params = self.params if params is None else params

        assert type(n) is int

        pars = [par[1].value for par in params.items()]
        self.T = np.asarray(pars[:n*(n+1)]).reshape((n, n+1))

        return self.T

    def get_photokin_params(self, params=None):
        params = self.params if params is None else params
        pars = [par[1].value for par in params.items()]

        if self.method is 'RFA':
            pars = pars[self.n*(self.n + 1):]

        return pars

    def init_model_params(self):
        params = super(_Photokinetic_Model, self).init_model_params()

        if self.method is not 'RFA':
            return params

        if self.T is not None:
            for i in range(self.n):
                for j in range(self.n + 1):
                    params.add(f't_{i+1}{j+1}', value=1 if i == j else 0, min=-np.inf, max=np.inf, vary=True)

        return params

    def update_T(self, new_T):
        if self.method is not 'RFA':
            return

        assert isinstance(new_T, np.ndarray) and new_T.shape == self.T.shape
        self.T = new_T

        # update params
        for i in range(self.n):
            for j in range(self.n):
                self.params[f't_{i + 1}{j + 1}'].value = self.T[i, j]

    @staticmethod
    def simul_photokin_model_wl_dep(I0, c0, K, eps, times, V=0.003, l=1, t0=0):
        """
        with wl dependence of K
        I0 - irradiation spectrum scalled by q0 = PDF * q0 so that integral(I0) = q0
        eps - spectra (= epsilons)
        t0 - time of start of irradiaiton
        l - length of cuvette
        V - volume in the cuvette in L
        c0 - total initial concentration vector of compounds
        """

        assert c0.shape[0] == K.shape[-1]

        def dc_dt(c, t):
            p_A = c[:, None] * eps * l  # hadamard product
            A = p_A.sum(axis=0)  # dot product

            FI0 = p_A * photokin_factor(A) * I0

            # # w x n x n   x   w x n x 1
            product = np.matmul(K, FI0.T[..., None])  # w x n x 1

            irr_on = 1 if t >= t0 else 0

            return irr_on * product.sum(axis=0).squeeze() / V

        result = odeint(dc_dt, c0, times)

        return result

    @staticmethod
    def simul_photokin_model(I0, c0, K=None, eps=None, times=None, V=0.003, l=1, t0=0, R=0.036,
                             use_backref_correction=True,  scaling_coef=1e6):
        """
        no wl dependence of K
        I0 - irradiation spectrum scalled by q0 = PDF * q0 so that integral(I0) = q0
        eps - spectra (= epsilons)
        t0 - time of start of irradiaiton
        l - length of cuvette
        V - volume in the cuvette in L
        c0 - total initial concentration vector of compounds
        R - reflectivity of cuvette, for fused silica, it is 0.036 around 400 nm
        use_Ieff_correction - if True, correction for effective irradiance will be used
        scaling_coef is there to keep numerical integrator stable, if the simulated profiles has low values,
        it can cause some numerical errors in the integrator, so the spectra are scaled down but initial concentrations
        and photon flux is scaled up by this coefficient, thereby making the result equal to non-scaled simulation
        """

        c0 = np.asarray(c0)

        assert c0.shape[0] == K.shape[0]
        ln10 = np.log(10)

        eps_s = eps / scaling_coef
        I0_s = I0 * scaling_coef
        c0_s = c0 * scaling_coef

        def dc_dt(c, t):
            p_A = c[:, None] * eps_s * l  # hadamard product - partial absorbance
            A = p_A.sum(axis=0)  # total absorbance

            F = photokin_factor(A)  # (1-10^-A) / A

            # Ieff = I0 * (1-R)*(1 + R*T) = I_solvent * (1 + R*T) / (1 - R), where T is transmittance
            # I0 is Isolvent, so irradiance measured after cuvette with pure solvent
            Ieff = I0_s * (1 + R * np.exp(-ln10 * A)) / (1 - R) if use_backref_correction else I0_s

            product = K * (F * Ieff * eps_s * l).sum(axis=-1)  # K @ sum(diag(F * I0 * eps * l))

            irr_on = 1 if t >= t0 else 0

            return irr_on * product.dot(c) / V  # final dot product / V

        # result = odeint(_dc_dt_nb, c0 * scaling_coef, times, args=(I0 * scaling_coef, K, eps * l / scaling_coef, V, t0)) if use_numba else odeint(dc_dt, c0, times)
        result = odeint(dc_dt, c0_s, times)

        return result / scaling_coef

    #
    # @staticmethod
    # def simulate(q0, c0, K, I_source, wavelengths=None, times=None, eps=None, V=0.003, l=1,  D=None, t0=0):
    #     """
    #     c0 is concentration vector at time, defined in times array as first element (initial condition), eps is vector of molar abs. coefficients,
    #     I_source is spectrum of irradiaiton source if this was used,
    #     if not, w_irr as irradiaton wavelength must be specified, K is transfer matrix, l is length of a cuvette, default 1 cm
    #     times are times for which to simulate the kinetics
    #     """
    #     # n = eps.shape[0]  # eps are epsilons - n x w matrix, where n is number of species and w is number of wavelengths
    #     # assert n == K.shape[0] == K.shape[1]
    #     c0 = np.asarray(c0)
    #
    #     c_tot = c0.sum()
    #
    #     # get absorbances from real data
    #     abs_at = interp2d(wavelengths, times, D, kind='linear', copy=True)
    #
    #     # const = l * np.log(10)
    #     # tol = 1e-3
    #
    #     def dc_dt(c, t):
    #
    #         # c_eps = c[..., None] * eps  # hadamard product
    #
    #         _c = np.append(c, c_tot - c.sum())
    #
    #         c_dot_eps = abs_at(wavelengths, t)
    #
    #         # I = c_eps * Half_Bilirubin_Multiset_Half.photokin_factor(c_dot_eps) * I_source
    #         FI0 = _Photokinetic_Model.photokin_factor(c_dot_eps) * I_source
    #
    #         tensor = FI0[:, None, None] * K * eps.T[:, None, :]  # I0 * F * K x diag(epsilon)
    #
    #         return q0 / V * tensor.sum(axis=0).dot(_c)
    #
    #         #
    #         # # w x n x n   x   w x n x 1
    #         # product = np.matmul(K, I.T[..., None])  # w x n x 1
    #         #
    #         # irr_on = 1 if t >= t0 else 0
    #         #
    #         # return irr_on * q0 / V * product.sum(axis=0).squeeze()
    #
    #     result = odeint(dc_dt, c0, times)
    #
    #     forth_comp = c_tot - result.sum(axis=1, keepdims=True)
    #
    #     result = np.hstack((result, forth_comp))
    #
    #     return result

    def calc_C(self, params=None, C_out=None):
        super(_Photokinetic_Model, self).calc_C(params)

        # if self.ST is None:
        #     raise ValueError("Spectra matrix must not be None.")

        return C_out


class PumpProbeCrossCorrelation(_Femto):

    name = 'Pump-Probe Cross-Correlation'
    _class = 'Femto'

    def open_model_settings(self, show_target_model=False):
        super(PumpProbeCrossCorrelation, self).open_model_settings(False)

    def calc_C(self, params=None, C_out=None):
        super(PumpProbeCrossCorrelation, self).calc_C(params, C_out)

        return None


class Global_Analysis_Femto(_Femto):

    name = 'Global Analysis'
    _class = 'Femto'
    use_numpy = True

    def init_model_params(self):
        params = super(Global_Analysis_Femto, self).init_model_params()

        # evolution model
        if self.spectra == 'EADS':
            self.species_names = [f'EADS{i + 1}' for i in range(self.n)]
            for i in range(self.n):
                # sec_label = self.species_names[i+1] if i < self.n - 1 else ""
                params.add(f'tau_{self.species_names[i]}', value=10**(i-1), min=0, max=np.inf)

        else:  # decay model
            self.species_names = [f'DADS{i + 1}' for i in range(self.n)]
            for i in range(self.n):
                params.add(f'tau_{self.species_names[i]}', value=10**(i-1), min=0, max=np.inf)

        return params

    def open_model_settings(self, show_target_model=False):
        super(Global_Analysis_Femto, self).open_model_settings(False)


    def calc_C(self, params=None, C_out=None):
        super(Global_Analysis_Femto, self).calc_C(params, C_out)

        fwhm, ks = self.get_kin_pars(params)
        mu = self.get_mu(params)
        n = self.n
        fwhm = self.get_tau(params)  # fwhm

        if self.spectra == 'EADS':
            K = np.zeros((n, n))
            for i in range(n):
                K[i, i] = -ks[i]
                if i < n - 1:
                    K[i + 1, i] = ks[i]

            j = np.zeros(n)
            j[0] = 1
            self.C = self.simulate_model(self.times, K, j, mu, fwhm)

        else:  # for DADS
            _tau = fwhm[:, None, None] if isinstance(fwhm, np.ndarray) else fwhm
            self.C = fold_exp(self.times[None, :, None] - mu[:, None, None], ks[None, None, :], _tau)

        return self.get_conc_matrix(C_out, self._connectivity)


class Target_Analysis_Femto(_Femto):

    name = 'Target Analysis'
    _class = 'Femto'

    def init_model_params(self):
        params = super(Target_Analysis_Femto, self).init_model_params()

        if self.target_model:
            for par_name, rate in self.target_model.get_names_rates():
                par_name = 'tau' + par_name[1:]
                params.add(par_name, value=1/rate, min=0, max=np.inf)

        return params

    def open_model_settings(self, show_target_model=False):
        super(Target_Analysis_Femto, self).open_model_settings(show_target_model=True)

    def calc_C(self, params=None, C_out=None):
        super(Target_Analysis_Femto, self).calc_C(params, C_out)

        fwhm, ks = self.get_kin_pars(params)
        mu = self.get_mu(params)
        n = self.n
        fwhm = self.get_tau(params)  # fwhm

        if self.j is None:
            self.j = np.zeros(n)
            self.j[0] = 1

        self.target_model.set_rates(ks)
        K = self.target_model.build_K_matrix()

        self.C = self.simulate_model(self.times, K, self.j, mu, fwhm)

        return self.get_conc_matrix(C_out, self._connectivity)


class Target_Analysis_Z_Femto(_Femto):

    name = 'Target Analysis Z isomer'
    _class = 'Femto'

    def __init__(self, times=None, connectivity=(0, 1, 2), wavelengths=None, method='femto'):
        super(Target_Analysis_Z_Femto, self).__init__(times=times,
                                                      connectivity=connectivity,
                                                      wavelengths=wavelengths,
                                                      method=method)

        self.solvation = True
        self.n_upsample = 3

    def init_model_params(self):
        params = super(Target_Analysis_Z_Femto, self).init_model_params()

        # self.params.add('phi', value=0.5, min=0, max=1, vary=False)
        params.add('tau_AB', value=0.25, min=0, max=np.inf)
        params.add('tau_BA', value=0.50, min=0, max=np.inf)
        params.add('tau_AB_C', value=5.65, min=0, max=np.inf)
        params.add('tau_CD', value=14.5, min=0, max=np.inf)
        params.add('tau_sol', value=1.9, min=0, max=np.inf)
        params.add('tau_diff', value=0.92, min=0, max=np.inf)

        return params

    def solvation_rates(self, t, k_AB, k_BA, k_diff, k_sol):
        _k_AB = k_AB + (k_diff * (1 - np.exp(-t * k_sol)) if self.solvation else 0)
        _k_BA = k_BA - (k_diff * (1 - np.exp(-t * k_sol)) if self.solvation else 0)

        return _k_AB, _k_BA

    def plot_solvation_rates(self):
        fwhm, ks = self.get_kin_pars(self.params)
        k_AB, k_BA, k_ABC, k_CD, k_sol, k_diff = ks
        _k_AB, _k_BA = self.solvation_rates(self.times, k_AB, k_BA, k_diff, k_sol)

        plt.plot(self.times, _k_AB, label='$k_{AB}$')
        plt.plot(self.times, _k_BA, label='$k_{BA}$')
        plt.show()

    @staticmethod
    def gauss(t, fwhm=1):
        assert fwhm != 0
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # https://en.wikipedia.org/wiki/Gaussian_function
        return np.exp(-t * t / (2 * sigma * sigma)) / (sigma * np.sqrt(2 * np.pi))

    @staticmethod
    def upsample_points(x, n=3):
        t_u = []
        for t1, t2 in zip(x[:-1], x[1:]):
            t_diff = t2 - t1
            for i in range(n):
                t_u.append(t1 + t_diff * i / n)
        t_u.append(x[-1])
        return np.asarray(t_u)

    @staticmethod
    @njit(fastmath=True)
    def func_nb(c, t, j, t0, fwhm, rates, K_temp):
        k_AB, k_BA, k_ABC, k_CD, k_sol, k_diff = rates
        _t = t - t0

        _k_AB = k_AB + k_diff * (1 - np.exp(-_t * k_sol))
        _k_BA = k_BA - k_diff * (1 - np.exp(-_t * k_sol))

        K_temp[0, 0] = -_k_AB - k_ABC
        K_temp[0, 1] = _k_BA
        K_temp[1, 0] = _k_AB
        K_temp[1, 1] = -_k_BA - k_ABC

        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        gauss = np.exp(-_t * _t / (2 * sigma * sigma)) / (sigma * np.sqrt(2 * np.pi))

        return K_temp.dot(c) + j * gauss


    def calc_C(self, params=None, C_out=None):
        super(Target_Analysis_Z_Femto, self).calc_C(params, C_out)

        fwhm, ks = self.get_kin_pars(params)
        mu = self.get_mu(params)
        n = self.n  # n must be 4
        fwhm = self.get_tau(params)  # fwhm

        k_AB, k_BA, k_ABC, k_CD, k_sol, k_diff = ks

        K = np.asarray([[-k_AB - k_ABC, k_BA, 0, 0],
                        [k_AB, -k_BA - k_ABC, 0, 0],
                        [k_ABC, k_ABC, -k_CD, 0],
                        [0, 0, k_CD, 0]])

        # def func(c, t, j, t0, fwhm):
        #     _k_AB, _k_BA = self.solvation_rates(t - t0, k_AB, k_BA, k_diff, k_sol)
        #
        #     K = np.asarray([[-_k_AB - k_ABC, _k_BA, 0, 0],
        #                     [_k_AB,   -_k_BA - k_ABC,  0, 0],
        #                     [k_ABC, k_ABC, -k_CD, 0],
        #                     [0,        0,  k_CD, 0]])
        #     return K.dot(c) + j * self.gauss(t - t0, fwhm)

        j = np.zeros(n)
        j[0] = 1

        t_upsampled = self.upsample_points(self.times, self.n_upsample) if self.n_upsample > 1 else self.times

        self.C = np.zeros((mu.shape[0], self.times.shape[0], n), dtype=np.float64)

        # with ProcessPoolExecutor() as exe:
        #     exe.submit()

        for i in range(mu.shape[0]):
            _fwhm = fwhm[i] if isinstance(fwhm, np.ndarray) else fwhm
            # self.C[i, ...] = odeint(func, np.zeros(n), t_upsampled, args=(j, mu[i], _fwhm))[::self.n_upsample]
            self.C[i, ...] = odeint(self.func_nb, np.zeros(n), t_upsampled, args=(j, mu[i], _fwhm, ks, K))[::self.n_upsample]


        return self.get_conc_matrix(C_out, self._connectivity)


class First_Order_Sequential_Model(_Model):

    name = 'Sequential model (1st order)'
    _class = 'Nano'

    def init_model_params(self):
        params = Parameters()
        params.add('c0', value=1, min=0, max=np.inf, vary=False)

        for i in range(self.n):
            sec_label = self.species_names[i+1] if i < self.n - 1 else ""
            params.add(f'tau_{self.species_names[i]}{sec_label}', value=1+i**2, min=0, max=np.inf)

        return params

    @staticmethod
    def get_EAS(t, ks):
        # based on Ivo H.M. van Stokkum equation in doi:10.1016/j.bbabio.2004.04.011
        # c_l = sum_{j=1}^l  b_jl * exp(-k_j * t)
        # for j < l: b_jl = b_{j, l-1} * k_{l-1} / (k_l - k_j)
        n = ks.shape[0]
        C = np.exp(-t[:, None] * ks[None, :])

        bjl = np.triu(np.ones((n, n)))  # make triangular upper matrix

        k_prod = np.cumprod(ks[:-1])  # products of rate constants

        k_mat = ks[None, :] - ks[:, None]  # differences between rate constants
        k_mat[k_mat == 0] = 1  # set zero differences to 1, because of calculation of products
        k_mat = np.cumprod(k_mat, axis=1)  # make product of them
        k_mat[:, 1:] = k_prod / k_mat[:, 1:]  # combine with rate constants

        bjl *= k_mat

        return C.dot(bjl)

    def calc_C(self, params=None, C_out=None):
        super(First_Order_Sequential_Model, self).calc_C(params, C_out)

        c0, *taus = [par[1].value for par in self.params.items()]

        self.C = self.get_EAS(self.times, 1 / np.asarray(taus)) * np.heaviside(self.times[:, None], 1)

        return self.get_conc_matrix(C_out, self._connectivity)


class First_Order_Parallel_Model(_Model):

    name = 'Parallel model (1st order)'
    _class = 'Nano'

    def init_model_params(self):
        params = Parameters()
        params.add('c0', value=1, min=0, max=np.inf, vary=False)

        for i in range(self.n):
            self.params.add(f'tau_{self.species_names[i]}', value=1+i**2, min=0, max=np.inf)

        return params

    def calc_C(self, params=None, C_out=None):
        super(First_Order_Parallel_Model, self).calc_C(params, C_out)

        c0, *taus = [par[1].value for par in self.params.items()]
        ks = 1 / np.asarray(taus)

        self.C = c0 * np.exp(-self.times[:, None] * ks[None, :]) * np.heaviside(self.times[:, None], 1)

        return self.get_conc_matrix(C_out, self._connectivity)


class First_Order_Target_Model(_Model):

    name = 'Target model (1st order)'
    _class = 'Nano'

    def init_model_params(self):
        params = Parameters()
        params.add('c0', value=1, min=0, max=np.inf, vary=False)

        if self.target_model:
            for par_name, rate in self.target_model.get_names_rates():
                params.add(par_name, value=rate, min=0, max=np.inf)

        return params

    def open_model_settings(self, show_target_model=False):
        if GenericInputDialog.if_opened_activate():
            return

        widgets = []
        models, cbModel = self.setup_target_models(widgets)

        def set_result():
            self.target_model = TargetModel.load(models[cbModel.currentIndex()])
            self.species_names = self.target_model.get_compartments()
            self.init_params()

        self.model_settigs_dialog = GenericInputDialog(widget_list=widgets, label_text="",
                                                       title=f'{self.name} settings',
                                                       set_result=set_result)
        self.model_settigs_dialog.show()
        self.model_settigs_dialog.exec()

    def calc_C(self, params=None, C_out=None):
        super(First_Order_Target_Model, self).calc_C(params, C_out)

        c0, *ks = [par[1].value for par in self.params.items()]

        self.target_model.set_rates(ks)
        K = self.target_model.build_K_matrix()
        # print(K)

        if self.j is None or self.j.shape[0] != K.shape[0]:
            self.j = np.zeros(K.shape[0])
            self.j[0] = 1

        self.C = get_target_C_profile(self.times, K, self.j * c0)

        return self.get_conc_matrix(C_out, self._connectivity)


    #
    # def calc_C(self, params=None, C_out=None):
    #     super(First_Order_Target_Model, self).calc_C(params, C_out)
    #
    #     c0, *taus = [par[1].value for par in self.params.items()]
    #     ks = 1 / np.asarray(taus)
    #
    #     self.C = c0 * np.exp(-self.times[:, None] * ks[None, :]) * np.heaviside(self.times[:, None], 1)
    #
    #     return self.get_conc_matrix(C_out, self._connectivity)


class Sequential_Model_FK(_Photokinetic_Model):

    name = 'Sequential model'
    _class = 'Steady state photokinetics'

    def __init__(self):
        super(Sequential_Model_FK, self).__init__()

        self.Irr_norm = None  # normalized irradiation spectrum
        self.use_numba = True

    def init_model_params(self):
        params = super(Sequential_Model_FK, self).init_model_params()
        params.add('c0', value=1e-3, min=0, max=np.inf, vary=False)  # initial concentration of species A
        params.add('q0', value=1e-8, min=0, max=np.inf, vary=False)  # total photon flux
        params.add('l', value=1, min=0, max=np.inf, vary=False)  # length of cuvette
        params.add('V', value=1e-3, min=0, max=np.inf, vary=False)  # volume of solution in cuvette
        params.add('t0', value=0, min=0, max=np.inf, vary=False)  # time of start of irradiation

        for i in range(self.n):
            sec_label = self.species_names[i + 1] if i < self.n - 1 else ""
            params.add(f'Phi_{self.species_names[i]}{sec_label}', value=np.round(np.exp(-i*0.5), 2), min=0, max=1)

        return params

    def calc_C(self, params=None, C_out=None):
        super(Sequential_Model_FK, self).calc_C(params, C_out)

        c0, q0, l, V, t0, *phis = self.get_photokin_params()
        n = self.n

        K = np.zeros((n, n), dtype=np.float64)  # setup a model matrix

        for i in range(n):  # fill the matrix with sequential model QYs
            K[i, i] = -phis[i]
            if i < n - 1:
                K[i+1, i] = phis[i]

        c0_vec = np.zeros(n, dtype=np.float64)
        c0_vec[0] = c0  # initial concentration vector

        irr = self.Irr_norm if self.Irr_norm is not None else self.get_LED_source()

        self.C = self.simul_photokin_model(irr * q0, c0_vec, K, self.ST, self.times, V=V, l=l, t0=t0,
                                           use_numba=self.use_numba)

        return self.get_conc_matrix(C_out, self._connectivity)


class CisTransIsomerization(_Photokinetic_Model):

    name = 'Cis Trans isomerization'
    _class = 'Steady state photokinetics'

    def __init__(self):
        super(CisTransIsomerization, self).__init__()

        LUX_file = r'C:\Users\dominik\Documents\RealTimeSync\Projects\2020-Bilirubin - 2nd half\UV-VIS\QY\new setup final\LEDs norm.txt'

        _data_irr = np.genfromtxt(LUX_file, delimiter='\t')
        leds = _data_irr[1:, 1:]
        leds /= np.trapz(leds, axis=0)  # normalize to unit area
        self.LED385 = leds[:, 0]
        self.LED405 = leds[:, 1]
        self.LED415 = leds[:, 2]
        self.LED450 = leds[:, 3]

        path = r'C:\Users\dominik\Documents\RealTimeSync\Projects\2020-Bilirubin - 2nd half\UV-VIS\QY\q rel.txt'

        self.q_rel = np.genfromtxt(path, delimiter='\t')[1:, 1]
        self.overlap385 = np.trapz(self.LED385 * self.q_rel)
        self.overlap405 = np.trapz(self.LED405 * self.q_rel)
        self.overlap415 = np.trapz(self.LED415 * self.q_rel)
        self.overlap450 = np.trapz(self.LED450 * self.q_rel)

        self.use_numba = False

    def init_model_params(self):
        params = super(CisTransIsomerization, self).init_model_params()
        # params.add('c0', value=1e-3, min=0, max=np.inf, vary=False)  # initial concentration of species A
        params.add('IZ', value=75.9e-6, min=0, max=np.inf, vary=False)  # total photon flux
        params.add('IE', value=75.9e-6, min=0, max=np.inf, vary=False)  # total photon flux
        params.add('l', value=1, min=0, max=np.inf, vary=False)  # length of cuvette
        params.add('V', value=2.5e-3, min=0, max=np.inf, vary=False)  # volume of solution in cuvette
        params.add('t0Z', value=0, min=0, max=np.inf, vary=False)  # time of start of irradiation
        params.add('t0E', value=0, min=0, max=np.inf, vary=False)  # time of start of irradiation
        params.add('xZ', value=0.234, min=0, max=np.inf, vary=False)  # time of start of irradiation
        # xZ = 0.234 from last measurement of 2E

        # params.add('t02', value=0, min=0, max=np.inf, vary=False)  # time of start of irradiation

        params.add('Phi_ZE', value=0.2, min=0, max=1, vary=True)  # time of start of irradiation
        params.add('Phi_EZ', value=0.2, min=0, max=1, vary=True)  # time of start of irradiation
        params.add('Phi_ZED', value=0.005, min=0, max=1, vary=True)  # time of start of irradiation

        return params

    def get_conc_vector(self, spectrum, populations):
        pop = np.asarray(populations, dtype=np.float64)
        pop /= pop.sum()
        assert pop.shape[0] == self.ST.shape[0]

        ST_avrg = (pop[:, None] * self.ST).sum(axis=0)

        STST_sum = (ST_avrg * ST_avrg).sum()

        c0 = (spectrum * ST_avrg).sum() / STST_sum

        return pop * c0

    def calc_C(self, params=None, C_out=None):
        super(CisTransIsomerization, self).calc_C(params, C_out)

        IZ, IE, l, V, t0Z, t0E, xZ, Phi_ZE, Phi_EZ, Phi_ZED = self.get_photokin_params()

        K = np.asarray([[-Phi_ZE - Phi_ZED, Phi_EZ, 0],
                        [Phi_ZE, -Phi_EZ - Phi_ZED, 0],
                        [Phi_ZED, Phi_ZED, 0]])

        irr = self.LED385

        args = [
            [IZ * self.overlap385, [1, 0, 0], t0Z],
            [IE * self.overlap385, [xZ, 1 - xZ, 0], t0E]
        ]

        for i in range(len(args)):
            q0, pop, t0 = args[i]

            s, e = self.aug_matrix._C_indiv_range(i)
            mat = self.aug_matrix.matrices[i, 0]

            c0 = self.get_conc_vector(mat.Y[0], pop)

            C_out[s:e, :] = self.simul_photokin_model(q0 * irr, c0, times=mat.times, eps=self.ST, V=V, t0=t0, l=1, K=K,
                                                      use_backref_correction=True)

        return C_out


class AB_Model(_Model):
    """Simple A->B model, default is 1st order reaction, 2nd order and n-th order can be selected as well. Also,
    user can define whether both A and B are visible, or only A or B is visible. Default is [True, False] - only A is
     visible"""

    order = '1st'
    n = 2
    name = 'A→B (variable order)'
    _class = 'Nano'

    def __init__(self, times=None, order='1st'):
        """order == '1st' - 1st order kinetics (default)
        order == '2nd' - 2ns order reaction kinetics
        order == 'n-th' - n-th order reaction kinetics

        For 1st order, c0=1 in params is set to fixed as default."""
        self.order = order

        super(AB_Model, self).__init__(times)

        self.species_names = np.array(list('AB'), dtype=np.str)

        # self.description = "Simple A->B model of n-th order. d[A]/dt = -k[A]^n, [A]_0 = c_0"

    def init_params(self):
        self.params = Parameters()

        self.params.add('c0', value=1, min=0, max=np.inf, vary=True)
        self.params.add('k', value=1, min=0, max=np.inf)

        if self.order == '1st':
            self.params['c0'].vary = False
            self.params.add('n', value=1, min=0, max=10, vary=False)
        elif self.order == '2nd':
            self.params.add('n', value=2, min=0, max=10, vary=False)
        else:
            self.params['c0'].vary = False
            self.params.add('n', value=1.1, min=0, max=10)

    def calc_C(self, params=None, C_out=None):
        super(AB_Model, self).calc_C(params, C_out)

        c0, k, n = self.params['c0'].value, self.params['k'].value, self.params['n'].value

        self.C[:, 0] = self.cA(c0, k, n)  # A
        self.C[:, 1] = np.heaviside(self.times, 1) * (c0 - self.C[:, 0])  # B

        return self.get_conc_matrix(C_out, self._connectivity)


class AB_mixed12_Model(_Model):
    """Mixed first and second order kinetics, d[A]/dt = -k1[A] - k2[A]^2"""
    n = 2
    name = 'A→B (mixed 1st and 2nd order)'
    _class = 'Nano'

    def __init__(self, times=None):
        super(AB_mixed12_Model, self).__init__(times)

        # self.species_names = np.array(list('AB'), dtype=np.str)

        self.description = "A->B model of mixed first and second order. d[A]/dt = -k1[A] - k2[A]^2, [A]_0 = c_0"

    def init_params(self):
        self.params = Parameters()
        self.params.add('c0', value=1, min=0, max=np.inf, vary=False)
        self.params.add('k1', value=1, min=0, max=np.inf)
        self.params.add('k2', value=0.2, min=0, max=np.inf)

    def calc_C(self, params=None, C_out=None):
        super(AB_mixed12_Model, self).calc_C(params, C_out)

        c0, k1, k2 = self.params['c0'].value, self.params['k1'].value, self.params['k2'].value

        self.C[:, 0] = np.heaviside(self.times, 1) * c0 * k1 / (
                (c0 * k2 + k1) * np.exp(k1 * self.times) - c0 * k2)  # A
        self.C[:, 1] = np.heaviside(self.times, 1) * (c0 - self.C[:, 0])  # B

        return self.get_conc_matrix(C_out, self._connectivity)


class ABDE_Model(_Model):
    """ABDE kinetic model, first order."""

    n = 4
    name = 'TA-2 species: A→B, C→D (1st order)'
    _class = 'Nano'

    def __init__(self, times=None):
        super(ABDE_Model, self).__init__(times)

        # self.species_names = np.array(list('ABCD'), dtype=np.str)

        self.description = "TODOchange++++Simple A->B->C model of 1st order. d[A]/dt = -k1[A] - k2[A]^2, [A]_0 = c_0"

    def init_params(self):
        self.params = Parameters()
        self.params.add('c0', value=1, min=0, max=np.inf, vary=False)
        self.params.add('c1', value=1, min=0, max=np.inf, vary=False)
        self.params.add('k1', value=0.9, min=0, max=np.inf)
        self.params.add('k2', value=0.7, min=0, max=np.inf)

    def calc_C(self, params=None, C_out=None):
        super(ABDE_Model, self).calc_C(params)

        c0, c1, k1, k2 = [par[1].value for par in self.params.items()]

        self.C[:, 0] = np.heaviside(self.times, 1) * c0 * np.exp(-self.times * k1)
        self.C[:, 1] = np.heaviside(self.times, 1) * c0 * (1 - np.exp(-self.times * k1))
        self.C[:, 2] = np.heaviside(self.times, 1) * c0 * np.exp(-self.times * k2)
        self.C[:, 3] = np.heaviside(self.times, 1) * c0 * (1 - np.exp(-self.times * k2))

        return self.get_conc_matrix(C_out, self._connectivity)

#
# class ABC_Model(_Model):
#     """ABC kinetic model, first order."""
#
#     n = 3
#     name = 'A→B→C (1st order)'
#
#     def __init__(self, times=None):
#         super(ABC_Model, self).__init__(times)
#
#         self.species_names = np.array(list('ABC'), dtype=np.str)
#
#         self.description = "Simple A->B->C model of 1st order. d[A]/dt = -k1[A] - k2[A]^2, [A]_0 = c_0"
#
#     def init_params(self):
#         self.params = Parameters()
#         self.params.add('c0', value=1, min=0, max=np.inf, vary=False)
#         self.params.add('k1', value=1, min=0, max=np.inf)
#         self.params.add('k2', value=0.5, min=0, max=np.inf)
#
#     def calc_C(self, params=None, C_out=None):
#         super(ABC_Model, self).calc_C(params)
#
#         c0, k1, k2 = self.params['c0'].value, self.params['k1'].value, self.params['k2'].value
#
#         self.C[:, 0] = np.heaviside(self.times, 1) * c0 * np.exp(-self.times * k1)
#         self.C[:, 1] = self.cB(self.times, c0, k1, k2)
#         self.C[:, 2] = np.heaviside(self.times, 1) * (c0 - self.C[:, 0] - self.C[:, 1])
#
#         return self.get_conc_matrix(C_out, self._connectivity)


class ABC_zero_Model(_Model):
    """ABC kinetic model, first order. then zero"""

    n = 3
    name = 'zero, A→B→C (1st, zero order)'
    _class = 'Nano'

    def __init__(self, times=None):
        super(ABC_zero_Model, self).__init__(times)

        # self.species_names = np.array(list('ABC'), dtype=np.str)

        self.description = "Simple A->B->C model of 1st order. d[A]/dt = -k1[A] - k2[A]^2, [A]_0 = c_0"

    def init_params(self):
        self.params = Parameters()
        self.params.add('c0', value=1, min=0, max=np.inf, vary=False)
        self.params.add('t0', value=11, min=0, max=np.inf)
        self.params.add('k1', value=0.03, min=0, max=np.inf)
        self.params.add('k2', value=0.4, min=0, max=np.inf)
        self.params.add('k3', value=0.0003, min=0, max=np.inf)
        self.params.add('k4', value=0.005, min=0, max=np.inf)



    def calc_C(self, params=None, C_out=None):
        super(ABC_zero_Model, self).calc_C(params)

        c0, t0, k1, k2, k3, k4 = [par[1].value for par in self.params.items()]

        def dc_dt(c, t):
            cA, cB, cC = c

            dA_dt = -k1 * cA / (k2 + cA)
            dB_dt = k1 * cA / (k2 + cA) - k3 * cB / (k4 + cB)
            dC_dt = k3 * cB / (k4 + cB)

            ret = np.asarray([dA_dt, dB_dt, dC_dt])

            return ret if t >= t0 else np.zeros_like(ret)

        x0 = np.linspace(0, self.times[0], num=100)
        _init_x = odeint(dc_dt, [c0, 0, 0], x0)[-1, :]  # take the row in the result matrix
        result = odeint(dc_dt, _init_x, self.times)

        result *= (result >= 0)

        self.C = result

        # self.C[:, 0] = np.heaviside(self.times, 1) * np.exp(-self.times * k1)
        # self.C[:, 1] = np.heaviside(self.times, 1) * self.cB(self.times, c0, k1, k2)
        # self.C[:, 2] = np.heaviside(self.times, 1) * (c0 - self.C[:, 0] - self.C[:, 1])

        return self.get_conc_matrix(C_out, self._connectivity)

# class ABCD_Model(_Model):
#     """ABCD kinetic model, first order."""
#
#     n = 4
#     name = 'A→B→C→D (1st order)'
#
#     def __init__(self, times=None, visible=None):
#         super(ABCD_Model, self).__init__(times, visible)
#
#         self.species_names = np.array(list('ABCD'), dtype=np.str)
#
#         # self.description = "Simple A->B->C model of 1st order. d[A]/dt = -k1[A] - k2[A]^2, [A]_0 = c_0"
#
#     def init_params(self):
#         self.params = Parameters()
#         self.params.add('c0', value=1, min=0, max=np.inf, vary=False)
#         self.params.add('k1', value=1, min=0, max=np.inf)
#         self.params.add('k2', value=0.5, min=0, max=np.inf)
#         self.params.add('k3', value=0.2, min=0, max=np.inf)
#
#     def calc_C(self, params=None, C_out=None):
#         super(ABCD_Model, self).calc_C(params)
#
#         c0, k1, k2, k3 = self.params['c0'].value, self.params['k1'].value, self.params['k2'].value, self.params[
#             'k3'].value
#
#         self.C[:, 0] = np.heaviside(self.times, 1) * c0 * np.exp(-self.times * k1)
#         self.C[:, 1] = self.cB(self.times, c0, k1, k2)
#
#         def dC_dt(cC, t):
#             cB = self.cB(t, c0, k1, k2)
#             return k2 * cB - k3 * cC  # d[C]/dt = k2[B] - k3[C]
#
#         # initial condition, cC(t=0) = 0
#
#         # initial conditiona, cC(t=0) = 0, cD(t=0) = 0
#         x0 = np.linspace(0, self.times[0], num=100)
#         _init_x = odeint(dC_dt, 0, x0)[-1, :]  # take the row in the result matrix
#         result = odeint(dC_dt, _init_x, self.times)
#
#         self.C[:, 2] = np.heaviside(self.times, 1) * result.flatten()
#         self.C[:, 3] = np.heaviside(self.times, 1) * (
#                 c0 - self.C[:, 0] - self.C[:, 1] - self.C[:, 2])
#
#         return self.get_conc_matrix(C_out, self._connectivity)


#
# class CP_Model(_Model):
#     """ABCD kinetic model, first order."""
#
#     n = 4
#     name = 'CP fitting model ABCZ'
#
#     def __init__(self, times=None, visible=None):
#         super(CP_Model, self).__init__(times, visible)
#
#         self.species_names = np.array(list('ABCZ'), dtype=np.str)
#
#     def init_params(self):
#         self.params = Parameters()
#         self.params.add('c0', value=1, min=0, max=np.inf, vary=False)
#         self.params.add('kTT', value=1, min=0, max=np.inf)  # BP 2nd order decay
#         self.params.add('kBP', value=1, min=0, max=np.inf)  # BP 1st order decay
#         self.params.add('k1', value=1, min=0, max=np.inf)  # kq[Q]
#         self.params.add('kT1T0', value=0.5, min=0, max=np.inf)  # k(T1->T0)
#         self.params.add('qT1', value=0.5, min=0, max=1)  # kb, ka + kb = k1
#         self.params.add('kd', value=0.3, min=0, max=np.inf)  # kd
#         # ka = pT1 * k1
#         # kb = (1 - pT1) * k1
#
#     def calc_C(self, params=None):
#         super(CP_Model, self).calc_C(params)
#
#         def cA(t, c0, k, kTT):
#             return np.heaviside(t, 1) * c0 * k / (
#                     (c0 * kTT + k) * np.exp(k * t) - c0 * kTT)  # A
#
#         kTT = self.params['kTT'].value
#         kBP = self.params['kBP'].value
#         c0, k1, kT1T0, qT1, kd = self.params['c0'].value, self.params['k1'].value, self.params['kT1T0'].value, \
#                                  self.params[
#                                      'qT1'].value, self.params['kd'].value
#
#         self.C_full[:, 0] = cA(self.times, c0, k1 + kBP, kTT)
#         self.C_full[:, 1] = self.cB(self.times, c0, qT1 * k1, kT1T0)
#
#         def dC_dt(cC, t):
#             cB = self.cB(t, c0, qT1 * k1, kT1T0)
#             return (1 - qT1) * k1 * cA(t, c0, k1 + kBP, kTT) + kT1T0 * cB - kd * cC  # d[C]/dt = k3[A] + k2[B] - k4[C]
#
#         # initial condition, cC(t=0) = 0
#         result = odeint(dC_dt, 0, self.times)
#         self.C_full[:, 2] = np.heaviside(self.times, 1) * result.flatten()
#
#         self.C_full[:, 3] = np.ones(self.times.shape[0])
#
#         return self.get_conc_matrix()

#
# class ABCDE_Model(_Model):
#     """ABCDE kinetic model, first order."""
#
#     n = 5
#     name = 'A→B→C→D→E (1st order)'
#
#     def __init__(self, times=None, visible=None):
#         super(ABCDE_Model, self).__init__(times, visible)
#
#         self.species_names = np.array(list('ABCDE'), dtype=np.str)
#
#     def init_params(self):
#         self.params = Parameters()
#         self.params.add('c0', value=1, min=0, max=np.inf, vary=False)
#         self.params.add('k1', value=1, min=0, max=np.inf)
#         self.params.add('k2', value=0.5, min=0, max=np.inf)
#         self.params.add('k3', value=0.4, min=0, max=np.inf)
#         self.params.add('k4', value=0.3, min=0, max=np.inf)
#
#     def calc_C(self, params=None, C_out=None):
#         super(ABCDE_Model, self).calc_C(params)
#
#         c0, k1, k2, k3, k4 = self.params['c0'].value, self.params['k1'].value, self.params['k2'].value, \
#                              self.params['k3'].value, self.params['k4'].value
#
#         self.C[:, 0] = np.heaviside(self.times, 1) * c0 * np.exp(-self.times * k1)
#         self.C[:, 1] = self.cB(self.times, c0, k1, k2)
#
#         def solve(conc, t):
#             cC, cD = conc
#             cB = self.cB(t, c0, k1, k2)
#             dC_dt = k2 * cB - k3 * cC  # d[C]/dt = k2[B] - k3[C]
#             dD_dt = k3 * cC - k4 * cD  # d[D]/dt = k3[C] - k4[D]
#             return [dC_dt, dD_dt]
#
#         # initial conditiona, cC(t=0) = 0, cD(t=0) = 0
#         x0 = np.linspace(0, self.times[0], num=100)
#         _init_x = odeint(solve, [0, 0], x0)[-1, :]  # take the row in the result matrix
#         result = odeint(solve, _init_x, self.times)
#
#         self.C[:, 2] = np.heaviside(self.times, 1) * result[:, 0]
#         self.C[:, 3] = np.heaviside(self.times, 1) * result[:, 1]
#
#         self.C[:, 4] = np.heaviside(self.times, 1) * (
#                 c0 - self.C[:, 0] - self.C[:, 1] - self.C[:, 2] - self.C[:, 3])
#
#         return self.get_conc_matrix(C_out, self._connectivity)


class Delayed_Fl(_Model):
    """Delayed fluorescence"""

    n = 4
    name = 'Delayed_Fl'
    _class = 'Nano'

    def __init__(self, times=None, visible=None):
        super(Delayed_Fl, self).__init__(times, visible)

        self.species_names = np.array(list('ABCZ'), dtype=np.str)

    def init_params(self):
        self.params = Parameters()
        self.params.add('c0', value=1, min=0, max=np.inf, vary=False)
        self.params.add('k0', value=1, min=0, max=np.inf)  # BP decay without quencher
        self.params.add('k1', value=1, min=0, max=np.inf)  # kq[CP]
        self.params.add('kTTbp', value=1, min=0, max=np.inf)  # TT annihilation of BP decay rate constant
        self.params.add('kqTT', value=0.5, min=0,
                        max=np.inf)  # rate constant of 3BP and 3DR annihilation (mixed TT annihilation)
        self.params.add('kISC', value=0.4, min=0, max=np.inf)  # shifting the maximum of the emission maxima
        self.params.add('kd', value=0.3, min=0, max=np.inf)  # 1st order decay rate constant of DR

    def calc_C(self, params=None, C_out=None):
        super(Delayed_Fl, self).calc_C(params)

        c0, k0, k1, kTTbp, kqTT, kISC, kd = self.params['c0'].value, self.params['k0'].value, self.params['k1'].value, \
                                            self.params['kTTbp'].value, self.params['kqTT'].value, self.params[
                                                'kISC'].value, self.params['kd'].value

        def solve(conc, t):
            cA, cB, cC = conc

            dcA_dt = -(k0 + k1) * cA - kTTbp * cA * cA - kqTT * cA * cB
            dcB_dt = + k1 * cA - kd * cB - kqTT * cA * cB + kISC * cC
            dcC_dt = + kqTT * cA * cB - kISC * cC

            return [dcA_dt, dcB_dt, dcC_dt]

        # initial conditiona, cC(t=0) = 0, cD(t=0) = 0
        x0 = np.linspace(0, self.times[0], num=100)
        _init = odeint(solve, [c0, 0, 0], x0)[-1, :]  # take the last 3 points in the result matrix
        result = odeint(solve, _init, self.times)  # evolve concentrations for time

        self.C[:, 0] = np.heaviside(self.times, 1) * result[:, 0]
        self.C[:, 1] = np.heaviside(self.times, 1) * result[:, 1]
        self.C[:, 2] = np.heaviside(self.times, 1) * result[:, 2]
        self.C[:, 3] = np.ones(self.times.shape[0])
        #
        # self.C_full[:, 4] = np.heaviside(self.times, 1) * (
        #         c0 - self.C_full[:, 0] - self.C_full[:, 1] - self.C_full[:, 2] - self.C_full[:, 3])

        return self.get_conc_matrix(C_out, self._connectivity)


class ABC_DEF_Model(_Model):
    """Two independent ABC kinetics, first order."""

    n = 6
    name = 'A→B→C, D→E→F (1st order)'
    _class = 'Nano'

    def __init__(self, times=None, visible=None):
        super(ABC_DEF_Model, self).__init__(times, visible)

        self.species_names = np.array(list('ABCDEF'), dtype=np.str)

        # self.description = "Simple A->B->C model of 1st order. d[A]/dt = -k1[A] - k2[A]^2, [A]_0 = c_0"

    def init_params(self):
        self.params = Parameters()
        self.params.add('c01', value=1, min=0, max=np.inf, vary=False)
        self.params.add('c02', value=1, min=0, max=np.inf, vary=False)
        self.params.add('k1', value=1, min=0, max=np.inf)
        self.params.add('k2', value=0.5, min=0, max=np.inf)
        self.params.add('k3', value=1.1, min=0, max=np.inf)
        self.params.add('k4', value=0.6, min=0, max=np.inf)

    def calc_C(self, params=None, C_out=None):
        super(ABC_DEF_Model, self).calc_C(params)

        c01, c02, k1, k2, k3, k4 = self.params['c01'].value, self.params['c02'].value, self.params['k1'].value, \
                                   self.params['k2'].value, self.params['k3'].value, self.params['k4'].value

        self.C[:, 0] = np.heaviside(self.times, 1) * c01 * np.exp(-self.times * k1)
        self.C[:, 1] = self.cB(self.times, c01, k1, k2)
        self.C[:, 2] = np.heaviside(self.times, 1) * (c01 - self.C[:, 0] - self.C[:, 1])

        self.C[:, 3] = np.heaviside(self.times, 1) * c02 * np.exp(-self.times * k3)
        self.C[:, 4] = self.cB(self.times, c02, k3, k4)
        self.C[:, 5] = np.heaviside(self.times, 1) * (c02 - self.C[:, 3] - self.C[:, 4])

        return self.get_conc_matrix(C_out, self._connectivity)


class Photosens_Model_Aug(_Model):
    n = 2
    name = 'Photosensitizaton augmented A→B→C (1st order)'
    _class = 'Nano'

    def __init__(self, times=None, aug_matrix=None):
        super(Photosens_Model_Aug, self).__init__(times)

        self.species_names = np.array(list('ABC'), dtype=np.str)
        self.aug_matrix = aug_matrix

        self.description = "Simple A->B->C model of 1st order. d[A]/dt = -k1[A] - k2[A]^2, [A]_0 = c_0"

    def init_params(self):
        self.params = Parameters()
        self.params.add('c0', value=1, min=0, max=np.inf, vary=False)
        self.params.add('k0', value=0.2, min=0, max=np.inf)
        self.params.add('k2', value=1, min=0, max=np.inf)
        self.params.add('kq', value=0.5, min=0, max=np.inf)
        # self.params.add('k3', value=1.5, min=0, max=np.inf)
        self.params.add('kd', value=0.5, min=0, max=np.inf)

    def calc_C(self, params=None, C_out=None):
        super(Photosens_Model_Aug, self).calc_C(params)

        if self.aug_matrix is None:
            raise ValueError("Augmented matrix must not be None in augmented model.")

        def cB(t, c0, k0, k1, kd):
            if np.abs(k1 - kd) < self._err:
                return np.heaviside(t, 1) * c0 * k1 * (np.exp(k0 * t) - 1) * np.exp(-(k1 + k0) * t) / k0
            else:
                return np.heaviside(t, 1) * (k1 * c0 / (kd - k1 - k0)) * (np.exp(-(k1 + k0) * t) - np.exp(-kd * t))

        c0, k0, kq, kd = self.params['c0'].value, self.params['k0'].value, self.params['kq'].value, self.params[
            'kd'].value
        k2 = self.params['k2'].value

        rows = self.aug_matrix.r
        k1s = np.zeros(rows)
        #
        # k1s[0] = 0
        # k1s[1] = self.params['k1'].value
        # k1s[2] = self.params['k2'].value
        # k1s[3] = self.params['k3'].value

        k1s[0] = 0 * kq
        k1s[1] = 8.293E-06 * kq
        k1s[2] = 2.073E-05 * kq
        k1s[3] = 4.147E-05 * kq
        k1s[4] = 8.293E-05 * kq
        k1s[5] = 1.244E-04 * kq
        k1s[6] = 1.037E-03 * kq

        i0 = 0  # last index

        # for each matrix in first column
        for i, mat in enumerate(self.aug_matrix._matrices[:, 0]):
            t = mat.times
            i1 = i0 + mat.times.shape[0]

            # self.C[i0:i1, 0] = np.heaviside(t, 1) * c0 * np.exp(-t * (k0 + k1s[i]))

            self.C[i0:i1, 0] = np.heaviside(t, 1) * c0 * (k0 + k1s[i]) / (
                    (c0 * k2 + (k0 + k1s[i])) * np.exp((k0 + k1s[i]) * t) - c0 * k2)  # A
            self.C[i0:i1, 1] = cB(t, c0, k0, k1s[i], kd)

            i0 = i1

        return self.get_conc_matrix(C_out, self._connectivity)


class ABC_NR(_Model):
    """ABCD kinetic model, first order."""

    n = 3
    name = 'A→B, A→C→B (1st order)'
    _class = 'Nano'

    def __init__(self, times=None, visible=None):
        super(ABC_NR, self).__init__(times, visible)

        self.species_names = np.array(list('ABC'), dtype=np.str)

        # self.description = "Simple A->B->C model of 1st order. d[A]/dt = -k1[A] - k2[A]^2, [A]_0 = c_0"

    def init_params(self):
        self.params = Parameters()
        self.params.add('c0', value=1, min=0, max=np.inf, vary=False)
        self.params.add('k_A', value=0.02, min=0, max=np.inf)
        self.params.add('p_B', value=0.5, min=0, max=1)
        self.params.add('k_CB', value=0.2, min=0, max=np.inf)

    def calc_C(self, params=None, C_out=None):
        super(ABC_NR, self).calc_C(params)

        c0, k_A, p_B, k_CB = [par[1].value for par in self.params.items()]

        K = np.asarray([[-k_A, 0, 0],
                        [p_B * k_A, 0, k_CB],
                        [(1 - p_B) * k_A, 0, -k_CB]])

        def dc_dt(c, t):
            return np.dot(K, c)

        # initial conditiona, cC(t=0) = 0, cD(t=0) = 0
        x0 = np.linspace(0, self.times[0], num=100)
        _init_x = odeint(dc_dt, [c0, 0, 0], x0)[-1, :]  # take the row in the result matrix
        result = odeint(dc_dt, _init_x, self.times)

        self.C[:, 0] = np.heaviside(self.times, 1) * result[:, 0]
        self.C[:, 1] = np.heaviside(self.times, 1) * result[:, 1]
        self.C[:, 2] = np.heaviside(self.times, 1) * result[:, 2]

        return self.get_conc_matrix(C_out, self._connectivity)


class Bridge_Splitting(_Model):
    n = 2
    name = 'Bridge Splitting: D+2L->2M'
    _class = 'Equilibrium'

    def __init__(self, times=None, ST=None, wavelengths=None):
        super(Bridge_Splitting, self).__init__(times)

        self.species_names = np.array(list('DMXXXXXXXX'), dtype=np.str)
        self.wavelengths = wavelengths
        self.ST = ST

        self.description = "asddddd"

    def init_model_params(self):
        params = Parameters()
        params.add('K', value=2, min=0, max=np.inf, vary=True)  # equilibrium constant
        params.add('c0', value=1e-5, min=0, max=np.inf, vary=False)  # intial dimer concentration

        return params

    def calc_C(self, params=None, C_out=None):
        super(Bridge_Splitting, self).calc_C(params)

        # if self.ST is None:
        #     raise ValueError("Spectra matrix must not be none.")

        K, c0 = [par[1].value for par in self.params.items()]

        # self.times in this case are ligand concentrations

        cL = self.times
        # alpha = cL * (-K * cL + np.sqrt(K * (K * cL * cL + 16 * c0))) / (8 * c0)

        alpha = np.zeros(self.times.shape[0])
        for i, cLi in enumerate(cL):
            polyi = [4 * K * c0 * c0, -4 * K * c0 * c0 - 4 * K * c0 * cLi + 4 * c0, 4 * K * c0 * cLi + K * cLi * cLi,
                     -K * cLi * cLi]
            roots = np.roots(polyi)  # find roots of this polynomial

            for root in roots:
                if (1 >= root >= alpha[i - 1]) if i > 0 else (1 >= root >= 0):
                    if not np.iscomplex(root):
                        alpha[i] = root
                        break

            # condition = (roots >= 0) if i < 1 else (roots >= alpha[i-1])
            # alphai = roots[condition][0]  # take first positive or zero root
            # alpha[i] = alphai

        self.C[:, 0] = np.heaviside(cL, 1) * (1 - alpha) * c0
        self.C[:, 1] = np.heaviside(cL, 1) * 2 * c0 * alpha

        return self.get_conc_matrix(C_out, self._connectivity)


class Bridge_Splitting_Simple(_Model):
    n = 2
    name = 'Bridge Splitting: D+L->M'
    _class = 'Equilibrium'

    def __init__(self, times=None, ST=None, wavelengths=None):
        super(Bridge_Splitting_Simple, self).__init__(times)

        self.species_names = np.array(list('DMXXXXXXXX'), dtype=np.str)
        self.wavelengths = wavelengths
        self.ST = ST

        self.description = "asddddd"

    def init_model_params(self):
        params = Parameters()
        params.add('K', value=2, min=0, max=np.inf, vary=True)  # equilibrium constant
        params.add('c0', value=1e-5, min=0, max=np.inf, vary=False)  # intial dimer concentration

        return params

    def calc_C(self, params=None, C_out=None):
        super(Bridge_Splitting_Simple, self).calc_C(params)

        K, c0 = [par[1].value for par in self.params.items()]

        cL = self.times
        alpha = K * cL / (1 + K * cL)

        self.C[:, 0] = np.heaviside(cL, 1) * (1 - alpha) * c0
        self.C[:, 1] = np.heaviside(cL, 1) * c0 * alpha

        return self.get_conc_matrix(C_out, self._connectivity)


class Dimerization_Equilibrium(_Model):
    n = 2
    name = 'Dimerization: D<->2M dilution experiment'
    _class = 'Equilibrium'
    normalized = False

    def __init__(self, times=None, ST=None, wavelengths=None):
        super(Dimerization_Equilibrium, self).__init__(times)

        self.species_names = np.array(['M', 'D'] + list('XXXXXXX'), dtype=np.str)
        self.wavelengths = wavelengths
        self.ST = ST

        self.description = "asddddd"

    def init_model_params(self):
        params = Parameters()
        # params.add('c0', value=1e-5, min=0, max=np.inf, vary=False)  # initial concentration
        params.add('K', value=1e-5, min=0, max=np.inf, vary=True)

        return params

    def calc_C(self, params=None, C_out=None):
        super(Dimerization_Equilibrium, self).calc_C(params)

        K, = [par[1].value for par in self.params.items()]

        G0 = self.times     # total initial concentration

        G = - K / 4 + np.sqrt(K * K / 16 + K * G0 / 2)
        D = G * G / K

        if self.normalized:
            G /= G0
            D /= G0

        self.C[:, 0] = G
        self.C[:, 1] = D

        return self.get_conc_matrix(C_out, self._connectivity)


class HG_Equilibrium(_Model):
    n = 2
    name = '1:1 Equilibrium: G+H->HG'
    _class = 'Equilibrium'


    def __init__(self, times=None, ST=None, wavelengths=None):
        super(HG_Equilibrium, self).__init__(times)

        self.species_names = np.array(['G', 'HG'] + list('XXXXXXX'), dtype=np.str)
        self.wavelengths = wavelengths
        self.ST = ST

        self.description = "asddddd"

    def init_model_params(self):
        params = Parameters()
        # params.add('H0', value=2, min=0, max=np.inf, vary=False)
        params.add('G0', value=1e-5, min=0, max=np.inf, vary=False)
        params.add('K', value=1e5, min=0, max=np.inf, vary=True)

        return params

    def calc_C(self, params=None, C_out=None):
        super(HG_Equilibrium, self).calc_C(params)

        G0, K = [par[1].value for par in self.params.items()]

        H0 = self.times * 1e-5  # factor used to convert the equivalents to initial concentration of the host
        HG = 0.5 * (G0 + H0 + 1/K) - np.sqrt(0.25 * (G0 + H0 + 1/K) ** 2 - H0 * G0)
        G = G0 - HG

        self.C[:, 0] = G
        self.C[:, 1] = HG

        return self.get_conc_matrix(C_out, self._connectivity)


class H2G_Equilibrium(_Model):
    n = 3
    name = '2:1 Equilibrium: G+H->HG, HG+H->H2G'
    _class = 'Equilibrium'

    def __init__(self, times=None, ST=None, wavelengths=None):
        super(H2G_Equilibrium, self).__init__(times)

        self.species_names = np.array(['G', 'HG', 'H2G'] + list('XXXXXXX'), dtype=np.str)
        self.wavelengths = wavelengths
        self.ST = ST

        self.description = "asddddd"

    def init_model_params(self):
        params = Parameters()
        # params.add('H0', value=2, min=0, max=np.inf, vary=False)
        params.add('G0', value=1e-5, min=0, max=np.inf, vary=False)
        params.add('K1', value=1e5, min=0, max=np.inf, vary=True)
        params.add('K2', value=1e5, min=0, max=np.inf, vary=True)

        return params

    def calc_C(self, params=None, C_out=None):
        super(H2G_Equilibrium, self).calc_C(params)

        G0, K1, K2 = [par[1].value for par in self.params.items()]

        H0 = self.times * 1e-5 # factor used to convert the equivalents to initial concentration of the host

        H, G, HG, H2G = self.simulate(K1, K2, G0, H0)

        self.C[:, 0] = G
        self.C[:, 1] = HG
        self.C[:, 2] = H2G

        return self.get_conc_matrix(C_out, self._connectivity)

    @staticmethod
    def polys(K1, K2, G0, H0):
        return [K1 * K2, K1 * (2 * K2 * G0 - K2 * H0 + 1), K1 * (G0 - H0) + 1, -H0]

    @staticmethod
    def simulate(K1, K2, G0, H0):
        """H0 is array"""
        H = np.empty_like(H0)

        for i in range(H0.shape[0]):
            roots = np.roots(H2G_Equilibrium.polys(K1, K2, G0, H0[i]))  # find roots of this polynomial

            real_roots = roots[~np.iscomplex(roots)].astype(np.float64)
            real_roots = real_roots[real_roots >= 0]
            H[i] = real_roots[0]

        G = G0 / (1 + K1 * H + K1 * K2 * H ** 2)
        HG = K1 * H * G  # mf_hg[i]  = (g0*h*k1)/(h0*(1 + (k1*h) + (k2*k1*h*h)))
        H2G = K1 * K2 * G * H ** 2  # (2*g0*k2*k1*h*h)/(h0*(1 + (k1*h) + (k2*k1*h*h)))

        return H, G, HG, H2G



class HDHG_Equilibrium(_Model):
    n = 3
    name = 'Equilibriums: D<->2G, G+H<->HG'
    _class = 'Equilibrium'

    def __init__(self, times=None, ST=None, wavelengths=None):
        super(HDHG_Equilibrium, self).__init__(times)

        self.species_names = np.array(['D', 'G', 'HG'] + list('XXXXXXX'), dtype=np.str)
        self.wavelengths = wavelengths
        self.ST = ST

        self.description = "asddddd"

    def init_model_params(self):
        params = Parameters()
        # params.add('H0', value=2, min=0, max=np.inf, vary=False)
        params.add('G0', value=1e-5, min=0, max=np.inf, vary=False)
        params.add('K1', value=1e5, min=0, max=np.inf, vary=True)  # dimerization
        params.add('K2', value=1e5, min=0, max=np.inf, vary=True)  # host guest eq.

        return params

    def calc_C(self, params=None, C_out=None):
        super(HDHG_Equilibrium, self).calc_C(params)

        G0, K1, K2 = [par[1].value for par in self.params.items()]

        H0 = self.times * 1e-5   # factor used to convert the equivalents to initial concentration of the host

        G, HG, D = self.simulate(K1, K2, G0, H0)

        self.C[:, 0] = D
        self.C[:, 1] = G
        self.C[:, 2] = HG

        return self.get_conc_matrix(C_out, self._connectivity)

    @staticmethod
    def polys(K1, K2, G0, H0):
        return [2 * K2 / K1, 2/K1 + K2, H0*K2 + 1 - G0*K2, -G0]

    @staticmethod
    def simulate(K1, K2, G0, H0):
        """H0 is array"""
        G = np.empty_like(H0)

        for i in range(H0.shape[0]):
            roots = np.roots(HDHG_Equilibrium.polys(K1, K2, G0, H0[i]))  # find roots of this polynomial

            real_roots = roots[~np.iscomplex(roots)].astype(np.float64)
            real_roots = real_roots[real_roots >= 0]
            G[i] = real_roots[0]

        HG = K2 * H0 * G / (1 + K2 * G)
        D = G * G / K1

        return G, HG, D


class Half_Bilirubin_1st_Model(_Model):
    n = 4
    name = '1st Half Bilirubin Photokinetics'
    _class = 'Steady state photokinetics'

    def __init__(self, times=None, ST=None, wavelengths=None):
        super(Half_Bilirubin_1st_Model, self).__init__(times)

        self.species_names = np.array(list('ZEHUXXXXX'), dtype=np.str)
        self.wavelengths = wavelengths
        self.ST = ST
        self.I_source = None

        self.description = "Simple A->B->C model of 1st order. d[A]/dt = -k1[A] - k2[A]^2, [A]_0 = c_0"

    def init_model_params(self):
        params = Parameters()
        params.add('q0', value=9.103e-10, min=0, max=np.inf, vary=False)  # molar photon flux in mol s-1
        params.add('V', value=2.5e-3, min=0, max=np.inf, vary=False)  # volume in mL
        params.add('w_irr', value=370, min=0, max=np.inf, vary=False)  # irradiating wavelength
        params.add('c0', value=2.90881898e-05, min=0, max=np.inf,
                        vary=False)  # total starting concentration of isomers, cZ+cE
        params.add('xZ', value=1, min=0, max=1,
                        vary=False)  # amount of Z in the mixture, 1: only Z, 0:, only E
        params.add('Phi_ZE', value=0.25, min=0, max=1)
        params.add('Phi_EZ', value=0.23, min=0, max=1)
        params.add('Phi_EHL', value=0.001, min=0, max=1)
        params.add('Phi_HLBl', value=0.0001, min=0, max=1, vary=False)

        return params

    @staticmethod
    def fk_factor(x, c=np.log(10), tol=1e-2):
        # exp(-xc) = 1 - xc + (xc)^2 / 2 - (xc)^3 / 6 ...
        # (1 - exp(-xc)) / x  ~  c - xc^2 / 2 for low x
        return np.where(x <= tol, c - x * c * c / 2 + x * x * c * c * c / 6, (1 - np.exp(-x * c)) / x)

    @staticmethod
    def simulate(q0, V, c0, eps, K, times, wavelengths, l=1, I_source=None, w_irr=None):
        """
        c0 is concentration vector at time, defined in times arary as first element (initial condition), eps is vector of molar abs. coefficients,
        I_source is spectrum of irradiaiton source if this was used,
        if not, w_irr as irradiaton wavelength must be specified, K is transfer matrix, l is length of a cuvette, default 1 cm
        times are times for which to simulate the kinetics
        """
        n = eps.shape[0]  # eps are epsilons - n x w matrix, where n is number of species and w is number of wavelengths
        # assert n == K.shape[0] == K.shape[1]
        c0 = np.asarray(c0)

        if I_source is None and w_irr is None:
            raise ValueError("Either specify I_source or irradiation wavelength w_irr!")

        # K_rank = np.linalg.matrix_rank(K)
        # assert K_rank >= n - 1  # rank of a transfer matrix cannot be less than n - 1
        # if K_rank < n:  # transfer matrix has not a full rank -> do not simulate the last equation!
        #     c_tot = c0.sum()  # conservation of mass holds, calculate the total initial concentration
        #     assert c_tot != 0  # sum of all initial concentration of components cannot be zero
        #     K = K[:-1]  # remove the last row of K matrix

        integrate = I_source is not None
        eps_w_irr = np.zeros(eps.shape[0])  # define epsilons only at irradiaton wavelength
        if not integrate:
            w_idx = find_nearest_idx(wavelengths, w_irr)
            for i in range(n):
                eps_w_irr[i] = eps[i][w_idx]
        else:
            I_source /= np.trapz(I_source, x=wavelengths)  # normalize irr source spectrum

        ln10 = np.log(10)

        def dc_dt(c, t):
            # if K has full rank, simulate all equations, but if not, calculate the concentration of
            # last component by conservation of mass, c_tot - c1 - c2 - ... - cn
            _c = c #if K_rank == n else np.append(c, c_tot - c.sum())

            c_eps = _c.reshape((-1, 1)) * eps if integrate else (_c * eps_w_irr).reshape((-1, 1))  # hadamard product

            # dot product, 1D array for integrated version, scalar number for one-wavelength version
            c_dot_eps = c_eps.sum(axis=0)

            # calculate part of absorbed light for each component
            # x_abs = c_eps * (1 - np.exp(-l * c_dot_eps * ln10)) / c_dot_eps
            x_abs = c_eps * Half_Bilirubin_1st_Model.fk_factor(c_dot_eps, c=l * ln10)

            # integrate if source spectrum is defined
            integrals = np.trapz(x_abs * I_source, x=wavelengths, axis=1) if integrate else x_abs

            return q0 * np.dot(K, integrals).flatten() / V  # final matrix multiplication

        # result = odeint(dc_dt, c0 if K_rank == n else c0[:-1], times)
        result = odeint(dc_dt, c0, times)

        # if K_rank < n:
        #     # calculate the time profile for last component and stack it to results
        #     result = np.hstack((result, c_tot - result.sum(axis=1, keepdims=True)))
        return result

    def calc_C(self, params=None, C_out=None):
        super(Half_Bilirubin_1st_Model, self).calc_C(params)

        if self.ST is None:
            raise ValueError("Spectra matrix must not be none.")

        q0, V, w_irr, c0, xZ, Phi_ZE, Phi_EZ, Phi_EHL, Phi_HLBl = [par[1].value for par in self.params.items()]

        K = np.asarray([[-Phi_ZE, Phi_EZ, 0],
                        [Phi_ZE, -Phi_EZ - Phi_EHL, 0],
                        [0, Phi_EHL, 0]])

        # K = np.asarray([[-Phi_ZE, Phi_EZ, 0, 0],
        #                 [Phi_ZE, -Phi_EZ - Phi_EHL, 0, 0],
        #                 [0, Phi_EHL, - Phi_HLBl, 0],
        #                 [0, 0, Phi_HLBl, 0]])

        n = K.shape[0]

        c0_vec = [c0*xZ, (1-xZ)*c0] + (n - 2) * [0]

        x0 = np.linspace(0, self.times[0], num=10)
        _init_x = self.simulate(q0, V, c0_vec, self.ST[:n, :], K, x0, self.wavelengths, l=1,
                               I_source=self.I_source, w_irr=w_irr)[-1, :]

        # _init_x = odeint(solve, [c0 * xZ, c0 * (1 - xZ), 0, 0], x0)[-1, :]  # take the row in the result matrix

        result = self.simulate(q0, V, _init_x, self.ST[:n, :], K, self.times, self.wavelengths, l=1,
                               I_source=self.I_source, w_irr=w_irr)

        self.C = np.heaviside(self.times, 1)[..., None] * result

        # if any(np.isnan(self.C)):
        #     self.C = np.nan_to_num(self.C)
        #     Console.show_message("WARNING: Concentration matrix contained nan values.")

        # self.C[:, 0] = np.heaviside(self.times, 1) * result[:, 0]
        # self.C[:, 1] = np.heaviside(self.times, 1) * result[:, 1]
        # self.C[:, 2] = np.heaviside(self.times, 1) * result[:, 2]
        # self.C[:, 3] = np.heaviside(self.times, 1) * result[:, 3]

        return self.get_conc_matrix(C_out, self._connectivity)


class SingletOxygenProduction(_Photokinetic_Model):
    n = 1
    name = 'Determination of singlet oxygen production QY relative'
    _class = 'Steady state photokinetics'

    def __init__(self, times=None, ST=None, wavelengths=None):
        super(SingletOxygenProduction, self).__init__(times)

        self.species_names = ['DPBF'] + list('XXXXXXXXX')
        self.wavelengths = wavelengths
        self.ST = ST

        # self.I_LED_norm = None

    def init_model_params(self):
        params = Parameters()
        params.add('J', value=2.842e-06, min=0, max=np.inf, vary=False)  # molar photon flux in mol s-1
        params.add('k_r', value=2.83e9, min=0, max=np.inf, vary=False)  # irradiating wavelength
        params.add('k_d', value=105263.15, min=0, max=np.inf, vary=False)  # decay rate constant of singlet ox. in MeOH
        params.add('Phi_Delta', value=0.0025, min=0, max=np.inf, vary=True)  # quantum yield of singlet ox. production
        params.add('c0', value=5.809e-05, min=0, max=np.inf, vary=True)  # total starting concentration of DPBF

        return params

    def calc_C(self, params=None, C_out=None):
        super(SingletOxygenProduction, self).calc_C(params)

        if self.ST is None:
            raise ValueError("Spectra matrix must not be none.")

        J, k_r, k_d, Phi_Delta, c0 = [par[1].value for par in self.params.items()]

        irr_source = self.get_LED_source()

        def dc_dt(c, t):
            tidx = find_nearest_idx(self.times, t)
            integral = np.trapz(irr_source * (1 - 10 ** (-self.D[tidx, :])), self.wavelengths)
            return -Phi_Delta * J * k_r * c * integral / (k_d + k_r * c)

        self.C[:, 0] = odeint(dc_dt, c0, self.times).squeeze()

        return self.get_conc_matrix(C_out, self._connectivity)


class SingletOxygenProductionAbs(_Photokinetic_Model):
    n = 1
    name = 'Determination of singlet oxygen production QY absolute'
    _class = 'Steady state photokinetics'

    def __init__(self, times=None, ST=None, wavelengths=None):
        super(SingletOxygenProductionAbs, self).__init__(times)

        self.species_names = ['DPBF'] + list('XXXXXXXXX')
        self.wavelengths = wavelengths
        self.ST = ST

    def init_model_params(self):
        params = Parameters()
        params.add('I', value=1.105e-3, min=0, max=np.inf, vary=False)  # Current in ampere on the photodide, without any cuvette
        params.add('V', value=0.00291, min=0, max=np.inf, vary=False)  # Volume of the solution
        params.add('k_r', value=2.83e9, min=0, max=np.inf, vary=False)  # reaction of DPBF with singlet oxygen
        params.add('k_d', value=105263.15, min=0, max=np.inf, vary=False)  # decay rate constant of singlet ox. in MeOH
        params.add('Phi_Delta', value=0.0025, min=0, max=np.inf, vary=False)  # quantum yield of singlet ox. production
        params.add('R', value=0.036, min=0, max=np.inf, vary=False)  # Cuvette reflectivity
        params.add('c0_DPBF', value=63e-6, min=0, max=np.inf, vary=True)  # Concentration of DPBF

        return params

    def calc_C(self, params=None, C_out=None):
        super(SingletOxygenProductionAbs, self).calc_C(params)

        if self.ST is None:
            raise ValueError("Spectra matrix must not be none.")

        I, V, k_r, k_d, Phi_Delta, R, c0_DPBF = [par[1].value for par in self.params.items()]

        irr_source = self.get_LED_source()
        q_rel = self.get_q_rel()

        # calculate incident spectral photon flux
        spectral_flux = I * np.trapz(q_rel * irr_source) * irr_source / V  # I * integral(q_rel * PDF) * PDF

        def dc_dt(c, t):
            tidx = find_nearest_idx(self.times, t)
            A = self.D[tidx, :]  # current absorbance
            T = 10**(-A)  # calculate transmittance

            effective_spectral_flux = spectral_flux * (1 - R) * (1 + R * T)

            integral = np.trapz(effective_spectral_flux * (1 - T), self.wavelengths)  # integrate

            return -Phi_Delta * k_r * c * integral / (k_d + k_r * c)

        # c0 = 63e-6 if self.C[0, 0] == 0 else self.C[0, 0]

        self.C[:, 0] = odeint(dc_dt, c0_DPBF, self.times).squeeze()

        return self.get_conc_matrix(C_out, self._connectivity)

class Z3_Photokinetics(_Photokinetic_Model):
    n = 4
    name = '3Z photokinetic and thermal target model'
    _class = 'Steady state photokinetics'

    def __init__(self, times=None, ST=None, wavelengths=None):
        super(Z3_Photokinetics, self).__init__(times)

        self.species_names = ['3Z', 'PDP', 'Hydroperoxide', 'Imine']  + list('ABCDEFGH')
        self.wavelengths = wavelengths
        self.ST = ST

    def init_model_params(self):
        params = Parameters()
        params.add('I', value=1.105e-3, min=0, max=np.inf, vary=True)  # Current in ampere on the photodide, without any cuvette
        params.add('V', value=0.0025, min=0, max=np.inf, vary=False)  # Volume of the solution
        params.add('k_r', value=1e9, min=0, max=np.inf, vary=False)  # reaction of 3Z with singlet oxygen
        params.add('k_d', value=105263.15, min=0, max=np.inf, vary=False)  # decay rate constant of singlet ox. in MeOH
        params.add('Phi_Delta', value=0.7, min=0, max=np.inf, vary=False)  # quantum yield of singlet ox. production from RB
        params.add('alpha', value=0.5, min=0, max=1, vary=False)  #  branching coefficient
        # params.add('eps_3Z', value=63e-6, min=0, max=np.inf, vary=False)  # Concentration of DPBF

        params.add('tau1', value=96, min=0, max=np.inf, vary=True)  # Concentration of DPBF
        params.add('tau2', value=300, min=0, max=np.inf, vary=True)  # Concentration of DPBF
        params.add('tau3', value=16373, min=0, max=np.inf, vary=True)  # Concentration of DPBF

        return params

    def calc_C(self, params=None, C_out=None):
        super(Z3_Photokinetics, self).calc_C(params)

        if self.ST is None:
            raise ValueError("Spectra matrix must not be none.")

        R = 0.036  # Cuvette reflectivity

        I, V, k_r, k_d, Phi_Delta, alpha, tau1, tau2, tau3 = [par[1].value for par in self.params.items()]

        irr_source = self.get_LED_source()
        q_rel = self.get_q_rel()

        # calculate incident spectral photon flux
        spectral_flux = I * np.trapz(q_rel * irr_source) * irr_source / V  # I * integral(q_rel * PDF) * PDF

        # K = np.asarray([[0, 0, 0, 0],
        #                 [0, -1/tau1, 0, 0],
        #                 [0, 1/tau1, -1/tau2, 0],
        #                 [0, 0, 1/tau2, -1/tau3]])

        # 3Z, PDP, OOH, imine
        K = np.asarray([[0, 0, 0, 0],
                        [0, -1/tau3, 0, 1/tau2],
                        [0, 0, -1/tau1, 0],
                        [0, 0, 1/tau1, -1/tau2]])

        def dc_dt(c, t):
            tidx = find_nearest_idx(self.times, t)
            A = self.D[tidx, :]  # current absorbance
            T = 10**(-A)  # calculate transmittance

            effective_spectral_flux = spectral_flux * (1 - R) * (1 + R * T)
            integral = np.trapz(effective_spectral_flux * (1 - T), self.wavelengths)  # integrate

            vec = np.zeros_like(c)
            decay = -Phi_Delta * k_r * c[0] * integral / (k_d + k_r * c[0])  # photochemical decay by singlet ox
            vec[0] = decay  # decay of 3Z
            vec[1] = -decay * alpha  # rise of PDP
            vec[2] = -decay * (1 - alpha)  # rise of hydroperoxide

            return K.dot(c) + vec

        c0_3Z = 6.45e-5

        self.C = odeint(dc_dt, np.asarray([c0_3Z, 0, 0, 0]), self.times).squeeze()

        return self.get_conc_matrix(C_out, self._connectivity)


# class Half_Bilirubin_Multiset(_Model):
#     n = 4
#     name = 'Half-Bilirubin Multiset Model'
#
#     def __init__(self, times=None, ST=None, wavelengths=None, aug_matrix=None):
#         super(Half_Bilirubin_Multiset, self).__init__(times)
#
#         self.species_names = np.array(list('ZEHD'), dtype=np.str)
#         self.wavelengths = wavelengths
#         self.ST = ST
#         self.interp_kind = 'linear'
#         self.wl_range = (340, 480)
#
#         # path = r"C:\Users\Dominik\Documents\MUNI\Organic Photochemistry\Projects\2019-Bilirubin project\UV-VIS\QY measurement\Photodiode\new setup"
#         path = r"C:\Users\dominik\Documents\Projects\Bilirubin\UV-Vis data"
#
#         fname = path + r'\em sources.txt'
#         data = np.loadtxt(fname, delimiter='\t', skiprows=1)
#
#         self.I_330 = data[:, 1] / np.trapz(data[:, 1], x=self.wavelengths)
#         self.I_375 = data[:, 2] / np.trapz(data[:, 2], x=self.wavelengths)
#         self.I_400 = data[:, 3] / np.trapz(data[:, 3], x=self.wavelengths)
#         self.I_450 = data[:, 4] / np.trapz(data[:, 4], x=self.wavelengths)
#         self.I_480 = data[:, 5] / np.trapz(data[:, 5], x=self.wavelengths)
#
#         data_led = np.loadtxt(path + r'\LED sources.txt', delimiter='\t', skiprows=1)
#
#         self.LED_355 = data_led[:, 1] / np.trapz(data_led[:, 1], x=self.wavelengths)
#         self.LED_375 = data_led[:, 2] / np.trapz(data_led[:, 2], x=self.wavelengths)
#         self.LED_405 = data_led[:, 3] / np.trapz(data_led[:, 3], x=self.wavelengths)
#         self.LED_420 = data_led[:, 4] / np.trapz(data_led[:, 4], x=self.wavelengths)
#         self.LED_450 = data_led[:, 5] / np.trapz(data_led[:, 5], x=self.wavelengths)
#         self.LED_470 = data_led[:, 6] / np.trapz(data_led[:, 6], x=self.wavelengths)
#         self.LED_490 = data_led[:, 7] / np.trapz(data_led[:, 7], x=self.wavelengths)
#
#         fname = path + r'\q rel cut.txt'
#         data = np.loadtxt(fname, delimiter='\t', skiprows=1)
#         self.Diode_q_rel = data[:, 1]
#
#         self._overlap330 = np.trapz(self.Diode_q_rel * self.I_330, x=self.wavelengths)
#         self._overlap375 = np.trapz(self.Diode_q_rel * self.I_375, x=self.wavelengths)
#         self._overlap400 = np.trapz(self.Diode_q_rel * self.I_400, x=self.wavelengths)
#         self._overlap450 = np.trapz(self.Diode_q_rel * self.I_450, x=self.wavelengths)
#         self._overlap480 = np.trapz(self.Diode_q_rel * self.I_480, x=self.wavelengths)
#
#         self.aug_matrix = aug_matrix
#
#         self.description = ""
#
#     def init_params(self):
#         self.params = Parameters()
#         # self.params.add('IZ', value=38.1e-6, min=0, max=np.inf, vary=False)  # molar photon flux in mol s-1
#         # self.params.add('IE', value=37.7e-6, min=0, max=np.inf, vary=False)  # molar photon flux in mol s-1
#         #
#         # self.params.add('V', value=3e-3, min=0, max=np.inf, vary=False)  # volume in L
#         # self.params.add('w_irr', value=400, min=0, max=np.inf, vary=False)  # irradiating wavelength
#
#         # self.params.add('c0Z330', value=4.74E-05, min=0, max=np.inf, vary=True)
#         # self.params.add('c0E330', value=3.72E-05, min=0, max=np.inf, vary=True)
#         #
#         # self.params.add('c0Z400', value=4.68E-05, min=0, max=np.inf, vary=True)
#         # self.params.add('c0E400', value=3.65E-05, min=0, max=np.inf, vary=True)
#         #
#         # self.params.add('c0Z480', value=4.78E-05, min=0, max=np.inf, vary=True)
#         # self.params.add('c0E480', value=3.66E-05, min=0, max=np.inf, vary=True)
#         #
#         # self.params.add('c0Z375', value=6.66E-05, min=0, max=np.inf, vary=True)
#         # self.params.add('c0Z450', value=6.57E-05, min=0, max=np.inf, vary=True)
#
#         # self.params.add('c0Z355LED', value=4.57E-05, min=0, max=np.inf, vary=True)
#         self.params.add('q0Z355LED', value=6.438e-08, min=0, max=np.inf, vary=True)
#
#         # self.params.add('c0Z375LED', value=5.13E-05, min=0, max=np.inf, vary=True)
#         self.params.add('q0Z375LED', value=9.845e-08, min=0, max=np.inf, vary=True)
#
#         # self.params.add('c0Z405LED', value=5.18E-05, min=0, max=np.inf, vary=True)
#         self.params.add('q0Z405LED', value=9.635e-08, min=0, max=np.inf, vary=True)
#
#         # self.params.add('c0Z450LED', value=5.12E-05, min=0, max=np.inf, vary=True)
#         self.params.add('q0Z450LED', value=4.273e-07, min=0, max=np.inf, vary=True)
#
#         # self.params.add('c0Z470LED', value=5.12E-05, min=0, max=np.inf, vary=True)
#         self.params.add('q0Z470LED', value=7.03e-07, min=0, max=np.inf, vary=True)
#
#         # self.params.add('c0Z490LED', value=4.28E-05, min=0, max=np.inf, vary=True)
#         self.params.add('q0Z490LED', value=4.184e-07, min=0, max=np.inf, vary=True)
#
#         # self.params.add('c0Z490355LED', value=4.915e-05, min=0, max=np.inf, vary=True)
#         # self.params.add('q0Z490355LED', value=4.903e-07, min=0, max=np.inf, vary=True)
#
#         # amount of Z in the mixture, 1: only Z, 0:, only E
#         self.params.add('xZ_Z', value=1, min=0, max=1, vary=False)
#         self.params.add('xZ_E', value=0.055, min=0, max=1, vary=False)
#
#         self.params.add('Phi_ZE', value=0.160581292, min=0, max=1)
#         self.params.add('Phi_ZE_1', value=0.16771559, min=0, max=1)
#         self.params.add('Phi_ZE_2', value=0.193611305, min=0, max=1)
#         self.params.add('Phi_ZE_3', value=0.193611305, min=0, max=1, vary=True)
#         self.params.add('Phi_ZE_4', value=0.193611305, min=0, max=1, vary=True)
#
#         self.params.add('Phi_EZ', value=0.368744317, min=0, max=1)
#         self.params.add('Phi_EZ_1', value=0.3, min=0, max=1)
#         self.params.add('Phi_EZ_2', value=0.3, min=0, max=1)
#         self.params.add('Phi_EZ_3', value=0.3, min=0, max=1, vary=True)
#         self.params.add('Phi_EZ_4', value=0.3, min=0, max=1, vary=True)
#
#         self.params.add('Phi_ZHL', value=0.004081171, min=0, max=1, vary=True)
#         self.params.add('Phi_ZHL_1', value=0.004, min=0, max=1)
#         self.params.add('Phi_ZHL_2', value=0.004, min=0, max=1)
#         self.params.add('Phi_ZHL_3', value=0.004, min=0, max=1, vary=True)
#         self.params.add('Phi_ZHL_4', value=0.004, min=0, max=1, vary=True)
#
#         self.params.add('Phi_HLZ', value=0.006594641, min=0, max=1, vary=True)
#         self.params.add('Phi_HLZ_1', value=0.004, min=0, max=1)
#         self.params.add('Phi_HLZ_2', value=0.004, min=0, max=1)
#         self.params.add('Phi_HLZ_3', value=0.004, min=0, max=1, vary=True)
#         self.params.add('Phi_HLZ_4', value=0.004, min=0, max=1, vary=True)
#
#
#         # HL decay QY
#         self.params.add('Phi_HL', value=0.003677229, min=0, max=1, vary=True)
#         self.params.add('Phi_HL_1', value=0.004, min=0, max=1)
#         self.params.add('Phi_HL_2', value=0.004,  min=0, max=1)
#         self.params.add('Phi_HL_3', value=0.004, min=0, max=1, vary=True)
#         self.params.add('Phi_HL_4', value=0.004, min=0, max=1, vary=True)
#
#
#     #
#     # @staticmethod
#     # def fk_factor(x, c=np.log(10), tol=1e-2):
#     #     # exp(-xc) = 1 - xc + (xc)^2 / 2 - (xc)^3 / 6 ...
#     #     # (1 - exp(-xc)) / x  ~  c - xc^2 / 2 for low x
#     #     return np.where(x <= tol, c - x * c * c / 2 + x * x * c * c * c / 6, (1 - np.exp(-x * c)) / x)
#
#     def plot_phis(self, y_scale='log'):
#         if self.params is None:
#             return
#
#         # c0Z330, c0E330, c0Z400, c0E400, c0Z480, c0E480, c0Z375, c0Z450, \
#         q0Z355LED, q0Z375LED, q0Z405LED, q0Z450LED, q0Z470LED, q0Z490LED, \
#         xZ_Z, xZ_E, \
#         Phi_ZE, Phi_ZE1, Phi_ZE2, Phi_ZE3, Phi_ZE4, \
#         Phi_EZ, Phi_EZ1, Phi_EZ2, Phi_EZ3, Phi_EZ4, \
#         Phi_ZHL, Phi_ZHL1, Phi_ZHL2, Phi_ZHL3, Phi_ZHL4, \
#         Phi_HLZ, Phi_HLZ1, Phi_HLZ2, Phi_HLZ3, Phi_HLZ4, \
#         Phi_HL, Phi_HL1, Phi_HL2, Phi_HL3, Phi_HL4 = [par[1].value for par in self.params.items()]
#
#         _Phi_ZE = self.Phi_interp([Phi_ZE, Phi_ZE1, Phi_ZE2, Phi_ZE3, Phi_ZE4], self.wl_range)
#         _Phi_EZ = self.Phi_interp([Phi_EZ, Phi_EZ1, Phi_EZ2, Phi_EZ3, Phi_EZ4], self.wl_range)
#         _Phi_ZHL = self.Phi_interp([Phi_ZHL, Phi_ZHL1, Phi_ZHL2, Phi_ZHL3, Phi_ZHL4], self.wl_range)
#         _Phi_HLZ = self.Phi_interp([Phi_HLZ, Phi_HLZ1, Phi_HLZ2, Phi_HLZ3, Phi_HLZ4], self.wl_range)
#         _Phi_HL = self.Phi_interp([Phi_HL, Phi_HL1, Phi_HL2, Phi_HL3, Phi_HL4], self.wl_range)
#
#         plt.plot(self.wavelengths, _Phi_ZE, label="$\Phi_{ZE}$")
#         plt.plot(self.wavelengths, _Phi_EZ, label="$\Phi_{EZ}$")
#         plt.plot(self.wavelengths, _Phi_ZHL, label="$\Phi_{ZHL}$")
#         plt.plot(self.wavelengths, _Phi_HLZ, label="$\Phi_{HLZ}$")
#         plt.plot(self.wavelengths, _Phi_HL, label="$\Phi_{HL-decay}$")
#
#         plt.yscale(y_scale)
#
#         plt.xlabel('Wavelength')
#         plt.ylabel('$\Phi$')
#         plt.legend()
#
#         plt.show()
#
#
#
#     @staticmethod
#     def Phi(phis, wavelengths, lambda_C=400):
#         assert isinstance(phis, (list, np.ndarray))
#         return sum(par * ((lambda_C - wavelengths) / 100) ** i for i, par in enumerate(phis))
#
#     def Phi_interp(self, phis, wl_range=(340, 480), crop_01=True):
#         n = len(phis)
#         assert n >= 3
#         x = np.linspace(wl_range[0], wl_range[1], n, endpoint=True)
#         f = interp1d(x, np.asarray(phis), kind=self.interp_kind, fill_value='extrapolate')
#
#         y = f(self.wavelengths)
#         if crop_01:
#             y *= (y > 0)
#             y = np.where(y > 1, 1, y)
#
#         return y
#
#
#     @staticmethod
#     def simulate(q0, c0, K, I_source, wavelengths=None, times=None, eps=None, V=0.003, l=1,  D=None):
#         """
#         c0 is concentration vector at time, defined in times arary as first element (initial condition), eps is vector of molar abs. coefficients,
#         I_source is spectrum of irradiaiton source if this was used,
#         if not, w_irr as irradiaton wavelength must be specified, K is transfer matrix, l is length of a cuvette, default 1 cm
#         times are times for which to simulate the kinetics
#         """
#         # n = eps.shape[0]  # eps are epsilons - n x w matrix, where n is number of species and w is number of wavelengths
#         # assert n == K.shape[0] == K.shape[1]
#         c0 = np.asarray(c0)
#
#
#         # get absorbances from real data
#         abs_at = interp2d(wavelengths, times, D, kind='linear', copy=True)
#
#         const = l * np.log(10)
#         tol = 1e-3
#
#         def dc_dt(c, t):
#             # c_eps = c[..., None] * eps if integrate else (c * eps_w_irr)[..., None]  # hadamard product
#             # c_dot_eps = c_eps.sum(axis=0)
#             # # x_abs = c_eps * Half_Bilirubin_Multiset.fk_factor(c_dot_eps, c=l * ln10) * (I_source if integrate else 1)
#             # x_abs = c_eps * fk_factor_numba(c_dot_eps, c=l * ln10) * (I_source if integrate else 1)
#             # # w x n x n   x   w x n x 1
#             # product = np.matmul(K, x_abs.T[..., None])  # w x n x 1
#             # return q0 / V * (np.trapz(product, x=wavelengths, axis=0) if integrate else product).squeeze()
#
#             c_eps = c[..., None] * eps  # hadamard product
#
#             # c_dot_eps = c_eps.sum(axis=0)
#             c_dot_eps = abs_at(wavelengths, t)
#
#             # x_abs = c_eps * Half_Bilirubin_Multiset.fk_factor(c_dot_eps, c=l * ln10) * (I_source if integrate else 1)
#             # x_abs = c_eps * fk_factor_numba(c_dot_eps, c=l * ln10) * I_source
#
#             x_abs = c_eps * np.where(c_dot_eps <= tol, const - c_dot_eps * const * const / 2,
#                                      (1 - np.exp(-c_dot_eps * const)) / c_dot_eps) * I_source
#
#             # w x n x n   x   w x n x 1
#             product = np.matmul(K, x_abs.T[..., None])  # w x n x 1
#
#             return q0 / V * np.trapz(product, x=wavelengths, axis=0).squeeze()
#
#         result = odeint(dc_dt, c0, times)
#
#         return result
#
#     def calc_C(self, params=None, C_out=None):
#         super(Half_Bilirubin_Multiset, self).calc_C(params)
#
#         if self.ST is None:
#             raise ValueError("Spectra matrix must not be none.")
#
#         # c0Z330, c0E330, c0Z400, c0E400, c0Z480, c0E480, xZ_Z, xZ_E, Phi_ZE, Phi_EZ, Phi_EHL, Phi_ZHL, Phi_HLE, Phi_ZE_1, Phi_EZ_1, Phi_EHL_1 = [par[1].value for par in self.params.items()]
#         # c0Z330, c0E330, c0Z400, c0E400, c0Z480, c0E480, c0Z375, c0Z450, \
#         #         # c0Z355LED, q0Z355LED, c0Z375LED, q0Z375LED, c0Z405LED, q0Z405LED, c0Z450LED, q0Z450LED, c0Z470LED, q0Z470LED, c0Z490LED, q0Z490LED, \
#         #         # xZ_Z, xZ_E, \
#         #         # Phi_ZE, Phi_ZE1, Phi_ZE2, Phi_ZE3, Phi_ZE4, \
#         #         # Phi_EZ, Phi_EZ1, Phi_EZ2, Phi_EZ3, Phi_EZ4, \
#         #         # Phi_ZHL, Phi_ZHL1, Phi_ZHL2, Phi_ZHL3, Phi_ZHL4, \
#         #         # Phi_HLZ, Phi_HLZ1, Phi_HLZ2, Phi_HLZ3, Phi_HLZ4, \
#         #         # Phi_HL, Phi_HL1, Phi_HL2, Phi_HL3, Phi_HL4 = [par[1].value for par in self.params.items()]
#
#         q0Z355LED, q0Z375LED, q0Z405LED, q0Z450LED, q0Z470LED, q0Z490LED, \
#         xZ_Z, xZ_E, \
#         Phi_ZE, Phi_ZE1, Phi_ZE2, Phi_ZE3, Phi_ZE4, \
#         Phi_EZ, Phi_EZ1, Phi_EZ2, Phi_EZ3, Phi_EZ4, \
#         Phi_ZHL, Phi_ZHL1, Phi_ZHL2, Phi_ZHL3, Phi_ZHL4, \
#         Phi_HLZ, Phi_HLZ1, Phi_HLZ2, Phi_HLZ3, Phi_HLZ4, \
#         Phi_HL, Phi_HL1, Phi_HL2, Phi_HL3, Phi_HL4 = [par[1].value for par in self.params.items()]
#
#         # _Phi_ZE = self.Phi([Phi_ZE, Phi_ZE1, Phi_ZE2, Phi_ZE3, Phi_ZE4], 400, self.wavelengths)
#         #         # _Phi_EZ = self.Phi([Phi_EZ, Phi_EZ1, Phi_EZ2, Phi_EZ3, Phi_EZ4], 400, self.wavelengths)
#         #         # _Phi_ZHL = self.Phi([Phi_ZHL, Phi_ZHL1, Phi_ZHL2, Phi_ZHL3, Phi_ZHL4], 400, self.wavelengths)
#         #         # _Phi_HLZ = self.Phi([Phi_HLZ, Phi_HLZ1, Phi_HLZ2, Phi_HLZ3, Phi_HLZ4], 400, self.wavelengths)
#         #         # _Phi_HL = self.Phi([Phi_HL, Phi_HL1, Phi_HL2, Phi_HL3, Phi_HL4], 400, self.wavelengths)
#
#         _Phi_ZE = self.Phi_interp([Phi_ZE, Phi_ZE1, Phi_ZE2, Phi_ZE3, Phi_ZE4], self.wl_range)
#         _Phi_EZ = self.Phi_interp([Phi_EZ, Phi_EZ1, Phi_EZ2, Phi_EZ3, Phi_EZ4], self.wl_range)
#         _Phi_ZHL = self.Phi_interp([Phi_ZHL, Phi_ZHL1, Phi_ZHL2, Phi_ZHL3, Phi_ZHL4], self.wl_range)
#         _Phi_HLZ = self.Phi_interp([Phi_HLZ, Phi_HLZ1, Phi_HLZ2, Phi_HLZ3, Phi_HLZ4], self.wl_range)
#         _Phi_HL = self.Phi_interp([Phi_HL, Phi_HL1, Phi_HL2, Phi_HL3, Phi_HL4], self.wl_range)
#
#
#         IZ330, IE330, IZ400, IE400, V = 16.7e-6, 17e-6, 38.1e-6, 37.7e-6, 3e-3
#         IZ375, IZ450 = 24.9e-6, 47.9e-6
#         IZ480, IE480 = 48e-6, 48e-6
#
#
#
#         # Phi_HLE = self.Phi([Phi_HLE], 400, self.wavelengths)
#
#         _0 = np.zeros(self.wavelengths.shape) if isinstance(_Phi_ZE, np.ndarray) else 0
#
#         # K = np.asarray([[-Phi_ZE - Phi_ZHL,   Phi_EZ,        Phi_HLZ],
#         #                 [Phi_ZE,              -Phi_EZ,            _0],
#         #                 [Phi_ZHL,             _0,          -Phi_HLZ]])
#
#         K = np.asarray([[-_Phi_ZE - _Phi_ZHL, _Phi_EZ, _Phi_HLZ, _0],
#                         [_Phi_ZE, -_Phi_EZ, _0, _0],
#                         [_Phi_ZHL, _0, -_Phi_HLZ - _Phi_HL, _0],
#                         [_0, _0, _Phi_HL, _0]])
#
#         # # # no photoproduct
#         # K_no_D = np.asarray([[-Phi_ZE - Phi_ZHL, Phi_EZ, Phi_HLZ, _0],
#         #                 [Phi_ZE, -Phi_EZ, _0, _0],
#         #                 [Phi_ZHL, _0, -Phi_HLZ, _0],
#         #                 [_0, _0, _0, _0]])
#         #
#         # K_no_D = np.transpose(K_no_D, (2, 0, 1))
#         K = np.transpose(K, (2, 0, 1))
#
#
#         q_tot_Z330, q_tot_E330 = IZ330 * self._overlap330, IE330 * self._overlap330
#         q_tot_Z400, q_tot_E400 = IZ400 * self._overlap400, IE400 * self._overlap400
#         q_tot_Z480, q_tot_E480 = IZ480 * self._overlap480, IE480 * self._overlap480
#         q_tot_Z375, q_tot_Z450 = IZ375 * self._overlap375, IZ450 * self._overlap450
#
#
#         # pool = Pool(processes=6)
#
#         # N = 8
#         # # pools = []
#         # i = 0
#
#         # [q0, c0, K, irr source],
#
#         args = [
#             ['Z', q_tot_Z330, '-initial concentraition vector', K, self.I_330],  #Z 330
#             ['E', q_tot_E330, '-initial concentraition vector', K, self.I_330],  # E start
#             ['Z', q_tot_Z375, '-initial concentraition vector', K, self.I_375],
#             ['Z', q_tot_Z400, '-initial concentraition vector', K, self.I_400],
#             ['E', q_tot_E400, '-initial concentraition vector', K, self.I_400],  # E start
#             ['Z', q_tot_Z450, '-initial concentraition vector', K, self.I_450],
#             ['Z', q_tot_Z480, '-initial concentraition vector', K, self.I_480],
#             ['E', q_tot_E480, '-initial concentraition vector', K, self.I_480],  # E start
#
#             ['Z', q0Z355LED, '-initial concentraition vector', K, self.LED_355],  # LED 355
#             ['Z', q0Z375LED, '-initial concentraition vector', K, self.LED_375],  # LED 375
#             ['Z', q0Z405LED, '-initial concentraition vector', K, self.LED_405],  # LED 405
#             ['Z', q0Z450LED, '-initial concentraition vector', K, self.LED_450],  # LED 450
#             ['Z', q0Z470LED, '-initial concentraition vector', K, self.LED_470],  # LED 470
#             ['Z', q0Z490LED, '-initial concentraition vector', K, self.LED_490],  # LED 490
#
#         ]
#
#         # E is combined spectrum of
#         Z, E_com = self.ST[0], self.ST[0] * xZ_E + (1 - xZ_E) * self.ST[1]
#         ZZ_sum = (Z * Z).sum()
#         EE_com_sum = (E_com * E_com).sum()
#
#         for i in range(len(args)):
#             s, e = self.aug_matrix._C_indiv_range(i)
#             t = self.aug_matrix.matrices[i, 0].times
#
#             A0 = self.aug_matrix.matrices[i, 0].Y[0]
#             # calculation of initial concentration of Z/E by least squares
#             c0 = (A0 * Z).sum() / ZZ_sum if args[i][0] == 'Z' else (A0 * E_com).sum() / EE_com_sum
#
#             args[i][2] = [c0, 0, 0, 0] if args[i][0] == 'Z' else [xZ_E * c0, (1 - xZ_E) * c0, 0, 0]
#
#             C_out[s:e, :] = self.simulate(*args[i][1:], wavelengths=self.wavelengths,
#                                           times=t, eps=self.ST, V=V, l=1, D=self.aug_matrix.matrices[i, 0].Y)
#
#         i = len(args)
#         #
#         # # apply closure constrain on 490 nm LED C profiles,  keep it as a fitting parameter
#         # s, e = self.aug_matrix._C_indiv_range(i)
#         # C_out[s:e, 3] = 0  # no protoproducts
#         # C_out[s:e, :] = c0Z490LED * C_out[s:e, :] / C_out[s:e, :].sum(axis=1, keepdims=True)
#
#
#         # 355 nm LED
#         # i += 1
#
#         s, e = self.aug_matrix._C_indiv_range(i)
#         c0_490_355 = C_out[s-1, :]  # concentrations at the end of 490 nm irr are the start of 355 nm irr
#
#         t = self.aug_matrix.matrices[i, 0].times
#
#         C_out[s:e, :] = self.simulate(q0Z355LED, c0_490_355, K, self.LED_355,
#                                       wavelengths=self.wavelengths, times=t, eps=self.ST, V=V, l=1,
#                                       D=self.aug_matrix.matrices[i, 0].Y)
#
#         return C_out

#
# class Half_Bilirubin_Multiset_Half(_Photokinetic_Model):
#     n = 4
#     name = '(Half) Half-Bilirubin Multiset Model'
#     n_pars_per_QY = 5
#     _class = 'Steady state photokinetics'
#     # wl_range_ZEHL = (340, 360, 402, 443, 480, 490)
#     # wl_range_HLED = (370, 409, 490)
#
#     wl_range_ZE = (355, 414, 490)
#     wl_range_EHL = (355, 380, 414, 490)
#     # wl_range_HLED = (370, 409, 485)
#     wl_range_HLED = (370, 409)
#
#     #
#     # wl_range_ZE = (355, 380, 414, 450, 490)
#     # wl_range_EHL = (355, 380, 414, 450, 490)
#     # wl_range_HLED = (370, 409, 450, 485)
#
#
#     def __init__(self, times=None, ST=None, wavelengths=None, aug_matrix=None):
#
#         super(Half_Bilirubin_Multiset_Half, self).__init__(times, method='RFA')  # RSA - resolving factor analysis
#
#         # self.wavelengths = wavelengths
#         # self.ST = ST
#         self.interp_kind = 'quadratic'
#
#         # path = r"C:\Users\Dominik\Documents\MUNI\Organic Photochemistry\Projects\2019-Bilirubin project\UV-VIS\QY measurement\Photodiode\new setup"
#         path = r"C:\Users\dominik\Documents\Projects\Bilirubin\new setup"
#
#         self.Z_true = np.loadtxt(path + r'\Z-epsilon.txt', delimiter='\t', skiprows=1, usecols=1)
#
#         fname = path + r'\em sources.txt'
#         data = np.loadtxt(fname, delimiter='\t', skiprows=1)
#
#         self.I_330 = data[:, 1] / np.trapz(data[:, 1], x=self.wavelengths)
#         self.I_375 = data[:, 2] / np.trapz(data[:, 2], x=self.wavelengths)
#         self.I_400 = data[:, 3] / np.trapz(data[:, 3], x=self.wavelengths)
#         self.I_450 = data[:, 4] / np.trapz(data[:, 4], x=self.wavelengths)
#         self.I_480 = data[:, 5] / np.trapz(data[:, 5], x=self.wavelengths)
#
#         fname = path + r'\em sources Z purif.txt'
#         data = np.loadtxt(fname, delimiter='\t', skiprows=1)
#
#         self.I_350p = data[:, 1] / np.trapz(data[:, 1], x=self.wavelengths)
#         self.I_400p = data[:, 2] / np.trapz(data[:, 2], x=self.wavelengths)
#         self.I_500p = data[:, 3] / np.trapz(data[:, 3], x=self.wavelengths)
#
#
#         data_led = np.loadtxt(path + r'\LED sources.txt', delimiter='\t', skiprows=1)
#
#         self.LED_355 = data_led[:, 1] / np.trapz(data_led[:, 1], x=self.wavelengths)
#         self.LED_375 = data_led[:, 2] / np.trapz(data_led[:, 2], x=self.wavelengths)
#         self.LED_405 = data_led[:, 3] / np.trapz(data_led[:, 3], x=self.wavelengths)
#         self.LED_420 = data_led[:, 4] / np.trapz(data_led[:, 4], x=self.wavelengths)
#         self.LED_450 = data_led[:, 5] / np.trapz(data_led[:, 5], x=self.wavelengths)
#         self.LED_470 = data_led[:, 6] / np.trapz(data_led[:, 6], x=self.wavelengths)
#         self.LED_490 = data_led[:, 7] / np.trapz(data_led[:, 7], x=self.wavelengths)
#
#
#         fname = path + r'\q rel cut.txt'
#         data = np.loadtxt(fname, delimiter='\t', skiprows=1)
#         self.Diode_q_rel = data[:, 1]
#
#         self._overlap330 = np.trapz(self.Diode_q_rel * self.I_330, x=self.wavelengths)
#         self._overlap375 = np.trapz(self.Diode_q_rel * self.I_375, x=self.wavelengths)
#         self._overlap400 = np.trapz(self.Diode_q_rel * self.I_400, x=self.wavelengths)
#         self._overlap450 = np.trapz(self.Diode_q_rel * self.I_450, x=self.wavelengths)
#         self._overlap480 = np.trapz(self.Diode_q_rel * self.I_480, x=self.wavelengths)
#
#         self._overlap350p = np.trapz(self.Diode_q_rel * self.I_350p, x=self.wavelengths)
#         self._overlap400p = np.trapz(self.Diode_q_rel * self.I_400p, x=self.wavelengths)
#         self._overlap500p = np.trapz(self.Diode_q_rel * self.I_500p, x=self.wavelengths)
#
#
#     # def update_params(self, n_pars_per_QY=5, wl_range=(340, 480)):
#     #     self.wl_range = wl_range
#     #     self.n_pars_per_QY = n_pars_per_QY
#     #     self.init_params()
#
#     def init_params(self):
#         super(Half_Bilirubin_Multiset_Half, self).init_params()
#         # self.params = Parameters()
#
#         # wls_QY = np.linspace(self.wl_range[0], self.wl_range[1], self.n_pars_per_QY, endpoint=True, dtype=int)
#
#         # amount of Z in the mixture, 1: only Z, 0:, only E
#         self.params.add('xZ_Z', value=1, min=0, max=1, vary=False)
#         self.params.add('xZ_E', value=0.068, min=0, max=1, vary=False)
#
#         self.params.add('q0_355_LED', value=5e-8, min=0, max=np.inf, vary=True)
#         self.params.add('q0_405_LED', value=5e-8, min=0, max=np.inf, vary=True)
#         # self.params.add('q0_450_LED', value=5e-8, min=0, max=np.inf, vary=True)
#         # self.params.add('q0_470_LED', value=5e-8, min=0, max=np.inf, vary=True)
#         self.params.add('q0_490_LED', value=1e-7, min=0, max=np.inf, vary=False)
#
#         for wl in self.wl_range_ZE:
#             self.params.add(f'Phi_ZE_{wl}', value=0.20, min=0, max=1, vary=True)
#
#         for wl in self.wl_range_ZE:
#             self.params.add(f'Phi_EZ_{wl}', value=0.20, min=0, max=1, vary=True)
#
#         for wl in self.wl_range_EHL:
#             self.params.add(f'Phi_EHL_{wl}', value=0.005, min=0, max=1, vary=True)
#
#         for wl in self.wl_range_HLED:
#             self.params.add(f'Phi_HLE_{wl}', value=0.003, min=0, max=1, vary=True)
#
#         for wl in self.wl_range_HLED:
#             self.params.add(f'Phi_HLD_{wl}', value=0.001, min=0, max=1, vary=True)
#
#     # calculates the least squares solution of concentrations of species of a given spectrum with given populations of species
#     def get_conc_vector(self, spectrum, populations):
#         pop = np.asarray(populations, dtype=np.float64)
#         pop /= pop.sum()
#
#         # assert pop.shape[0] == self.ST.shape[0]
#
#         ST_avrg = (pop[:, None] * self.ST[:3, :]).sum(axis=0)
#
#         STST_sum = (ST_avrg * ST_avrg).sum()
#
#         c0 = (spectrum * ST_avrg).sum() / STST_sum
#
#         return pop * c0
#
#     def get_interpolated_curves(self, pars, return_points=False):
#
#         # pars = values[5:]
#
#         # n = self.n_pars_per_QY
#         n1 = len(self.wl_range_ZE)
#         n2 = len(self.wl_range_EHL)
#         n3 = len(self.wl_range_HLED)
#
#         p_ZE = pars[:n1]
#         p_EZ = pars[n1: 2 * n1]
#         p_EHL = pars[2 * n1: 2 * n1 + n2]
#         p_HLE = pars[2 * n1 + n2: 2 * n1 + n2 + n3]
#         p_HLD = pars[2 * n1 + n2 + n3:]
#
#         _Phi_ZE = self.Phi_interp(p_ZE, self.wl_range_ZE, kind=self.interp_kind)
#         _Phi_EZ = self.Phi_interp(p_EZ, self.wl_range_ZE, kind=self.interp_kind)
#         _Phi_EHL = self.Phi_interp(p_EHL, self.wl_range_EHL, kind='cubic')
#         _Phi_HLE = self.Phi_interp(p_HLE, self.wl_range_HLED, kind='linear')
#         _Phi_HLD = self.Phi_interp(p_HLD, self.wl_range_HLED, kind='linear')
#
#         if return_points:
#             return p_ZE, p_EZ, p_EHL, p_HLE, p_HLD, _Phi_ZE, _Phi_EZ, _Phi_EHL, _Phi_HLE, _Phi_HLD
#         else:
#             return _Phi_ZE, _Phi_EZ, _Phi_EHL, _Phi_HLE, _Phi_HLD
#
#     def get_quantiles_ST(self, cov_mat, values, quantiles=(0.025, 0.975), n_samples=5000):
#
#         if self.method is not 'RFA':
#             return
#
#         values = values[:self.n ** 2]
#         cov_mat = cov_mat[:self.n ** 2, :self.n ** 2]
#
#         X = multivariate_normal.rvs(mean=values, cov=cov_mat, size=n_samples)
#
#         samples = np.zeros((n_samples, self.n, self.wavelengths.shape[0]))
#
#         for i in range(n_samples):
#             T = np.asarray(X[i]).reshape((self.n, self.n))
#             samples[i] = T.dot(self.VT)  # ST samples
#
#         q = np.quantile(samples, quantiles, axis=0)
#
#         return q
#
#     def get_quantiles(self, cov_mat, values, quantiles=(0.025, 0.975), n_samples=5000):
#
#         values = values[5:] if self.method is not 'RFA' else values[5 + self.n ** 2:]
#         # cov_mat = cov_mat[3:, 3:] if self.method is not 'RFA' else cov_mat[3 + self.n ** 2:, 3 + self.n ** 2:]
#         cov_mat = cov_mat[2:, 2:] if self.method is not 'RFA' else cov_mat[2 + self.n ** 2:, 2 + self.n ** 2:]
#
#         # cov_mat = cov_mat[5:, 5:] if self.method is not 'RFA' else cov_mat[5 + self.n ** 2:, 5 + self.n ** 2:]
#
#         X = multivariate_normal.rvs(mean=values, cov=cov_mat, size=n_samples)
#
#         n_wls = self.wavelengths.shape[0]
#
#         Phi_ZE_samples = np.zeros((n_samples, n_wls))
#         Phi_EZ_samples = np.zeros((n_samples, n_wls))
#         Phi_EHL_samples = np.zeros((n_samples, n_wls))
#         Phi_HLE_samples = np.zeros((n_samples, n_wls))
#         Phi_HLD_samples = np.zeros((n_samples, n_wls))
#
#         for i in range(n_samples):
#             Phi_ZE_samples[i], Phi_EZ_samples[i], Phi_EHL_samples[i], Phi_HLE_samples[i], Phi_HLD_samples[i] = self.get_interpolated_curves(X[i])
#
#         ZEq = np.quantile(Phi_ZE_samples, quantiles, axis=0)
#         EZq = np.quantile(Phi_EZ_samples, quantiles, axis=0)
#         EHLq = np.quantile(Phi_EHL_samples, quantiles, axis=0)
#         HLEq = np.quantile(Phi_HLE_samples, quantiles, axis=0)
#         HLDq = np.quantile(Phi_HLD_samples, quantiles, axis=0)
#
#         return ZEq, EZq, EHLq, HLEq, HLDq
#
#     def save_QY_quantiles(self, quantiles, fname='fit_QY.txt'):
#
#         pars = [par[1].value for par in self.params.items()]
#         pars = pars[5:] if self.method is not 'RFA' else pars[5 + self.n ** 2:]
#         # pars = pars[7:] if self.method is not 'RFA' else pars[7 + self.n ** 2:]
#
#         _Phi_ZE, _Phi_EZ, _Phi_EHL, _Phi_HLE, _Phi_HLD = self.get_interpolated_curves(pars, False)
#
#         buffer = ", ZE, EZ , EHL, HLE, -HL, ZEq1, ZEq2, EZq1, EZq2, EHLq1, EHLq2, HLEq1, EHLq2, -HLq1, -HLq2\n"
#
#         mat = np.vstack((self.wavelengths, _Phi_ZE, _Phi_EZ, _Phi_EHL, _Phi_HLE, _Phi_HLD) + tuple(element for tupl in quantiles for element in tupl))
#         buffer += '\n'.join("\t".join(f"{num}" for num in row) for row in mat.T)
#
#         with open(fname, 'w', encoding='utf8') as f:
#             f.write(buffer)
#
#     def plot_ST(self, quantiles=None, alpha_q=0.2):
#
#         prop_cycle = plt.rcParams['axes.prop_cycle']
#         colors = prop_cycle.by_key()['color']
#
#         for i in range(self.n):
#             plt.plot(self.wavelengths, self.ST[i], color=colors[i], label=self.species_names[i])
#             if quantiles is not None:
#                 plt.fill_between(self.wavelengths, quantiles[0][i], quantiles[1][i], alpha=alpha_q, color=colors[i])
#
#         plt.xlabel('Wavelength')
#         plt.ylabel('epsilon...')
#         plt.legend()
#
#         plt.show()
#
#
#     def plot_phis(self, y_scale='log', quantiles=None, alpha_q=0.2):
#         if self.params is None:
#             return
#
#         pars = [par[1].value for par in self.params.items()]
#         pars = pars[5:] if self.method is not 'RFA' else pars[5 + self.n ** 2:]
#         # pars = pars[7:] if self.method is not 'RFA' else pars[7 + self.n ** 2:]
#
#         p_ZE, p_EZ, p_EHL, p_HLE, p_HLD, _Phi_ZE, _Phi_EZ, _Phi_EHL, _Phi_HLE, _Phi_HLD = self.get_interpolated_curves(pars, True)
#
#         prop_cycle = plt.rcParams['axes.prop_cycle']
#         colors = prop_cycle.by_key()['color']
#
#         # x = np.linspace(self.wl_range[0], self.wl_range[1], n, dtype=int)
#
#         plt.plot(self.wavelengths, _Phi_ZE, label="$\Phi_{ZE}$")
#         plt.plot(self.wavelengths, _Phi_EZ, label="$\Phi_{EZ}$")
#         plt.plot(self.wavelengths, _Phi_EHL, label="$\Phi_{EHL}$")
#         plt.plot(self.wavelengths, _Phi_HLE, label="$\Phi_{HLE}$")
#         plt.plot(self.wavelengths, _Phi_HLD, label="$\Phi_{HLD}$")
#
#         plt.scatter(self.wl_range_ZE, p_ZE, marker='o', edgecolor='black')
#         plt.scatter(self.wl_range_ZE, p_EZ, marker='o', edgecolor='black')
#         plt.scatter(self.wl_range_EHL, p_EHL, marker='o', edgecolor='black')
#         plt.scatter(self.wl_range_HLED, p_HLE, marker='o', edgecolor='black')
#         plt.scatter(self.wl_range_HLED, p_HLD, marker='o', edgecolor='black')
#
#         if quantiles:
#             for i, q in enumerate(quantiles):
#                 plt.fill_between(self.wavelengths, q[0], q[1], alpha=alpha_q, color=colors[i])
#
#         plt.yscale(y_scale)
#
#         plt.xlabel('Wavelength')
#         plt.ylabel('$\Phi$')
#         plt.legend()
#
#         plt.show()
#
#     @staticmethod
#     @vectorize()
#     def photokin_factor(A):
#         ln10 = np.log(10)
#         ll2 = ln10 * ln10 / 2
#
#         if A < 1e-3:
#             return ln10 - A * ll2
#         else:
#             return (1 - np.exp(-A * ln10)) / A
#
#     def Phi_interp(self, phis, wl_range, kind='quadratic', crop_01=True):
#         n = len(phis)
#         # assert n >= 3
#         # x = np.linspace(wl_range[0], wl_range[1], n, dtype=int)
#         x = np.asarray(wl_range)
#         f = interp1d(x, np.asarray(phis), kind=kind, fill_value='extrapolate')
#
#         y = f(self.wavelengths)
#         if crop_01:
#             y *= (y > 0)
#             y = np.where(y > 1, 1, y)
#
#         return y
#
#     @staticmethod
#     def simulate(q0, c0, K, I_source, wavelengths=None, times=None, eps=None, V=0.003, l=1,  D=None, t0=0):
#         """
#         c0 is concentration vector at time, defined in times arary as first element (initial condition), eps is vector of molar abs. coefficients,
#         I_source is spectrum of irradiaiton source if this was used,
#         if not, w_irr as irradiaton wavelength must be specified, K is transfer matrix, l is length of a cuvette, default 1 cm
#         times are times for which to simulate the kinetics
#         """
#         # n = eps.shape[0]  # eps are epsilons - n x w matrix, where n is number of species and w is number of wavelengths
#         # assert n == K.shape[0] == K.shape[1]
#         c0 = np.asarray(c0)
#
#         c_tot = c0.sum()
#
#         # get absorbances from real data
#         # abs_at = interp2d(wavelengths, times, D, kind='linear', copy=True)
#
#         # const = l * np.log(10)
#         # tol = 1e-3
#
#         def dc_dt(c, t):
#
#             _c = np.append(c, c_tot - c.sum())
#             # #
#             # c_dot_eps = abs_at(wavelengths, t)
#             #
#             # # I = c_eps * Half_Bilirubin_Multiset_Half.photokin_factor(c_dot_eps) * I_source
#             # FI0 = Half_Bilirubin_Multiset_Half.photokin_factor(c_dot_eps) * I_source
#             #
#             # tensor = FI0[:, None, None] * K * eps.T[:, None, :]  # I0 * F * K x diag(epsilon)
#             #
#             # return q0 / V * tensor.sum(axis=0).dot(_c)
#
#             c_dot_eps = _c[:, None] * eps
#             c_eps = c_dot_eps.sum(axis=0)
#
#             FI0 = c_dot_eps * Half_Bilirubin_Multiset_Half.photokin_factor(c_eps) * I_source
#
#             # # w x n x n   x   w x n x 1
#             product = np.matmul(K, FI0.T[..., None])  # w x n x 1
#             #
#             # irr_on = 1 if t >= t0 else 0
#             #
#             # return irr_on * q0 / V * product.sum(axis=0).squeeze()
#             return q0 / V * product.sum(axis=0).squeeze()
#
#
#         result = odeint(dc_dt, c0, times)
#
#         forth_comp = c_tot - result.sum(axis=1, keepdims=True)
#
#         result = np.hstack((result, forth_comp))
#
#         return result
#
#     def calc_C(self, params=None, C_out=None):
#         super(Half_Bilirubin_Multiset_Half, self).calc_C(params)
#
#         if self.ST is None:
#             raise ValueError("Spectra matrix must not be none.")
#
#         pars = [par[1].value for par in self.params.items()]
#         pars = pars if self.method is not 'RFA' else pars[self.n ** 2:]
#         xZ_Z, xZ_E, q0_355_LED, q0_405_LED, q0_490_LED = pars[:5]
#         _Phi_ZE, _Phi_EZ, _Phi_EHL, _Phi_HLE, _Phi_HLD = self.get_interpolated_curves(pars[5:])
#
#         # xZ_Z, xZ_E, q0_355_LED, q0_405_LED, q0_450_LED, q0_470_LED, q0_490_LED = pars[:7]
#         # _Phi_ZE, _Phi_EZ, _Phi_EHL, _Phi_HLE, _Phi_HLD = self.get_interpolated_curves(pars[7:])
#
#         IZ330, IE330, IZ400, IE400, V = 16.7e-6, 17e-6, 38.1e-6, 37.7e-6, 3e-3
#         IZ375, IZ450 = 24.9e-6, 47.9e-6
#         IZ480, IE480 = 48e-6, 48e-6
#
#         IZ350p, IE350p, IZ400p, IE400p, IZ500p, IE500p = 38.6e-6, 39.8e-6, 53.0e-6, 52.0e-6, 50.1e-6, 51.2e-6
#
#         _0 = np.zeros(self.wavelengths.shape)  # if isinstance(_Phi_ZE, np.ndarray) else 0
#
#         # K = np.asarray([[-_Phi_ZE,              _Phi_EZ,             _0],
#         #                 [_Phi_ZE,      -_Phi_EHL -_Phi_EZ,     _Phi_HLE],
#         #                 [_0,                 _Phi_EHL,         -_Phi_HLE]])
#
#         K = np.asarray([[-_Phi_ZE, _Phi_EZ, _0,                   _0],
#                            [_Phi_ZE, -_Phi_EHL -_Phi_EZ,  _Phi_HLE,  _0],
#                            [_0, _Phi_EHL, -_Phi_HLE - _Phi_HLD,      _0],
#                            [_0, _0, _Phi_HLD,                       _0]])
#
#         # # alternative model if HL would be formed from Z
#         # K = np.asarray([[-_Phi_ZE -_Phi_EHL, _Phi_EZ, _Phi_HLE, _0],
#         #                 [_Phi_ZE,  - _Phi_EZ, _0, _0],
#         #                 [_Phi_EHL, _0, -_Phi_HLE - _Phi_HLD, _0],
#         #                 [_0, _0, _Phi_HLD, _0]])
#
#         K = K[:-1]  # remove last row of matrix
#
#         K = np.transpose(K, (2, 0, 1))
#
#         # q_tot_Z330, q_tot_E330 = IZ330 * self._overlap330, IE330 * self._overlap330
#         # q_tot_Z400, q_tot_E400 = IZ400 * self._overlap400, IE400 * self._overlap400
#         # q_tot_Z480, q_tot_E480 = IZ480 * self._overlap480, IE480 * self._overlap480
#         # q_tot_Z375, q_tot_Z450 = IZ375 * self._overlap375, IZ450 * self._overlap450
#
#         q_tot_Z350p, q_tot_E350p = IZ350p * self._overlap350p, IE350p * self._overlap350p
#         q_tot_Z400p, q_tot_E400p = IZ400p * self._overlap400p, IE400p * self._overlap400p
#         q_tot_Z500p, q_tot_E500p = IZ500p * self._overlap500p, IE500p * self._overlap500p
#
#         # args = [
#         #     ['Z', q_tot_Z330, '-initial concentraition vector', K, self.I_330],  # Z 330
#         #     ['E', q_tot_E330, '-initial concentraition vector', K, self.I_330],  # E start
#         #     ['Z', q_tot_Z375, '-initial concentraition vector', K, self.I_375],
#         #     ['Z', q_tot_Z400, '-initial concentraition vector', K, self.I_400],
#         #     ['E', q_tot_E400, '-initial concentraition vector', K, self.I_400],  # E start
#         #     ['Z', q_tot_Z450, '-initial concentraition vector', K, self.I_450],
#         #     ['Z', q_tot_Z480, '-initial concentraition vector', K, self.I_480],
#         #     ['E', q_tot_E480, '-initial concentraition vector', K, self.I_480],  # E start
#         #
#         #     ['Z', q0_355_LED, '-initial concentraition vector', K, self.LED_355],
#         #     ['Z', q0_405_LED, '-initial concentraition vector', K, self.LED_405],
#         #
#         #     ['Z', q0_450_LED, '-initial concentraition vector', K, self.LED_450],
#         #     ['Z', q0_470_LED, '-initial concentraition vector', K, self.LED_470],
#         #
#         #     ['Z', q0_490_LED, '-initial concentraition vector', K, self.LED_490],
#         #     #
#         #     ['HL', q0_355_LED, '-initial concentraition vector', K, self.LED_355],
#         #     ['HL', q0_405_LED, '-initial concentraition vector', K, self.LED_405],
#         #
#         # ]
#
#         # args new 'p'
#         args = [
#             ['Z', q_tot_Z350p, '-initial concentraition vector', K, self.I_350p],
#             ['E', q_tot_E350p, '-initial concentraition vector', K, self.I_350p],
#             ['Z', q_tot_Z400p, '-initial concentraition vector', K, self.I_400p],
#             ['E', q_tot_E400p, '-initial concentraition vector', K, self.I_400p],
#             ['Z', q_tot_Z500p, '-initial concentraition vector', K, self.I_500p],
#             ['E', q_tot_E500p, '-initial concentraition vector', K, self.I_500p],
#
#             ['Z', q0_355_LED, '-initial concentraition vector', K, self.LED_355],
#             ['Z', q0_405_LED, '-initial concentraition vector', K, self.LED_405],
#
#             # ['Z', q0_490_LED, '-initial concentraition vector', K, self.LED_490],
#             ['HL', q0_355_LED, '-initial concentraition vector', K, self.LED_355],
#             ['HL', q0_405_LED, '-initial concentraition vector', K, self.LED_405],
#         ]
#
#         # t0s = [0] * 8 + [9.5, 9.5, 3.8, 3.8, 8.8, 8, 10]
#
#         for i in range(len(args)):
#             s, e = self.aug_matrix._C_indiv_range(i)
#             t = self.aug_matrix.matrices[i, 0].times
#
#             A0 = self.aug_matrix.matrices[i, 0].Y[0]
#             # # calculation of initial concentration of Z/E by least squares
#             # c0 = (A0 * Z).sum() / ZZ_sum if args[i][0] == 'Z' else (A0 * E_com).sum() / EE_com_sum
#             #
#             # args[i][2] = [c0, 0, 0] if args[i][0] == 'Z' else [xZ_E * c0, (1 - xZ_E) * c0, 0]
#
#             if args[i][0] == 'Z':
#                 args[i][2] = self.get_conc_vector(A0, [xZ_Z, 1 - xZ_Z, 0])
#
#             elif args[i][0] == 'E':
#                 args[i][2] = self.get_conc_vector(A0, [xZ_E, 1 - xZ_E, 0])
#
#             else:
#                 args[i][2] = self.get_conc_vector(A0, [0, 0, 1])
#
#
#             C_out[s:e, :] = self.simulate(*args[i][1:], wavelengths=self.wavelengths,
#                                           times=t, eps=self.ST, V=V, l=1, D=self.aug_matrix.matrices[i, 0].Y, t0=0)
#
#         return C_out
#


class Test_Bilirubin_Multiset(_Photokinetic_Model):
    name = 'Test-Bilirubin Multiset Model'
    _class = 'Steady state photokinetics'

    def __init__(self, times=None, ST=None, wavelengths=None, aug_matrix=None):
        super(Test_Bilirubin_Multiset, self).__init__(times)


        # path = r"C:\Users\Dominik\Documents\MUNI\Organic Photochemistry\Projects\2019-Bilirubin project\UV-VIS\QY measurement\Photodiode\new setup"
        # path = r"C:\Users\dominik\Documents\Projects\Bilirubin\UV-Vis data"
        #
        # fname = path + r'\em sources.txt'
        # data = np.loadtxt(fname, delimiter='\t', skiprows=1)
        #
        # self.I_330 = data[:, 1] / np.trapz(data[:, 1], x=self.wavelengths)
        # self.I_375 = data[:, 2] / np.trapz(data[:, 2], x=self.wavelengths)
        # self.I_400 = data[:, 3] / np.trapz(data[:, 3], x=self.wavelengths)
        # self.I_450 = data[:, 4] / np.trapz(data[:, 4], x=self.wavelengths)
        # self.I_480 = data[:, 5] / np.trapz(data[:, 5], x=self.wavelengths)
        #
        # data_led = np.loadtxt(path + r'\LED sources.txt', delimiter='\t', skiprows=1)
        #
        # self.LED_355 = data_led[:, 1] / np.trapz(data_led[:, 1], x=self.wavelengths)
        # self.LED_375 = data_led[:, 2] / np.trapz(data_led[:, 2], x=self.wavelengths)
        # self.LED_405 = data_led[:, 3] / np.trapz(data_led[:, 3], x=self.wavelengths)
        # self.LED_420 = data_led[:, 4] / np.trapz(data_led[:, 4], x=self.wavelengths)
        # self.LED_450 = data_led[:, 5] / np.trapz(data_led[:, 5], x=self.wavelengths)
        # self.LED_470 = data_led[:, 6] / np.trapz(data_led[:, 6], x=self.wavelengths)
        # self.LED_490 = data_led[:, 7] / np.trapz(data_led[:, 7], x=self.wavelengths)
        #
        # fname = path + r'\q rel cut.txt'
        # data = np.loadtxt(fname, delimiter='\t', skiprows=1)
        # self.Diode_q_rel = data[:, 1]
        #
        # self._overlap330 = np.trapz(self.Diode_q_rel * self.I_330, x=self.wavelengths)
        # self._overlap375 = np.trapz(self.Diode_q_rel * self.I_375, x=self.wavelengths)
        # self._overlap400 = np.trapz(self.Diode_q_rel * self.I_400, x=self.wavelengths)
        # self._overlap450 = np.trapz(self.Diode_q_rel * self.I_450, x=self.wavelengths)
        # self._overlap480 = np.trapz(self.Diode_q_rel * self.I_480, x=self.wavelengths)
        #
        # self._overlap500 = np.trapz(self.Diode_q_rel * self.I_500, x=self.wavelengths)

    def init_params(self):
        super(Test_Bilirubin_Multiset, self).init_params()

        self.params.add('c0Z355', value=5e-5, min=0, max=np.inf, vary=True)
        self.params.add('q0Z355', value=5e-8, min=0, max=np.inf, vary=False)

        self.params.add('c0E355', value=2e-5, min=0, max=np.inf, vary=True)
        self.params.add('q0E355', value=5e-8, min=0, max=np.inf, vary=False)

        self.params.add('c0Z470', value=4e-5, min=0, max=np.inf, vary=True)
        self.params.add('q0Z470', value=2e-6, min=0, max=np.inf, vary=False)


        self.params.add('Phi_ZE', value=0.2, min=0, max=1)
        self.params.add('Phi_EZ', value=0.25, min=0, max=1)
        self.params.add('Phi_EHL', value=0.005, min=0, max=1, vary=True)
        self.params.add('Phi_HLE', value=0.001, min=0, max=1, vary=True)


    def calc_C(self, params=None, C_out=None):
        super(Test_Bilirubin_Multiset, self).calc_C(params)


        c0Z355, q0Z355, c0E355, q0E355, c0Z470, q0Z470, Phi_ZE, Phi_EZ, Phi_ZHL, Phi_HLZ = [par[1].value for par in self.params.items()]
        V = 0.003


        _0 = np.zeros(self.wavelengths.shape) if isinstance(Phi_ZE, np.ndarray) else 0

        K = np.asarray([[-Phi_ZE - Phi_ZHL,   Phi_EZ,        Phi_HLZ],
                        [Phi_ZE,              -Phi_EZ,            _0],
                        [Phi_ZHL,             _0,          -Phi_HLZ]])

        # K = np.transpose(K, (2, 0, 1))


        # K = np.asarray([[-Phi_ZE - Phi_ZHL, Phi_EZ, Phi_HLZ, _0],
        #                 [Phi_ZE, -Phi_EZ, _0, _0],
        #                 [Phi_ZHL, _0, -Phi_HLZ - Phi_HL, _0],
        #                 [_0, _0, Phi_HL, _0]])
        #
        # # # no photoproduct
        # K_no_D = np.asarray([[-Phi_ZE - Phi_ZHL, Phi_EZ, Phi_HLZ, _0],
        #                 [Phi_ZE, -Phi_EZ, _0, _0],
        #                 [Phi_ZHL, _0, -Phi_HLZ, _0],
        #                 [_0, _0, _0, _0]])

        # K_no_D = np.transpose(K_no_D, (2, 0, 1))

        args = [
            [q0Z355, [c0Z355, 0, 0], K, self.LED_355],  #Z 355
            [q0E355, [0, c0E355, 0], K, self.LED_355],  # E 355
            [q0Z470, [c0Z470, 0, 0], K, self.LED_470],  # Z 470

        ]
        for i in range(len(args)):
            s, e = self.aug_matrix._C_indiv_range(i)
            t = self.aug_matrix._matrices[i, 0].times

            C_out[s:e, :] = self.simulate(*args[i], wavelengths=self.wavelengths,
                                          times=t, eps=self.ST, V=V, l=1, D=self.aug_matrix._matrices[i, 0].Y)



        return C_out


#
# class Z_purified(_Model):
#     n = 4
#     name = 'Z_purified Multiset model'
#     _class = 'Steady state photokinetics'
#     # n_pars_per_QY = 5
#     wl_range = (340, 480)
#
#     def __init__(self, times=None, ST=None, wavelengths=None, aug_matrix=None):
#         super(Z_purified, self).__init__(times)
#
#         self.wavelengths = wavelengths
#         self.aug_matrix = aug_matrix
#
#         self.ST = ST
#         # self.interp_kind = 'quadratic'
#         self.species_names = np.array(list('ZEHD'), dtype=np.str)
#
#         # path = r"C:\Users\dominik\Documents\Projects\Bilirubin\UV-Vis Z purified"
#         path = r"C:\Users\Dominik\Documents\MUNI\Organic Photochemistry\Projects\2019-Bilirubin project\UV-VIS\QY measurement\Photodiode\Z purified"
#
#         data_led = np.loadtxt(path + r'\LED sources.txt', delimiter='\t', skiprows=1)
#
#         self.LED_355 = data_led[:, 1] / np.trapz(data_led[:, 1], x=self.wavelengths)
#         self.LED_375 = data_led[:, 2] / np.trapz(data_led[:, 2], x=self.wavelengths)
#         self.LED_405 = data_led[:, 3] / np.trapz(data_led[:, 3], x=self.wavelengths)
#         self.LED_420 = data_led[:, 4] / np.trapz(data_led[:, 4], x=self.wavelengths)
#         self.LED_450 = data_led[:, 5] / np.trapz(data_led[:, 5], x=self.wavelengths)
#         self.LED_470 = data_led[:, 6] / np.trapz(data_led[:, 6], x=self.wavelengths)
#         self.LED_490 = data_led[:, 7] / np.trapz(data_led[:, 7], x=self.wavelengths)
#
#         fname = path + r'\em sources Z purif.txt'
#         data = np.loadtxt(fname, delimiter='\t', skiprows=1)
#
#         self.I_350 = data[:, 1] / np.trapz(data[:, 1], x=self.wavelengths)
#         self.I_400 = data[:, 2] / np.trapz(data[:, 2], x=self.wavelengths)
#         self.I_500 = data[:, 3] / np.trapz(data[:, 3], x=self.wavelengths)
#
#         fname = path + r'\q rel cut.txt'
#         data = np.loadtxt(fname, delimiter='\t', skiprows=1)
#         self.Diode_q_rel = data[:, 1]
#
#         self._overlap350 = np.trapz(self.Diode_q_rel * self.I_350, x=self.wavelengths)
#         self._overlap400 = np.trapz(self.Diode_q_rel * self.I_400, x=self.wavelengths)
#         self._overlap500 = np.trapz(self.Diode_q_rel * self.I_500, x=self.wavelengths)
#
#         self.description = ""
#
#
#     def init_params(self):
#         self.params = Parameters()
#
#         # amount of Z in the mixture, 1: only Z, 0:, only E
#
#         # self.params.add('q0_355_LED', value=1e-6, min=0, max=np.inf, vary=True)
#         # self.params.add('q0_405_LED', value=1e-6, min=0, max=np.inf, vary=True)
#         # self.params.add('q0_490_LED', value=1e-6, min=0, max=np.inf, vary=True)
#
#         # self.params.add('Phi_ZE_350', value=0.2, min=0, max=1, vary=True)
#         # self.params.add('Phi_EZ_350', value=0.2, min=0, max=1, vary=True)
#         # self.params.add('Phi_EHL_350', value=0.005, min=0, max=1, vary=True)
#         # self.params.add('Phi_HLE_350', value=0.005, min=0, max=1, vary=True)
#         # self.params.add('Phi_HLD_350', value=0.001, min=0, max=1, vary=True)
#         # self.params.add('Phi_ZD_350', value=0.00, min=0, max=1, vary=False)
#
#         # self.params.add('Phi_ZE_400', value=0.2, min=0, max=1, vary=True)
#         # self.params.add('Phi_EZ_400', value=0.2, min=0, max=1, vary=True)
#         # self.params.add('Phi_EHL_400', value=0.005, min=0, max=1, vary=True)
#         # self.params.add('Phi_HLE_400', value=0.005, min=0, max=1, vary=True)
#         # self.params.add('Phi_HLD_400', value=0.001, min=0, max=1, vary=True)
#         # self.params.add('Phi_ZD_400', value=0.00, min=0, max=1, vary=False)
#         #
#         self.params.add('Phi_ZE_500', value=0.2, min=0, max=1, vary=True)
#         self.params.add('Phi_EZ_500', value=0.2, min=0, max=1, vary=True)
#         self.params.add('Phi_EHL_500', value=0.005, min=0, max=1, vary=True)
#         self.params.add('Phi_HLE_500', value=0.005, min=0, max=1, vary=True)
#         self.params.add('Phi_HLD_500', value=0.001, min=0, max=1, vary=True)
#         self.params.add('Phi_ZD_500', value=0.00, min=0, max=1, vary=False)
#
#         self.params.add('pop_D_in_E', value=0.00, min=0, max=1, vary=False)
#         self.params.add('pop_D_in_HL', value=0.00, min=0, max=1, vary=False)
#
#         self.params.add('xZ_E', value=0.055, min=0, max=1, vary=False)
#
#         self.params.add('t0_0', value=6.6, min=0, max=20, vary=False)
#         self.params.add('t0_1', value=3.8, min=0, max=20, vary=False)
#
#
#
#
#     @staticmethod
#     def simulate(q0, c0, K, I_source, wavelengths=None, times=None, eps=None, V=0.003, l=1,  D=None, t0=0):
#         """
#         c0 is concentration vector at time, defined in times arary as first element (initial condition), eps is vector of molar abs. coefficients,
#         I_source is spectrum of irradiaiton source if this was used,
#         if not, w_irr as irradiaton wavelength must be specified, K is transfer matrix, l is length of a cuvette, default 1 cm
#         times are times for which to simulate the kinetics
#         """
#         # n = eps.shape[0]  # eps are epsilons - n x w matrix, where n is number of species and w is number of wavelengths
#         # assert n == K.shape[0] == K.shape[1]
#         c0 = np.asarray(c0)
#
#         # get absorbances from real data
#         abs_at = interp2d(wavelengths, times, D, kind='linear', copy=True)
#
#         const = l * np.log(10)
#         tol = 1e-3
#
#         def dc_dt(c, t):
#
#             c_eps = c[..., None] * eps  # hadamard product
#
#             c_dot_eps = abs_at(wavelengths, t)
#
#             x_abs = c_eps * np.where(c_dot_eps <= tol, const - c_dot_eps * const * const / 2,
#                                      (1 - np.exp(-c_dot_eps * const)) / c_dot_eps) * I_source
#
#             # w x n x n   x   w x n x 1
#             product = np.matmul(K, x_abs.T[..., None])  # w x n x 1
#
#             irr_on = 1 if t >= t0 else 0
#
#             return irr_on * q0 / V * np.trapz(product, x=wavelengths, axis=0).squeeze()
#
#         result = odeint(dc_dt, c0, times)
#
#         return result
#
#     # calculates the least squares solution of concentrations of species of a given spectrum with given populations of species
#     def get_conc_vector(self, spectrum, populations):
#         pop = np.asarray(populations, dtype=np.float64)
#         pop /= pop.sum()
#         assert pop.shape[0] == self.ST.shape[0]
#
#         ST_avrg = (pop[:, None] * self.ST).sum(axis=0)
#
#         STST_sum = (ST_avrg * ST_avrg).sum()
#
#         c0 = (spectrum * ST_avrg).sum() / STST_sum
#
#         return pop * c0
#
#     def calc_C(self, params=None, C_out=None):
#         super(Z_purified, self).calc_C(params)
#
#         if self.ST is None:
#             raise ValueError("Spectra matrix must not be none.")
#
#         # q0_355_LED, q0_405_LED, q0_490_LED, \
#         # Phi_ZE_350, Phi_EZ_350, Phi_EHL_350, Phi_HLE_350, Phi_HLD_350, Phi_ZD_350, \
#         # Phi_ZE_400, Phi_EZ_400, Phi_EHL_400, Phi_HLE_400, Phi_HLD_400, Phi_ZD_400, \
#         # Phi_ZE_500, Phi_EZ_500, Phi_EHL_500, Phi_HLE_500, Phi_HLD_500, Phi_ZD_500, \
#         # pop_D_in_E, pop_D_in_HL, xZ_E = [par[1].value for par in self.params.items()]
#
#         Phi_ZE_500, Phi_EZ_500, Phi_EHL_500, Phi_HLE_500, Phi_HLD_500, Phi_ZD_500, \
#         pop_D_in_E, pop_D_in_HL, xZ_E, t0_0, t0_1 = [par[1].value for par in self.params.items()]
#
#         # currents (in A) from power meter of incident photon flux
#         IZ350, IE350, IZ400, IE400, IZ500, IE500 = 38.6e-6, 39.8e-6, 53.0e-6, 52.0e-6, 50.1e-6, 51.2e-6
#         # volume of the sample in cuvette
#         V = 3e-3
#
#         q_tot_Z350, q_tot_E350 = IZ350 * self._overlap350, IE350 * self._overlap350
#         q_tot_Z400, q_tot_E400 = IZ400 * self._overlap400, IE400 * self._overlap400
#         q_tot_Z500, q_tot_E500 = IZ500 * self._overlap500, IE500 * self._overlap500
#
#         # xExZ = (1-xZ_E) / xZ_E
#
#         # K350 = np.asarray([[-Phi_ZE_350   -Phi_ZD_350,   Phi_EZ_350,         0,            0],
#         #                    [Phi_ZE_350,   -Phi_EZ_350 - Phi_EHL_350,    Phi_HLE_350,       0],
#         #                    [0,      Phi_EHL_350,   -Phi_HLE_350 - Phi_HLD_350,             0],
#         #                    [Phi_ZD_350,      0,             Phi_HLD_350,                   0]])
#
#         # K400 = np.asarray([[-Phi_ZE_400 - Phi_ZD_400, Phi_EZ_400, 0, 0],
#         #                    [Phi_ZE_400, -Phi_EZ_400 - Phi_EHL_400, Phi_HLE_400, 0],
#         #                    [0, Phi_EHL_400, -Phi_HLE_400 - Phi_HLD_400, 0],
#         #                    [Phi_ZD_400, 0, Phi_HLD_400, 0]])
#         #
#         K500 = np.asarray([[-Phi_ZE_500 - Phi_ZD_500, Phi_EZ_500, 0, 0],
#                            [Phi_ZE_500, -Phi_EZ_500 - Phi_EHL_500, Phi_HLE_500, 0],
#                            [0, Phi_EHL_500, -Phi_HLE_500 - Phi_HLD_500, 0],
#                            [Phi_ZD_500, 0, Phi_HLD_500, 0]])
#
#
#         # time of start of irradiation
#         # t0_s = [6.6, 6, 9.5, 9.5, 9.5, 9.5, 3.8, 3.8, 8.8, 8, 10]
#
#         t0_s = [9.5, 9.5, t0_0, t0_1, 8]
#
#         args = [
#             # ['Z', q_tot_Z350, '-initial concentraition vector', K350, self.I_350],
#             # ['E', q_tot_E350, '-initial concentraition vector', K350, self.I_350],
#             # ['Z', q_tot_Z400, '-initial concentraition vector', K400, self.I_400],
#             # ['E', q_tot_E400, '-initial concentraition vector', K400, self.I_400],
#             ['Z', q_tot_Z500, '-initial concentraition vector', K500, self.I_500],
#             ['E', q_tot_E500, '-initial concentraition vector', K500, self.I_500],
#             #
#             # ['Z', q0_355_LED, '-initial concentraition vector', K350, self.LED_355],
#             # ['Z', q0_405_LED, '-initial concentraition vector', K400, self.LED_405],
#             # ['Z', q0_490_LED, '-initial concentraition vector', K500, self.LED_490],
#             # #
#             # ['HL', q0_355_LED, '-initial concentraition vector', K350, self.LED_355],
#             # ['HL', q0_405_LED, '-initial concentraition vector', K400, self.LED_405],
#
#         ]
#
#         for i in range(len(args)):
#             s, e = self.aug_matrix._C_indiv_range(i)
#             t = self.aug_matrix.matrices[i, 0].times
#
#             A0 = self.aug_matrix.matrices[i, 0].Y[0]
#
#             if args[i][0] == 'Z':
#                 args[i][2] = self.get_conc_vector(A0, [1, 0, 0, 0])
#
#             elif args[i][0] == 'E':
#                 args[i][2] = self.get_conc_vector(A0, [xZ_E * (1 - pop_D_in_E), (1 - pop_D_in_E) * (1-xZ_E), 0, pop_D_in_E])
#
#             else:
#                 args[i][2] = self.get_conc_vector(A0, [0, 0, 1 - pop_D_in_HL, pop_D_in_HL])
#
#             C_out[s:e, :] = self.simulate(*args[i][1:], wavelengths=self.wavelengths,
#                                           times=t, eps=self.ST, V=V, l=1, D=self.aug_matrix.matrices[i, 0].Y, t0=t0_s[i])
#
#         return C_out


class PKA_Titration(_Model):
    """Mixed first and second order kinetics, d[A]/dt = -k1[A] - k2[A]^2"""
    n = 2   # subject of change
    name = 'pKa'
    _class = 'Equilibrium'

    def __init__(self, times=None):
        super(PKA_Titration, self).__init__(times)

        self.description = "TODO "

    @staticmethod
    def simulate(pkas, pH):
        n = len(pkas)
        _pkas = np.sort(np.asarray(pkas))

        fs = 10 ** (_pkas[:, None] - pH[None, :])  # factors for pKas

        profiles = np.ones((n + 1, pH.shape[0]))
        for i in range(n):
            profiles[i] = fs[i:, :].prod(axis=0, keepdims=False)

        profiles /= profiles.sum(axis=0)  # divide each profile by sum of all profiles

        return profiles.T

    def init_model_params(self):
        params = Parameters()
        params.add('c0', value=1, min=0, max=np.inf, vary=False)

        for i in range(self.n - 1):
            params.add(f'pKa_{i+1}', value=1, min=-np.inf, max=np.inf)

        return params

    def calc_C(self, params=None, C_out=None):
        super(PKA_Titration, self).calc_C(params, C_out)

        c0, *pKas = [par[1].value for par in self.params.items()]

        self.C = c0 * self.simulate(pKas, self.times)

        return self.get_conc_matrix(C_out, self._connectivity)


class Gibs_Eq(_Model):
    n = 2
    name = 'Gibs'
    _class = 'Equilibrium'

    def __init__(self, times=None):
        super(Gibs_Eq, self).__init__(times)

        self.species_names = np.array(list('AB'), dtype=np.str)

        self.description = "TODO "

    def init_model_params(self):
        params = Parameters()
        params.add('c0', value=1, min=0, max=np.inf, vary=False)
        params.add('dH', value=1e3, min=-np.inf, max=np.inf)
        params.add('dS', value=0, min=-np.inf, max=np.inf, vary=False)

        return params

    def calc_C(self, params=None, C_out=None):
        super(Gibs_Eq, self).calc_C(params, C_out)

        c0, dH, dS = [par[1].value for par in self.params.items()]

        R = 8.314
        T = self.times  # absolute temperatures

        pop = np.zeros((T.shape[0], 2))

        K = np.exp(-dH / (R * T) + dS / R)

        pop[:, 0] = 1 / (1 + K)
        pop[:, 1] = 1 - pop[:, 0]

        self.C = c0 * pop

        return self.get_conc_matrix(C_out, self._connectivity)



def plot_figures(model):
    plt.figure(1)
    n = model.n

    for i in range(n):
        max_y = np.max(model.C[:, i])
        plt.plot(model.times, model.C[:, i], label='Species {}'.format(model.get_species_name(i)))
        # plt.plot(model.times, model.C[:, i] / max_y, label='Species {}'.format(model.get_species_name(i))) # norm.

        # plt.subplot(1, n, i + n + 1)
        # plt.plot(self.times, self.C[:, i])
        # plt.title('Conc. {}'.format(i + 1))

    # cAcB = model.C[:, 0] * model.C[:, 1]
    # max_y = np.max(cAcB)
    # plt.plot(model.times, cAcB / max_y, label='cA * cB norm')  # norm.
    plt.legend()
    plt.show()


if __name__ == '__main__':
    times = np.linspace(0, 20, num=1001)

    # m = ABCDE_Model(times, visible=[True, True, False, True, True])
    m = Delayed_Fl(times, visible=[True, True, True, False])

    params = m.params

    # qT1 = 0.4  # zastoupení T1 stavu, if qT1 == 1, ABCD model
    params['c0'].value = 1
    params['k0'].value = 0.3
    params['k1'].value = 0.7
    params['kTTbp'].value = 0.1
    params['kqTT'].value = 5
    params['kISC'].value = 1
    params['kd'].value = 0.1  # T1->T0

    # params['k2'].value = 0.1
    # params['n'].value = 2

    C = m.calc_C(params)

    plot_figures(m)

    # print(C[0:5])
