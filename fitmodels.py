import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from lmfit import Parameters
from abc import abstractmethod
from gui_console import Console

from multiprocessing import Pool

import math
from numba import njit, prange
from scipy.interpolate import interp2d


def find_nearest_idx(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return idx - 1
    else:
        return idx


def find_nearest(array, value):
    idx = find_nearest_idx(array, value)
    return array[idx]


# virtual class that every model must inherits
class _Model(object):
    # n = 0  # number of visible species in model
    n = 0  # number of all possible species

    params = None
    species_names = None
    # connectivity = None

    name = 'AB Model'
    description = "..."

    _err = 1e-8

    def __init__(self, times=None, connectivity=(0, 1, 2)):
        self.times = times
        self.C = None
        self._connectivity = connectivity

        self.init_times(times)

        self.init_params()
        self.species_names = np.array(list('AB'), dtype=np.str)

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

    # # virtual method
    # def init_visible(self):
    #     self.visible = (self.n_full - 1) * [True] + [False]

    @abstractmethod
    def init_params(self):
        self.params = []

    def get_conc_matrix(self, C_out, connectivity=(1, 2, 3)):
        """Replaces the values in C_out according to calculated values based on conectivity"""
        if C_out is None:
            return self.C
        else:
            assert C_out.shape[1] == len(connectivity)

            for i in range(len(connectivity)):
                # for values > 0 - 0 means MCR fit, replace the values, then 1 - A, 2 - B, 3 - C, etc.
                if connectivity[i] > 0:
                    C_out[:, i] = self.C[:, connectivity[i] - 1]

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


class AB_Model(_Model):
    """Simple A->B model, default is 1st order reaction, 2nd order and n-th order can be selected as well. Also,
    user can define whether both A and B are visible, or only A or B is visible. Default is [True, False] - only A is
     visible"""

    order = '1st'
    n = 2
    name = 'A→B (variable order)'

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

    def __init__(self, times=None):
        super(AB_mixed12_Model, self).__init__(times)

        self.species_names = np.array(list('AB'), dtype=np.str)

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

    def __init__(self, times=None):
        super(ABDE_Model, self).__init__(times)

        self.species_names = np.array(list('ABCD'), dtype=np.str)

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


class ABC_Model(_Model):
    """ABC kinetic model, first order."""

    n = 3
    name = 'A→B→C (1st order)'

    def __init__(self, times=None):
        super(ABC_Model, self).__init__(times)

        self.species_names = np.array(list('ABC'), dtype=np.str)

        self.description = "Simple A->B->C model of 1st order. d[A]/dt = -k1[A] - k2[A]^2, [A]_0 = c_0"

    def init_params(self):
        self.params = Parameters()
        self.params.add('c0', value=1, min=0, max=np.inf, vary=False)
        self.params.add('k1', value=1, min=0, max=np.inf)
        self.params.add('k2', value=0.5, min=0, max=np.inf)

    def calc_C(self, params=None, C_out=None):
        super(ABC_Model, self).calc_C(params)

        c0, k1, k2 = self.params['c0'].value, self.params['k1'].value, self.params['k2'].value

        self.C[:, 0] = np.heaviside(self.times, 1) * c0 * np.exp(-self.times * k1)
        self.C[:, 1] = self.cB(self.times, c0, k1, k2)
        self.C[:, 2] = np.heaviside(self.times, 1) * (c0 - self.C[:, 0] - self.C[:, 1])

        return self.get_conc_matrix(C_out, self._connectivity)


class ABCD_Model(_Model):
    """ABCD kinetic model, first order."""

    n = 4
    name = 'A→B→C→D (1st order)'

    def __init__(self, times=None, visible=None):
        super(ABCD_Model, self).__init__(times, visible)

        self.species_names = np.array(list('ABCD'), dtype=np.str)

        # self.description = "Simple A->B->C model of 1st order. d[A]/dt = -k1[A] - k2[A]^2, [A]_0 = c_0"

    def init_params(self):
        self.params = Parameters()
        self.params.add('c0', value=1, min=0, max=np.inf, vary=False)
        self.params.add('k1', value=1, min=0, max=np.inf)
        self.params.add('k2', value=0.5, min=0, max=np.inf)
        self.params.add('k3', value=0.2, min=0, max=np.inf)

    def calc_C(self, params=None, C_out=None):
        super(ABCD_Model, self).calc_C(params)

        c0, k1, k2, k3 = self.params['c0'].value, self.params['k1'].value, self.params['k2'].value, self.params[
            'k3'].value

        self.C[:, 0] = np.heaviside(self.times, 1) * c0 * np.exp(-self.times * k1)
        self.C[:, 1] = self.cB(self.times, c0, k1, k2)

        def dC_dt(cC, t):
            cB = self.cB(t, c0, k1, k2)
            return k2 * cB - k3 * cC  # d[C]/dt = k2[B] - k3[C]

        # initial condition, cC(t=0) = 0

        # initial conditiona, cC(t=0) = 0, cD(t=0) = 0
        x0 = np.linspace(0, self.times[0], num=100)
        _init_x = odeint(dC_dt, 0, x0)[-1, :]  # take the row in the result matrix
        result = odeint(dC_dt, _init_x, self.times)

        self.C[:, 2] = np.heaviside(self.times, 1) * result.flatten()
        self.C[:, 3] = np.heaviside(self.times, 1) * (
                c0 - self.C[:, 0] - self.C[:, 1] - self.C[:, 2])

        return self.get_conc_matrix(C_out, self._connectivity)


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


class ABCDE_Model(_Model):
    """ABCDE kinetic model, first order."""

    n = 5
    name = 'A→B→C→D→E (1st order)'

    def __init__(self, times=None, visible=None):
        super(ABCDE_Model, self).__init__(times, visible)

        self.species_names = np.array(list('ABCDE'), dtype=np.str)

    def init_params(self):
        self.params = Parameters()
        self.params.add('c0', value=1, min=0, max=np.inf, vary=False)
        self.params.add('k1', value=1, min=0, max=np.inf)
        self.params.add('k2', value=0.5, min=0, max=np.inf)
        self.params.add('k3', value=0.4, min=0, max=np.inf)
        self.params.add('k4', value=0.3, min=0, max=np.inf)

    def calc_C(self, params=None, C_out=None):
        super(ABCDE_Model, self).calc_C(params)

        c0, k1, k2, k3, k4 = self.params['c0'].value, self.params['k1'].value, self.params['k2'].value, \
                             self.params['k3'].value, self.params['k4'].value

        self.C[:, 0] = np.heaviside(self.times, 1) * c0 * np.exp(-self.times * k1)
        self.C[:, 1] = self.cB(self.times, c0, k1, k2)

        def solve(conc, t):
            cC, cD = conc
            cB = self.cB(t, c0, k1, k2)
            dC_dt = k2 * cB - k3 * cC  # d[C]/dt = k2[B] - k3[C]
            dD_dt = k3 * cC - k4 * cD  # d[D]/dt = k3[C] - k4[D]
            return [dC_dt, dD_dt]

        # initial conditiona, cC(t=0) = 0, cD(t=0) = 0
        x0 = np.linspace(0, self.times[0], num=100)
        _init_x = odeint(solve, [0, 0], x0)[-1, :]  # take the row in the result matrix
        result = odeint(solve, _init_x, self.times)

        self.C[:, 2] = np.heaviside(self.times, 1) * result[:, 0]
        self.C[:, 3] = np.heaviside(self.times, 1) * result[:, 1]

        self.C[:, 4] = np.heaviside(self.times, 1) * (
                c0 - self.C[:, 0] - self.C[:, 1] - self.C[:, 2] - self.C[:, 3])

        return self.get_conc_matrix(C_out, self._connectivity)


class Delayed_Fl(_Model):
    """Delayed fluorescence"""

    n = 4
    name = 'Delayed_Fl'

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
        for i, mat in enumerate(self.aug_matrix.matrices[:, 0]):
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


class Half_Bilirubin_1st_Model(_Model):
    n = 4
    name = '1st Half Bilirubin Photokinetics'

    def __init__(self, times=None, ST=None, wavelengths=None):
        super(Half_Bilirubin_1st_Model, self).__init__(times)

        self.species_names = np.array(list('ZEHU'), dtype=np.str)
        self.wavelengths = wavelengths
        self.ST = ST
        self.I_source = None

        self.description = "Simple A->B->C model of 1st order. d[A]/dt = -k1[A] - k2[A]^2, [A]_0 = c_0"

    def init_params(self):
        self.params = Parameters()
        self.params.add('q0', value=9.103e-10, min=0, max=np.inf, vary=False)  # molar photon flux in mol s-1
        self.params.add('V', value=2.5e-3, min=0, max=np.inf, vary=False)  # volume in mL
        self.params.add('w_irr', value=370, min=0, max=np.inf, vary=False)  # irradiating wavelength
        self.params.add('c0', value=2.90881898e-05, min=0, max=np.inf,
                        vary=False)  # total starting concentration of isomers, cZ+cE
        self.params.add('xZ', value=1, min=0, max=1,
                        vary=False)  # amount of Z in the mixture, 1: only Z, 0:, only E
        self.params.add('Phi_ZE', value=0.25, min=0, max=1)
        self.params.add('Phi_EZ', value=0.23, min=0, max=1)
        self.params.add('Phi_EHL', value=0.001, min=0, max=1)
        self.params.add('Phi_HLBl', value=0.0001, min=0, max=1, vary=False)

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


class Bridge_Splitting(_Model):
    n = 2
    name = 'Bridge_Splitting equilibrium D+2L->2M'

    def __init__(self, times=None, ST=None, wavelengths=None):
        super(Bridge_Splitting, self).__init__(times)

        self.species_names = np.array(list('DM'), dtype=np.str)
        self.wavelengths = wavelengths
        self.ST = ST

        self.description = "asddddd"

    def init_params(self):
        self.params = Parameters()
        self.params.add('K', value=2, min=0, max=np.inf, vary=True)  # equilibrium constant
        self.params.add('c0', value=1e-5, min=0, max=np.inf, vary=False)  # intial dimer concentration

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
    name = 'Bridge_Splitting equilibrium D+L->M'

    def __init__(self, times=None, ST=None, wavelengths=None):
        super(Bridge_Splitting_Simple, self).__init__(times)

        self.species_names = np.array(list('DM'), dtype=np.str)
        self.wavelengths = wavelengths
        self.ST = ST

        self.description = "asddddd"

    def init_params(self):
        self.params = Parameters()
        self.params.add('K', value=2, min=0, max=np.inf, vary=True)  # equilibrium constant
        self.params.add('c0', value=1e-5, min=0, max=np.inf, vary=False)  # intial dimer concentration

    def calc_C(self, params=None, C_out=None):
        super(Bridge_Splitting_Simple, self).calc_C(params)

        K, c0 = [par[1].value for par in self.params.items()]

        cL = self.times
        alpha = K * cL / (1 + K * cL)

        self.C[:, 0] = np.heaviside(cL, 1) * (1 - alpha) * c0
        self.C[:, 1] = np.heaviside(cL, 1) * c0 * alpha

        return self.get_conc_matrix(C_out, self._connectivity)

    # @staticmethod
    # def fk_factor(x, c=np.log(10), tol=1e-2):
    #     # exp(-xc) = 1 - xc + (xc)^2 / 2 - (xc)^3 / 6 ...
    #     # (1 - exp(-xc)) / x  ~  c - xc^2 / 2 for low x
    #     return np.where(x <= tol, c - x * c * c / 2 + x * x * c * c * c / 6, (1 - np.exp(-x * c)) / x)

@njit(fastmath=True, parallel=True)
def fk_factor_numba(x, c=np.log(10), tol=1e-2):
    ret = np.zeros_like(x)

    for i in prange(x.shape[0]):
        if x[i] <= tol:
            ret[i] = c - x[i] * c ** 2 / 2 # + x[i] * x[i] * c ** 3 / 6
        else:
            ret[i] = (1 - np.exp(-x[i] * c)) / x[i]

    return ret



class Half_Bilirubin_Multiset(_Model):
    n = 4
    name = 'Half-Bilirubin Multiset Model'

    def __init__(self, times=None, ST=None, wavelengths=None, aug_matrix=None):
        super(Half_Bilirubin_Multiset, self).__init__(times)

        self.species_names = np.array(list('ZEHD'), dtype=np.str)
        self.wavelengths = wavelengths
        self.ST = ST

        # path = r"C:\Users\Dominik\Documents\MUNI\Organic Photochemistry\Projects\2019-Bilirubin project\UV-VIS\QY measurement\Photodiode\new setup"
        path = r"C:\Users\dominik\Documents\Projects\Bilirubin\UV-Vis data"

        fname = path + r'\em sources.txt'
        data = np.loadtxt(fname, delimiter='\t', skiprows=1)

        self.I_330 = data[:, 1] / np.trapz(data[:, 1], x=self.wavelengths)
        self.I_375 = data[:, 2] / np.trapz(data[:, 2], x=self.wavelengths)
        self.I_400 = data[:, 3] / np.trapz(data[:, 3], x=self.wavelengths)
        self.I_450 = data[:, 4] / np.trapz(data[:, 4], x=self.wavelengths)
        self.I_480 = data[:, 5] / np.trapz(data[:, 5], x=self.wavelengths)

        data_led = np.loadtxt(path + r'\LED sources.txt', delimiter='\t', skiprows=1)

        self.LED_355 = data_led[:, 1] / np.trapz(data_led[:, 1], x=self.wavelengths)
        self.LED_375 = data_led[:, 2] / np.trapz(data_led[:, 2], x=self.wavelengths)
        self.LED_405 = data_led[:, 3] / np.trapz(data_led[:, 3], x=self.wavelengths)
        self.LED_420 = data_led[:, 4] / np.trapz(data_led[:, 4], x=self.wavelengths)
        self.LED_450 = data_led[:, 5] / np.trapz(data_led[:, 5], x=self.wavelengths)
        self.LED_470 = data_led[:, 6] / np.trapz(data_led[:, 6], x=self.wavelengths)
        self.LED_490 = data_led[:, 7] / np.trapz(data_led[:, 7], x=self.wavelengths)

        fname = path + r'\q rel cut.txt'
        data = np.loadtxt(fname, delimiter='\t', skiprows=1)
        self.Diode_q_rel = data[:, 1]

        self._overlap330 = np.trapz(self.Diode_q_rel * self.I_330, x=self.wavelengths)
        self._overlap375 = np.trapz(self.Diode_q_rel * self.I_375, x=self.wavelengths)
        self._overlap400 = np.trapz(self.Diode_q_rel * self.I_400, x=self.wavelengths)
        self._overlap450 = np.trapz(self.Diode_q_rel * self.I_450, x=self.wavelengths)
        self._overlap480 = np.trapz(self.Diode_q_rel * self.I_480, x=self.wavelengths)

        self.aug_matrix = aug_matrix

        self.description = ""

    def init_params(self):
        self.params = Parameters()
        # self.params.add('IZ', value=38.1e-6, min=0, max=np.inf, vary=False)  # molar photon flux in mol s-1
        # self.params.add('IE', value=37.7e-6, min=0, max=np.inf, vary=False)  # molar photon flux in mol s-1
        #
        # self.params.add('V', value=3e-3, min=0, max=np.inf, vary=False)  # volume in L
        # self.params.add('w_irr', value=400, min=0, max=np.inf, vary=False)  # irradiating wavelength

        self.params.add('c0Z330', value=4.74E-05, min=0, max=np.inf, vary=True)
        self.params.add('c0E330', value=3.72E-05, min=0, max=np.inf, vary=True)

        self.params.add('c0Z400', value=4.68E-05, min=0, max=np.inf, vary=True)
        self.params.add('c0E400', value=3.65E-05, min=0, max=np.inf, vary=True)

        self.params.add('c0Z480', value=4.78E-05, min=0, max=np.inf, vary=True)
        self.params.add('c0E480', value=3.66E-05, min=0, max=np.inf, vary=True)

        self.params.add('c0Z375', value=6.66E-05, min=0, max=np.inf, vary=True)
        self.params.add('c0Z450', value=6.57E-05, min=0, max=np.inf, vary=True)

        self.params.add('c0Z355LED', value=4.57E-05, min=0, max=np.inf, vary=True)
        self.params.add('q0Z355LED', value=6.95E-08, min=0, max=np.inf, vary=True)

        self.params.add('c0Z375LED', value=5.13E-05, min=0, max=np.inf, vary=True)
        self.params.add('q0Z375LED', value=7.41E-08, min=0, max=np.inf, vary=True)

        self.params.add('c0Z405LED', value=5.18E-05, min=0, max=np.inf, vary=True)
        self.params.add('q0Z405LED', value=1.11E-07, min=0, max=np.inf, vary=True)

        self.params.add('c0Z450LED', value=5.12E-05, min=0, max=np.inf, vary=True)
        self.params.add('q0Z450LED', value=4.36E-07, min=0, max=np.inf, vary=True)

        self.params.add('c0Z470LED', value=5.12E-05, min=0, max=np.inf, vary=True)
        self.params.add('q0Z470LED', value=7.10E-07, min=0, max=np.inf, vary=True)

        self.params.add('c0Z490LED', value=4.28E-05, min=0, max=np.inf, vary=True)
        self.params.add('q0Z490LED', value=1.78E-06, min=0, max=np.inf, vary=True)

        # self.params.add('c0Z490355LED', value=4.915e-05, min=0, max=np.inf, vary=True)
        # self.params.add('q0Z490355LED', value=4.903e-07, min=0, max=np.inf, vary=True)

        # amount of Z in the mixture, 1: only Z, 0:, only E
        self.params.add('xZ_Z', value=1, min=0, max=1, vary=False)
        self.params.add('xZ_E', value=0.055, min=0, max=1, vary=False)

        self.params.add('Phi_ZE', value=0.160581292, min=0, max=1)
        self.params.add('Phi_ZE_1', value=0.030771559, min=-1, max=1)
        self.params.add('Phi_ZE_2', value=0.193611305, min=-1, max=1)
        self.params.add('Phi_ZE_3', value=-0.057237478, min=-1, max=1)
        self.params.add('Phi_ZE_4', value=-0.31305038, min=-1, max=1)


        self.params.add('Phi_EZ', value=0.368744317, min=0, max=1)
        self.params.add('Phi_EZ_1', value=-0.085357809, min=-1, max=1)
        self.params.add('Phi_EZ_2', value=-0.125563008, min=-1, max=1)
        self.params.add('Phi_EZ_3', value=-0.257495082, min=-1, max=1)
        self.params.add('Phi_EZ_4', value=0.98493781, min=-1, max=1)


        self.params.add('Phi_ZHL', value=0.004081171, min=0, max=1, vary=True)
        self.params.add('Phi_ZHL_1', value=-9.21E-05, min=-0.01, max=0.01)
        self.params.add('Phi_ZHL_2', value=0.028672891, min=-0.01, max=0.01)
        self.params.add('Phi_ZHL_3', value=0.01579831, min=-0.01, max=0.01)
        self.params.add('Phi_ZHL_4', value=-0.059034159, min=-0.01, max=0.01)


        self.params.add('Phi_HLZ', value=0.006594641, min=0, max=1, vary=True)
        self.params.add('Phi_HLZ_1', value=0.081719941, min=-0.01, max=0.01)
        self.params.add('Phi_HLZ_2', value=0, min=-0.01, max=0.01)
        self.params.add('Phi_HLZ_3', value=-0.006820378, min=-0.01, max=0.01)
        self.params.add('Phi_HLZ_4', value=-0.995929445, min=-0.01, max=0.01)


        # HL decay QY
        self.params.add('Phi_HL', value=0.003677229, min=0, max=1, vary=True)
        self.params.add('Phi_HL_1', value=-0.011269426, min=-0.01, max=0.01)
        self.params.add('Phi_HL_2', value=0.046829783,  min=-0.01, max=0.01)
        self.params.add('Phi_HL_3', value=0, min=-0.01, max=0.01)
        self.params.add('Phi_HL_4', value=-0.434018951, min=-0.01, max=0.01)



    @staticmethod
    def fk_factor(x, c=np.log(10), tol=1e-2):
        # exp(-xc) = 1 - xc + (xc)^2 / 2 - (xc)^3 / 6 ...
        # (1 - exp(-xc)) / x  ~  c - xc^2 / 2 for low x
        return np.where(x <= tol, c - x * c * c / 2 + x * x * c * c * c / 6, (1 - np.exp(-x * c)) / x)

    def plot_phis(self, y_scale='log'):
        if self.params is None:
            return

        c0Z330, c0E330, c0Z400, c0E400, c0Z480, c0E480, c0Z375, c0Z450, \
        c0Z355LED, q0Z355LED, c0Z375LED, q0Z375LED, c0Z405LED, q0Z405LED, c0Z450LED, q0Z450LED, c0Z470LED, q0Z470LED, c0Z490LED, q0Z490LED, \
        xZ_Z, xZ_E, \
        Phi_ZE, Phi_ZE1, Phi_ZE2, Phi_ZE3, Phi_ZE4, \
        Phi_EZ, Phi_EZ1, Phi_EZ2, Phi_EZ3, Phi_EZ4, \
        Phi_ZHL, Phi_ZHL1, Phi_ZHL2, Phi_ZHL3, Phi_ZHL4, \
        Phi_HLZ, Phi_HLZ1, Phi_HLZ2, Phi_HLZ3, Phi_HLZ4, \
        Phi_HL, Phi_HL1, Phi_HL2, Phi_HL3, Phi_HL4 = [par[1].value for par in self.params.items()]

        Phi_ZE = self.Phi([Phi_ZE, Phi_ZE1, Phi_ZE2, Phi_ZE3, Phi_ZE4], 400, self.wavelengths)
        Phi_EZ = self.Phi([Phi_EZ, Phi_EZ1, Phi_EZ2, Phi_EZ3, Phi_EZ4], 400, self.wavelengths)
        Phi_ZHL = self.Phi([Phi_ZHL, Phi_ZHL1, Phi_ZHL2, Phi_ZHL3, Phi_ZHL4], 400, self.wavelengths)
        Phi_HLZ = self.Phi([Phi_HLZ, Phi_HLZ1, Phi_HLZ2, Phi_HLZ3, Phi_HLZ4], 400, self.wavelengths)
        Phi_HL = self.Phi([Phi_HL, Phi_HL1, Phi_HL2, Phi_HL3, Phi_HL4], 400, self.wavelengths)

        plt.plot(self.wavelengths, Phi_ZE, label="$\Phi_{ZE}$")
        plt.plot(self.wavelengths, Phi_EZ, label="$\Phi_{EZ}$")
        plt.plot(self.wavelengths, Phi_ZHL, label="$\Phi_{ZHL}$")
        plt.plot(self.wavelengths, Phi_HLZ, label="$\Phi_{HLZ}$")
        plt.plot(self.wavelengths, Phi_HL, label="$\Phi_{HL-decay}$")

        plt.yscale(y_scale)

        plt.xlabel('Wavelength')
        plt.ylabel('$\Phi$')
        plt.legend()

        plt.show()



    @staticmethod
    def Phi(phis, wavelengths, lambda_C=400):
        assert isinstance(phis, (list, np.ndarray))
        return sum(par * ((lambda_C - wavelengths) / 100) ** i for i, par in enumerate(phis))

    @staticmethod
    def simulate(q0, c0, K, I_source, wavelengths=None, times=None, eps=None, V=0.003, l=1,  D=None):
        """
        c0 is concentration vector at time, defined in times arary as first element (initial condition), eps is vector of molar abs. coefficients,
        I_source is spectrum of irradiaiton source if this was used,
        if not, w_irr as irradiaton wavelength must be specified, K is transfer matrix, l is length of a cuvette, default 1 cm
        times are times for which to simulate the kinetics
        """
        # n = eps.shape[0]  # eps are epsilons - n x w matrix, where n is number of species and w is number of wavelengths
        # assert n == K.shape[0] == K.shape[1]
        c0 = np.asarray(c0)
        #
        # if I_source is None and w_irr is None:
        #     raise ValueError("Either specify I_source or irradiation wavelength w_irr!")

        # integrate = I_source is not None
        #
        # eps_w_irr = np.zeros(eps.shape[0])  # define epsilons only at irradiaton wavelength
        # if not integrate:
        #     w_idx = find_nearest_idx(wavelengths, w_irr)
        #     for i in range(n):
        #         eps_w_irr[i] = eps[i][w_idx]
        # else:
        #     I_source /= np.trapz(I_source, x=wavelengths)  # normalize irr source spectrum
            # K = np.transpose(K, (2, 0, 1))

        # ln10 = np.log(10)

        # get absorbances from real data
        abs_at = interp2d(wavelengths, times, D, kind='linear', copy=False)

        const = l * np.log(10)
        tol = 1e-3

        def dc_dt(c, t):
            # c_eps = c[..., None] * eps if integrate else (c * eps_w_irr)[..., None]  # hadamard product
            # c_dot_eps = c_eps.sum(axis=0)
            # # x_abs = c_eps * Half_Bilirubin_Multiset.fk_factor(c_dot_eps, c=l * ln10) * (I_source if integrate else 1)
            # x_abs = c_eps * fk_factor_numba(c_dot_eps, c=l * ln10) * (I_source if integrate else 1)
            # # w x n x n   x   w x n x 1
            # product = np.matmul(K, x_abs.T[..., None])  # w x n x 1
            # return q0 / V * (np.trapz(product, x=wavelengths, axis=0) if integrate else product).squeeze()

            c_eps = c[..., None] * eps  # hadamard product

            # c_dot_eps = c_eps.sum(axis=0)
            c_dot_eps = abs_at(wavelengths, t)

            # x_abs = c_eps * Half_Bilirubin_Multiset.fk_factor(c_dot_eps, c=l * ln10) * (I_source if integrate else 1)
            # x_abs = c_eps * fk_factor_numba(c_dot_eps, c=l * ln10) * I_source

            x_abs = c_eps * np.where(c_dot_eps <= tol, const - c_dot_eps * const * const / 2,
                                     (1 - np.exp(-c_dot_eps * const)) / c_dot_eps) * I_source

            # w x n x n   x   w x n x 1
            product = np.matmul(K, x_abs.T[..., None])  # w x n x 1

            return q0 / V * np.trapz(product, x=wavelengths, axis=0).squeeze()

        result = odeint(dc_dt, c0, times)

        return result

    def calc_C(self, params=None, C_out=None):
        super(Half_Bilirubin_Multiset, self).calc_C(params)

        if self.ST is None:
            raise ValueError("Spectra matrix must not be none.")

        # c0Z330, c0E330, c0Z400, c0E400, c0Z480, c0E480, xZ_Z, xZ_E, Phi_ZE, Phi_EZ, Phi_EHL, Phi_ZHL, Phi_HLE, Phi_ZE_1, Phi_EZ_1, Phi_EHL_1 = [par[1].value for par in self.params.items()]
        c0Z330, c0E330, c0Z400, c0E400, c0Z480, c0E480, c0Z375, c0Z450, \
        c0Z355LED, q0Z355LED, c0Z375LED, q0Z375LED, c0Z405LED, q0Z405LED, c0Z450LED, q0Z450LED, c0Z470LED, q0Z470LED, c0Z490LED, q0Z490LED, \
        xZ_Z, xZ_E, \
        Phi_ZE, Phi_ZE1, Phi_ZE2, Phi_ZE3, Phi_ZE4, \
        Phi_EZ, Phi_EZ1, Phi_EZ2, Phi_EZ3, Phi_EZ4, \
        Phi_ZHL, Phi_ZHL1, Phi_ZHL2, Phi_ZHL3, Phi_ZHL4, \
        Phi_HLZ, Phi_HLZ1, Phi_HLZ2, Phi_HLZ3, Phi_HLZ4, \
        Phi_HL, Phi_HL1, Phi_HL2, Phi_HL3, Phi_HL4 = [par[1].value for par in self.params.items()]

        Phi_ZE = self.Phi([Phi_ZE, Phi_ZE1, Phi_ZE2, Phi_ZE3, Phi_ZE4], 400, self.wavelengths)
        Phi_EZ = self.Phi([Phi_EZ, Phi_EZ1, Phi_EZ2, Phi_EZ3, Phi_EZ4], 400, self.wavelengths)
        Phi_ZHL = self.Phi([Phi_ZHL, Phi_ZHL1, Phi_ZHL2, Phi_ZHL3, Phi_ZHL4], 400, self.wavelengths)
        Phi_HLZ = self.Phi([Phi_HLZ, Phi_HLZ1, Phi_HLZ2, Phi_HLZ3, Phi_HLZ4], 400, self.wavelengths)
        Phi_HL = self.Phi([Phi_HL, Phi_HL1, Phi_HL2, Phi_HL3, Phi_HL4], 400, self.wavelengths)


        IZ330, IE330, IZ400, IE400, V = 16.7e-6, 17e-6, 38.1e-6, 37.7e-6, 3e-3
        IZ375, IZ450 = 24.9e-6, 47.9e-6
        IZ480, IE480 = 48e-6, 48e-6

        # crop QYs to 0-1
        Phi_ZE *= (Phi_ZE > 0)
        Phi_EZ *= (Phi_EZ > 0)
        Phi_ZHL *= (Phi_ZHL > 0)
        Phi_HLZ *= (Phi_HLZ > 0)
        Phi_HL *= (Phi_HL > 0)

        Phi_ZE = np.where(Phi_ZE > 1, 1, Phi_ZE)
        Phi_EZ = np.where(Phi_EZ > 1, 1, Phi_EZ)
        Phi_ZHL = np.where(Phi_ZHL > 1, 1, Phi_ZHL)
        Phi_HLZ = np.where(Phi_HLZ > 1, 1, Phi_HLZ)
        Phi_HL = np.where(Phi_HL > 1, 1, Phi_HL)

        # Phi_HLE = self.Phi([Phi_HLE], 400, self.wavelengths)

        _0 = np.zeros(self.wavelengths.shape) if isinstance(Phi_ZE, np.ndarray) else 0

        # K = np.asarray([[-Phi_ZE - Phi_ZHL,   Phi_EZ,        Phi_HLZ],
        #                 [Phi_ZE,              -Phi_EZ,            _0],
        #                 [Phi_ZHL,             _0,          -Phi_HLZ]])

        K = np.asarray([[-Phi_ZE - Phi_ZHL, Phi_EZ, Phi_HLZ, _0],
                        [Phi_ZE, -Phi_EZ, _0, _0],
                        [Phi_ZHL, _0, -Phi_HLZ - Phi_HL, _0],
                        [_0, _0, Phi_HL, _0]])

        # # # no photoproduct
        # K_no_D = np.asarray([[-Phi_ZE - Phi_ZHL, Phi_EZ, Phi_HLZ, _0],
        #                 [Phi_ZE, -Phi_EZ, _0, _0],
        #                 [Phi_ZHL, _0, -Phi_HLZ, _0],
        #                 [_0, _0, _0, _0]])
        #
        # K_no_D = np.transpose(K_no_D, (2, 0, 1))
        K = np.transpose(K, (2, 0, 1))


        q_tot_Z330, q_tot_E330 = IZ330 * self._overlap330, IE330 * self._overlap330
        q_tot_Z400, q_tot_E400 = IZ400 * self._overlap400, IE400 * self._overlap400
        q_tot_Z480, q_tot_E480 = IZ480 * self._overlap480, IE480 * self._overlap480
        q_tot_Z375, q_tot_Z450 = IZ375 * self._overlap375, IZ450 * self._overlap450


        # pool = Pool(processes=6)

        # N = 8
        # # pools = []
        # i = 0

        # [q0, c0, K, irr source],

        args = [
            [q_tot_Z330, [xZ_Z * c0Z330, (1 - xZ_Z) * c0Z330, 0, 0], K, self.I_330],  #Z 330
            [q_tot_E330, [xZ_E * c0E330, (1 - xZ_E) * c0E330, 0, 0], K, self.I_330],  # E start
            [q_tot_Z375, [xZ_Z * c0Z375, (1 - xZ_Z) * c0Z375, 0, 0], K, self.I_375],
            [q_tot_Z400, [xZ_Z * c0Z400, (1 - xZ_Z) * c0Z400, 0, 0], K, self.I_400],
            [q_tot_E400, [xZ_E * c0E400, (1 - xZ_E) * c0E400, 0, 0], K, self.I_400],  # E start
            [q_tot_Z450, [xZ_Z * c0Z450, (1 - xZ_Z) * c0Z450, 0, 0], K, self.I_450],
            [q_tot_Z480, [xZ_Z * c0Z480, (1 - xZ_Z) * c0Z480, 0, 0], K, self.I_480],
            [q_tot_E480, [xZ_E * c0E480, (1 - xZ_E) * c0E480, 0, 0], K, self.I_480],  # E start

            [q0Z355LED, [xZ_Z * c0Z355LED, (1 - xZ_Z) * c0Z355LED, 0, 0], K, self.LED_355],  # LED 355
            [q0Z375LED, [xZ_Z * c0Z375LED, (1 - xZ_Z) * c0Z375LED, 0, 0], K, self.LED_375],  # LED 375
            [q0Z405LED, [xZ_Z * c0Z405LED, (1 - xZ_Z) * c0Z405LED, 0, 0], K, self.LED_405],  # LED 405
            [q0Z450LED, [xZ_Z * c0Z450LED, (1 - xZ_Z) * c0Z450LED, 0, 0], K, self.LED_450],  # LED 450
            [q0Z470LED, [xZ_Z * c0Z470LED, (1 - xZ_Z) * c0Z470LED, 0, 0], K, self.LED_470],  # LED 470
            [q0Z490LED, [xZ_Z * c0Z490LED, (1 - xZ_Z) * c0Z490LED, 0, 0], K, self.LED_490],  # LED 470

        ]

        for i in range(len(args)):
            s, e = self.aug_matrix._C_indiv_range(i)
            t = self.aug_matrix.matrices[i, 0].times

            C_out[s:e, :] = self.simulate(*args[i], wavelengths=self.wavelengths,
                                          times=t, eps=self.ST, V=V, l=1, D=self.aug_matrix.matrices[i, 0].Y)

        i = len(args)
        #
        # # apply closure constrain on 490 nm LED C profiles,  keep it as a fitting parameter
        # s, e = self.aug_matrix._C_indiv_range(i)
        # C_out[s:e, 3] = 0  # no protoproducts
        # C_out[s:e, :] = c0Z490LED * C_out[s:e, :] / C_out[s:e, :].sum(axis=1, keepdims=True)


        # 355 nm LED
        # i += 1

        s, e = self.aug_matrix._C_indiv_range(i)
        c0_490_355 = C_out[s-1, :]  # concentrations at the end of 490 nm irr are the start of 355 nm irr

        t = self.aug_matrix.matrices[i, 0].times

        C_out[s:e, :] = self.simulate(q0Z355LED, c0_490_355, K, self.LED_355,
                                      wavelengths=self.wavelengths, times=t, eps=self.ST, V=V, l=1,
                                      D=self.aug_matrix.matrices[i, 0].Y)

        return C_out

        # # s, e = self.aug_matrix._C_indiv_range(i)
        # t = self.aug_matrix.matrices[i, 0].times
        #
        # pools.append(pool.apply_async(self.simulate, args=(q_tot_Z330, V, [xZ_Z * c0Z330, (1-xZ_Z)*c0Z330, 0], self.ST, K, t, self.wavelengths),
        #                               kwds={'I_source': self.I_330}))
        #
        # # self.C[s:e, :] = self.simulate(q_tot_Z330, V, [xZ_Z * c0Z330, (1-xZ_Z)*c0Z330, 0], self.ST, K, t, self.wavelengths, l=1,
        # #                        I_source=self.I_330, w_irr=None)
        #
        # # E 330
        # i += 1
        #
        # # s, e = self.aug_matrix._C_indiv_range(i)
        # t = self.aug_matrix.matrices[i, 0].times
        # pools.append(pool.apply_async(self.simulate, args=(q_tot_E330, V, [xZ_E * c0E330, (1 - xZ_E) * c0E330, 0], self.ST, K, t, self.wavelengths),
        #                               kwds={'I_source': self.I_330}))
        #
        # # self.C[s:e, :] = self.simulate(q_tot_E330, V, [xZ_E * c0E330, (1 - xZ_E) * c0E330, 0], self.ST, K, t,
        # #                                self.wavelengths, l=1, I_source=self.I_330, w_irr=None)
        #
        #
        # # Z 375
        #
        # i += 1
        #
        # # s, e = self.aug_matrix._C_indiv_range(i)
        # t = self.aug_matrix.matrices[i, 0].times
        #
        # pools.append(pool.apply_async(self.simulate, args=(q_tot_Z375, V, [xZ_Z * c0Z375, (1 - xZ_Z) * c0Z375, 0], self.ST, K, t, self.wavelengths),
        #                               kwds={'I_source': self.I_375}))
        #
        # # self.C[s:e, :] = self.simulate(q_tot_Z375, V, [xZ_Z * c0Z375, (1 - xZ_Z) * c0Z375, 0], self.ST, K, t,
        # #                                self.wavelengths, l=1, I_source=self.I_375, w_irr=None)
        #
        #
        # # Z 400
        #
        # i += 1
        #
        # # s, e = self.aug_matrix._C_indiv_range(i)
        # t = self.aug_matrix.matrices[i, 0].times
        #
        # pools.append(pool.apply_async(self.simulate, args=(q_tot_Z400, V, [xZ_Z * c0Z400, (1 - xZ_Z) * c0Z400, 0], self.ST, K, t, self.wavelengths),
        #                              kwds={'I_source': self.I_400}))
        #
        # # self.C[s:e, :] = self.simulate(q_tot_Z400, V, [xZ_Z * c0Z400, (1 - xZ_Z) * c0Z400, 0], self.ST, K, t,
        # #                                self.wavelengths, l=1, I_source=self.I_400, w_irr=None)
        #
        # # E 400
        #
        # i += 1
        #
        # # s, e = self.aug_matrix._C_indiv_range(i)
        # t = self.aug_matrix.matrices[i, 0].times
        #
        # pools.append(pool.apply_async(self.simulate, args=(q_tot_E400, V, [xZ_E * c0E400, (1 - xZ_E) * c0E400, 0], self.ST, K, t, self.wavelengths),
        #                               kwds={'I_source': self.I_400}))
        #
        # # self.C[s:e, :] = self.simulate(q_tot_E400, V, [xZ_E * c0E400, (1 - xZ_E) * c0E400, 0], self.ST, K, t,
        # #                                self.wavelengths, l=1, I_source=self.I_400, w_irr=None)
        #
        # # Z 450
        #
        # i += 1
        #
        # # s, e = self.aug_matrix._C_indiv_range(i)
        # t = self.aug_matrix.matrices[i, 0].times
        #
        # pools.append(pool.apply_async(self.simulate, args=(q_tot_Z450, V, [xZ_Z * c0Z450, (1 - xZ_Z) * c0Z450, 0], self.ST, K, t, self.wavelengths),
        #                               kwds={'I_source': self.I_450}))
        # #
        # # self.C[s:e, :] = self.simulate(q_tot_Z450, V, [xZ_Z * c0Z450, (1 - xZ_Z) * c0Z450, 0], self.ST, K, t,
        # #                                self.wavelengths, l=1, I_source=self.I_450, w_irr=None)
        #
        # # Z 480
        #
        # i += 1
        #
        # # s, e = self.aug_matrix._C_indiv_range(i)
        # t = self.aug_matrix.matrices[i, 0].times
        #
        # pools.append(pool.apply_async(self.simulate, args=(q_tot_Z480, V, [xZ_Z * c0Z480, (1 - xZ_Z) * c0Z480, 0], self.ST, K, t, self.wavelengths),
        #                               kwds={'I_source': self.I_480}))
        #
        # # self.C[s:e, :] = self.simulate(q_tot_Z480, V, [xZ_Z * c0Z480, (1 - xZ_Z) * c0Z480, 0], self.ST, K, t,
        # #                                self.wavelengths, l=1, I_source=self.I_480, w_irr=None)
        #
        # # E 480
        #
        # i += 1
        #
        # # s, e = self.aug_matrix._C_indiv_range(i)
        # t = self.aug_matrix.matrices[i, 0].times
        #
        # pools.append(pool.apply_async(self.simulate, args=(q_tot_E480, V, [xZ_E * c0E480, (1 - xZ_E) * c0E480, 0], self.ST, K, t, self.wavelengths),
        #                               kwds={'I_source': self.I_480}))
        #
        # # self.C[s:e, :] = self.simulate(q_tot_E480, V, [xZ_E * c0E480, (1 - xZ_E) * c0E480, 0], self.ST, K, t,
        # #                                self.wavelengths, l=1, I_source=self.I_480, w_irr=None)
        #
        #
        # for i in range(N):
        #     s, e = self.aug_matrix._C_indiv_range(i)
        #     self.C[s:e, :] = pools[i].get()
        #


        # # Z 490 Reactor
        #
        # i += 1
        #
        # s, e = self.aug_matrix._C_indiv_range(i)
        # t = self.aug_matrix.matrices[i, 0].times
        #
        # self.C[s:e, :] = self.simulate(q0Z490LED, V, [xZ_Z * c0Z490LED, 0, 0], self.ST, K, t,
        #                                self.wavelengths, l=1, I_source=self.LED_490, w_irr=None)

        # return self.get_conc_matrix(C_out, self._connectivity)








class Test_Bilirubin_Multiset(_Model):
    n = 3
    name = 'Test-Bilirubin Multiset Model'

    def __init__(self, times=None, ST=None, wavelengths=None, aug_matrix=None):
        super(Test_Bilirubin_Multiset, self).__init__(times)

        self.species_names = np.array(list('ZEHD'), dtype=np.str)
        self.wavelengths = wavelengths
        self.ST = ST

        # path = r"C:\Users\Dominik\Documents\MUNI\Organic Photochemistry\Projects\2019-Bilirubin project\UV-VIS\QY measurement\Photodiode\new setup"
        path = r"C:\Users\dominik\Documents\Projects\Bilirubin\UV-Vis data"

        fname = path + r'\em sources.txt'
        data = np.loadtxt(fname, delimiter='\t', skiprows=1)

        self.I_330 = data[:, 1] / np.trapz(data[:, 1], x=self.wavelengths)
        self.I_375 = data[:, 2] / np.trapz(data[:, 2], x=self.wavelengths)
        self.I_400 = data[:, 3] / np.trapz(data[:, 3], x=self.wavelengths)
        self.I_450 = data[:, 4] / np.trapz(data[:, 4], x=self.wavelengths)
        self.I_480 = data[:, 5] / np.trapz(data[:, 5], x=self.wavelengths)

        data_led = np.loadtxt(path + r'\LED sources.txt', delimiter='\t', skiprows=1)

        self.LED_355 = data_led[:, 1] / np.trapz(data_led[:, 1], x=self.wavelengths)
        self.LED_375 = data_led[:, 2] / np.trapz(data_led[:, 2], x=self.wavelengths)
        self.LED_405 = data_led[:, 3] / np.trapz(data_led[:, 3], x=self.wavelengths)
        self.LED_420 = data_led[:, 4] / np.trapz(data_led[:, 4], x=self.wavelengths)
        self.LED_450 = data_led[:, 5] / np.trapz(data_led[:, 5], x=self.wavelengths)
        self.LED_470 = data_led[:, 6] / np.trapz(data_led[:, 6], x=self.wavelengths)
        self.LED_490 = data_led[:, 7] / np.trapz(data_led[:, 7], x=self.wavelengths)

        fname = path + r'\q rel cut.txt'
        data = np.loadtxt(fname, delimiter='\t', skiprows=1)
        self.Diode_q_rel = data[:, 1]

        self._overlap330 = np.trapz(self.Diode_q_rel * self.I_330, x=self.wavelengths)
        self._overlap375 = np.trapz(self.Diode_q_rel * self.I_375, x=self.wavelengths)
        self._overlap400 = np.trapz(self.Diode_q_rel * self.I_400, x=self.wavelengths)
        self._overlap450 = np.trapz(self.Diode_q_rel * self.I_450, x=self.wavelengths)
        self._overlap480 = np.trapz(self.Diode_q_rel * self.I_480, x=self.wavelengths)

        self.aug_matrix = aug_matrix

        self.description = ""

    def init_params(self):
        self.params = Parameters()
        # self.params.add('IZ', value=38.1e-6, min=0, max=np.inf, vary=False)  # molar photon flux in mol s-1
        # self.params.add('IE', value=37.7e-6, min=0, max=np.inf, vary=False)  # molar photon flux in mol s-1
        #
        # self.params.add('V', value=3e-3, min=0, max=np.inf, vary=False)  # volume in L
        # self.params.add('w_irr', value=400, min=0, max=np.inf, vary=False)  # irradiating wavelength

        self.params.add('c0Z355', value=5e-5, min=0, max=np.inf, vary=True)
        self.params.add('q0Z355', value=5e-8, min=0, max=np.inf, vary=False)

        self.params.add('c0E355', value=2e-5, min=0, max=np.inf, vary=True)
        self.params.add('q0E355', value=5e-8, min=0, max=np.inf, vary=False)

        self.params.add('c0Z470', value=4e-5, min=0, max=np.inf, vary=True)
        self.params.add('q0Z470', value=2e-6, min=0, max=np.inf, vary=False)


        self.params.add('Phi_ZE', value=0.2, min=0, max=1)
        # self.params.add('Phi_ZE_1', value=0.012854019, min=-1, max=1)
        # self.params.add('Phi_ZE_2', value=-0.08318382, min=-1, max=1)
        # self.params.add('Phi_ZE_3', value=-0.117221184, min=-1, max=1)
        # self.params.add('Phi_ZE_4', value=-0.117221184, min=-1, max=1)


        self.params.add('Phi_EZ', value=0.25, min=0, max=1)
        # self.params.add('Phi_EZ_1', value=0.147305, min=-1, max=1)
        # self.params.add('Phi_EZ_2', value=0.030, min=-1, max=1)
        # self.params.add('Phi_EZ_3', value=-0.32, min=-1, max=1)
        # self.params.add('Phi_EZ_4', value=-0.32, min=-1, max=1)


        self.params.add('Phi_ZHL', value=0.005, min=0, max=1, vary=True)
        # self.params.add('Phi_ZHL_1', value=0.002772, min=-1, max=1)
        # self.params.add('Phi_ZHL_2', value=0.015506, min=-1, max=1)
        # self.params.add('Phi_ZHL_3', value=-0.018, min=-1, max=1)
        # self.params.add('Phi_ZHL_4', value=-0.018, min=-1, max=1)


        self.params.add('Phi_HLZ', value=0.001, min=0, max=1, vary=True)
        # self.params.add('Phi_HLZ_1', value=0.006, min=-1, max=1)
        # self.params.add('Phi_HLZ_2', value=0.0147815, min=-1, max=1)
        # self.params.add('Phi_HLZ_3', value=-0.053029012, min=-1, max=1)
        # self.params.add('Phi_HLZ_4', value=-0.053029012, min=-1, max=1)


        # # HL decay QY
        # self.params.add('Phi_HL', value=0.0001742, min=0, max=1, vary=True)
        # self.params.add('Phi_HL_1', value=0.0008615, min=-1, max=1)
        # self.params.add('Phi_HL_2', value=0.001029,  min=-1, max=1)
        # self.params.add('Phi_HL_3', value=-0.001198, min=-1, max=1)
        # self.params.add('Phi_HL_4', value=-0.001198, min=-1, max=1)



    @staticmethod
    def fk_factor(x, c=np.log(10), tol=1e-2):
        # exp(-xc) = 1 - xc + (xc)^2 / 2 - (xc)^3 / 6 ...
        # (1 - exp(-xc)) / x  ~  c - xc^2 / 2 for low x
        return np.where(x <= tol, c - x * c * c / 2 + x * x * c * c * c / 6, (1 - np.exp(-x * c)) / x)

    def plot_phis(self, y_scale='log'):
        if self.params is None:
            return

        c0Z355, q0Z355, c0E355, q0E355, c0Z470, q0Z470, Phi_ZE, Phi_EZ, Phi_ZHL, Phi_HLZ = [par[1].value for par in self.params.items()]
        # Phi_ZE, \ #Phi_ZE1, Phi_ZE2, Phi_ZE3, Phi_ZE4, \
        # Phi_EZ, \ #Phi_EZ1, Phi_EZ2, Phi_EZ3, Phi_EZ4, \
        # Phi_ZHL, \ #Phi_ZHL1, Phi_ZHL2, Phi_ZHL3, Phi_ZHL4, \
        # Phi_HLZ = [par[1].value for par in self.params.items()] #Phi_HLZ1, Phi_HLZ2, Phi_HLZ3, Phi_HLZ4, \
        # # Phi_HL, Phi_HL1, Phi_HL2, Phi_HL3, Phi_HL4 = [par[1].value for par in self.params.items()]

        # Phi_ZE = self.Phi([Phi_ZE, Phi_ZE1, Phi_ZE2, Phi_ZE3, Phi_ZE4], 400, self.wavelengths)
        # Phi_EZ = self.Phi([Phi_EZ, Phi_EZ1, Phi_EZ2, Phi_EZ3, Phi_EZ4], 400, self.wavelengths)
        # Phi_ZHL = self.Phi([Phi_ZHL, Phi_ZHL1, Phi_ZHL2, Phi_ZHL3, Phi_ZHL4], 400, self.wavelengths)
        # Phi_HLZ = self.Phi([Phi_HLZ, Phi_HLZ1, Phi_HLZ2, Phi_HLZ3, Phi_HLZ4], 400, self.wavelengths)
        # # Phi_HL = self.Phi([Phi_HL, Phi_HL1, Phi_HL2, Phi_HL3, Phi_HL4], 400, self.wavelengths)

        Phi_ZE = self.Phi([Phi_ZE], 400, self.wavelengths)
        Phi_EZ = self.Phi([Phi_EZ], 400, self.wavelengths)
        Phi_ZHL = self.Phi([Phi_ZHL], 400, self.wavelengths)
        Phi_HLZ = self.Phi([Phi_HLZ], 400, self.wavelengths)
        # Phi_HL = self.Phi([Phi_HL, Phi_HL1, Phi_HL2, Phi_HL3, Phi_HL4], 400, self.wavelengths)

        plt.plot(self.wavelengths, Phi_ZE, label="$\Phi_{ZE}$")
        plt.plot(self.wavelengths, Phi_EZ, label="$\Phi_{EZ}$")
        plt.plot(self.wavelengths, Phi_ZHL, label="$\Phi_{ZHL}$")
        plt.plot(self.wavelengths, Phi_HLZ, label="$\Phi_{HLZ}$")
        # plt.plot(self.wavelengths, Phi_HL, label="$\Phi_{HL-decay}$")

        plt.yscale(y_scale)

        plt.xlabel('Wavelength')
        plt.ylabel('$\Phi$')
        plt.legend()

        plt.show()



    @staticmethod
    def Phi(phis, wavelengths, lambda_C=400):
        assert isinstance(phis, (list, np.ndarray))
        return sum(par * ((lambda_C - wavelengths) / 100) ** i for i, par in enumerate(phis))

    @staticmethod
    def simulate(q0, c0, K, I_source, wavelengths=None, times=None, eps=None, V=0.003, l=1,  D=None):
        """
        c0 is concentration vector at time, defined in times arary as first element (initial condition), eps is vector of molar abs. coefficients,
        I_source is spectrum of irradiaiton source if this was used,
        if not, w_irr as irradiaton wavelength must be specified, K is transfer matrix, l is length of a cuvette, default 1 cm
        times are times for which to simulate the kinetics
        """
        # n = eps.shape[0]  # eps are epsilons - n x w matrix, where n is number of species and w is number of wavelengths
        # assert n == K.shape[0] == K.shape[1]
        c0 = np.asarray(c0)

        # get absorbances from real data
        # abs_at = interp2d(wavelengths, times, D, kind='linear', copy=False)

        const = l * np.log(10)
        tol = 1e-3

        def dc_dt(c, t):

            c_eps = c[..., None] * eps  # hadamard product

            c_dot_eps = c_eps.sum(axis=0)
            # c_dot_eps = abs_at(wavelengths, t)

            x_abs = c_eps * np.where(c_dot_eps <= tol, const - c_dot_eps * const * const / 2,
                                     (1 - np.exp(-c_dot_eps * const)) / c_dot_eps) * I_source

            # w x n x n   x   w x n x 1
            product = np.matmul(K, x_abs.T[..., None])  # w x n x 1

            return q0 / V * np.trapz(product, x=wavelengths, axis=0).squeeze()

        result = odeint(dc_dt, c0, times)

        return result

    def calc_C(self, params=None, C_out=None):
        super(Test_Bilirubin_Multiset, self).calc_C(params)

        if self.ST is None:
            raise ValueError("Spectra matrix must not be none.")

        # c0Z330, c0E330, c0Z400, c0E400, c0Z480, c0E480, c0Z375, c0Z450, \
        # c0Z355LED, q0Z355LED, c0Z375LED, q0Z375LED, c0Z405LED, q0Z405LED, c0Z450LED, q0Z450LED, c0Z470LED, q0Z470LED, c0Z490LED, q0Z490LED, \
        # xZ_Z, xZ_E, \
        # Phi_ZE, Phi_ZE1, Phi_ZE2, Phi_ZE3, Phi_ZE4, \
        # Phi_EZ, Phi_EZ1, Phi_EZ2, Phi_EZ3, Phi_EZ4, \
        # Phi_ZHL, Phi_ZHL1, Phi_ZHL2, Phi_ZHL3, Phi_ZHL4, \
        # Phi_HLZ, Phi_HLZ1, Phi_HLZ2, Phi_HLZ3, Phi_HLZ4, \
        # Phi_HL, Phi_HL1, Phi_HL2, Phi_HL3, Phi_HL4 = [par[1].value for par in self.params.items()]

        c0Z355, q0Z355, c0E355, q0E355, c0Z470, q0Z470, Phi_ZE, Phi_EZ, Phi_ZHL, Phi_HLZ = [par[1].value for par in self.params.items()]
        V = 0.003

        Phi_ZE = self.Phi([Phi_ZE], 400, self.wavelengths)
        Phi_EZ = self.Phi([Phi_EZ], 400, self.wavelengths)
        Phi_ZHL = self.Phi([Phi_ZHL], 400, self.wavelengths)
        Phi_HLZ = self.Phi([Phi_HLZ], 400, self.wavelengths)

        # Phi_ZE = self.Phi([Phi_ZE, Phi_ZE1, Phi_ZE2, Phi_ZE3, Phi_ZE4], 400, self.wavelengths)
        # Phi_EZ = self.Phi([Phi_EZ, Phi_EZ1, Phi_EZ2, Phi_EZ3, Phi_EZ4], 400, self.wavelengths)
        # Phi_ZHL = self.Phi([Phi_ZHL, Phi_ZHL1, Phi_ZHL2, Phi_ZHL3, Phi_ZHL4], 400, self.wavelengths)
        # Phi_HLZ = self.Phi([Phi_HLZ, Phi_HLZ1, Phi_HLZ2, Phi_HLZ3, Phi_HLZ4], 400, self.wavelengths)
        # Phi_HL = self.Phi([Phi_HL, Phi_HL1, Phi_HL2, Phi_HL3, Phi_HL4], 400, self.wavelengths)

        #
        # # crop QYs to 0-1
        # Phi_ZE *= (Phi_ZE > 0)
        # Phi_EZ *= (Phi_EZ > 0)
        # Phi_ZHL *= (Phi_ZHL > 0)
        # Phi_HLZ *= (Phi_HLZ > 0)
        # Phi_HL *= (Phi_HL > 0)

        # Phi_ZE = np.where(Phi_ZE > 1, 1, Phi_ZE)
        # Phi_EZ = np.where(Phi_EZ > 1, 1, Phi_EZ)
        # Phi_ZHL = np.where(Phi_ZHL > 1, 1, Phi_ZHL)
        # Phi_HLZ = np.where(Phi_HLZ > 1, 1, Phi_HLZ)
        # Phi_HL = np.where(Phi_HL > 1, 1, Phi_HL)

        # Phi_HLE = self.Phi([Phi_HLE], 400, self.wavelengths)

        _0 = np.zeros(self.wavelengths.shape) if isinstance(Phi_ZE, np.ndarray) else 0

        K = np.asarray([[-Phi_ZE - Phi_ZHL,   Phi_EZ,        Phi_HLZ],
                        [Phi_ZE,              -Phi_EZ,            _0],
                        [Phi_ZHL,             _0,          -Phi_HLZ]])

        K = np.transpose(K, (2, 0, 1))


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
            t = self.aug_matrix.matrices[i, 0].times

            C_out[s:e, :] = self.simulate(*args[i], wavelengths=self.wavelengths,
                                          times=t, eps=self.ST, V=V, l=1, D=self.aug_matrix.matrices[i, 0].Y)
        #
        # i = len(args)
        # #
        # # # apply closure constrain on 490 nm LED C profiles,  keep it as a fitting parameter
        # # s, e = self.aug_matrix._C_indiv_range(i)
        # # C_out[s:e, 3] = 0  # no protoproducts
        # # C_out[s:e, :] = c0Z490LED * C_out[s:e, :] / C_out[s:e, :].sum(axis=1, keepdims=True)
        #
        #
        # # 355 nm LED
        # # i += 1
        #
        # s, e = self.aug_matrix._C_indiv_range(i)
        # c0_490_355 = C_out[s-1, :]  # concentrations at the end of 490 nm irr are the start of 355 nm irr
        #
        # t = self.aug_matrix.matrices[i, 0].times
        #
        # C_out[s:e, :] = self.simulate(q0Z355LED, c0_490_355, K, self.LED_355,
        #                               wavelengths=self.wavelengths, times=t, eps=self.ST, V=V, l=1,
        #                               D=self.aug_matrix.matrices[i, 0].Y)
        #

        return C_out








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
