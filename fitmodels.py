import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from lmfit import Parameters
from abc import abstractmethod
from gui_console import Console

import math


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
        assert n == K.shape[0] == K.shape[1]
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

        # K = np.asarray([[-Phi_ZE, Phi_EZ, 0],
        #                 [Phi_ZE, -Phi_EZ - Phi_EHL, 0],
        #                 [0, Phi_EHL, 0]])

        K = np.asarray([[-Phi_ZE, Phi_EZ, 0, 0],
                        [Phi_ZE, -Phi_EZ - Phi_EHL, 0, 0],
                        [0, Phi_EHL, - Phi_HLBl, 0],
                        [0, 0, Phi_HLBl, 0]])

        n = K.shape[0]

        x0 = np.linspace(0, self.times[0], num=10)
        _init_x = self.simulate(q0, V, [c0*xZ, (1-xZ)*c0, 0, 0], self.ST[:n, :], K, x0, self.wavelengths, l=1,
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










class Half_Bilirubin_Multiset(_Model):
    n = 3
    name = 'Half-Bilirubin Multiset Model'

    def __init__(self, times=None, ST=None, wavelengths=None, aug_matrix=None):
        super(Half_Bilirubin_Multiset, self).__init__(times)

        self.species_names = np.array(list('ZEH'), dtype=np.str)
        self.wavelengths = wavelengths
        self.ST = ST

        fname = r'C:\Users\dominik\Documents\Projects\Bilirubin\UV-Vis data\em sources.txt'
        data = np.loadtxt(fname, dtype=np.float64, delimiter='\t', skiprows=1)
        self.I_source = data[:, 3]

        fname = r'C:\Users\dominik\Documents\Projects\Bilirubin\UV-Vis data\q rel cut.txt'
        data = np.loadtxt(fname, dtype=np.float64, delimiter='\t', skiprows=1)
        self.Diode_q_rel = data[:, 1]

        self.aug_matrix = aug_matrix

        self.description = ""

    def init_params(self):
        self.params = Parameters()
        self.params.add('IZ', value=38.1e-6, min=0, max=np.inf, vary=False)  # molar photon flux in mol s-1
        self.params.add('IE', value=37.7e-6, min=0, max=np.inf, vary=False)  # molar photon flux in mol s-1

        self.params.add('V', value=3e-3, min=0, max=np.inf, vary=False)  # volume in mL
        self.params.add('w_irr', value=400, min=0, max=np.inf, vary=False)  # irradiating wavelength

        self.params.add('c0Z', value=2.90881898e-05, min=0, max=np.inf, vary=True)
        self.params.add('c0E', value=2.90881898e-05, min=0, max=np.inf, vary=True)

        # amount of Z in the mixture, 1: only Z, 0:, only E
        self.params.add('xZ_Z', value=1, min=0, max=1, vary=False)
        self.params.add('xZ_E', value=0.2, min=0, max=1, vary=False)

        self.params.add('Phi_ZE', value=0.25, min=0, max=1)
        self.params.add('Phi_EZ', value=0.23, min=0, max=1)
        self.params.add('Phi_EHL', value=0.01, min=0, max=1)
        self.params.add('Phi_ZHL', value=0.0, min=0, max=1, vary=False)
        self.params.add('Phi_HLE', value=0, min=0, max=1, vary=False)


    @staticmethod
    def fk_factor(x, c=np.log(10), tol=1e-2):
        # exp(-xc) = 1 - xc + (xc)^2 / 2 - (xc)^3 / 6 ...
        # (1 - exp(-xc)) / x  ~  c - xc^2 / 2 for low x
        return np.where(x <= tol, c - x * c * c / 2 + x * x * c * c * c / 6, (1 - np.exp(-x * c)) / x)

    @staticmethod
    def Phi(phis, wavelengths, lambda_C=400):
        assert isinstance(phis, (list, np.ndarray))
        return sum(par * ((lambda_C - wavelengths) / 100) ** i for i, par in enumerate(phis))

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

        integrate = I_source is not None

        eps_w_irr = np.zeros(eps.shape[0])  # define epsilons only at irradiaton wavelength
        if not integrate:
            w_idx = find_nearest_idx(wavelengths, w_irr)
            for i in range(n):
                eps_w_irr[i] = eps[i][w_idx]
        else:
            I_source /= np.trapz(I_source, x=wavelengths)  # normalize irr source spectrum
            K = np.transpose(K, (2, 0, 1))

        ln10 = np.log(10)

        def dc_dt(c, t):

            c_eps = c[..., None] * eps if integrate else (c * eps_w_irr)[..., None]  # hadamard product

            c_dot_eps = c_eps.sum(axis=0)

            x_abs = c_eps * Half_Bilirubin_Multiset.fk_factor(c_dot_eps, c=l * ln10) * (I_source if integrate else 1)

            # w x n x n   x   w x n x 1
            product = np.matmul(K, x_abs.T[..., None])  # w x n x 1

            return q0 / V * (np.trapz(product, x=wavelengths, axis=0) if integrate else product).squeeze()

        result = odeint(dc_dt, c0, times)

        return result

    def calc_C(self, params=None, C_out=None):
        super(Half_Bilirubin_Multiset, self).calc_C(params)

        if self.ST is None:
            raise ValueError("Spectra matrix must not be none.")

        IZ, IE, V, w_irr, c0Z, c0E, xZ_Z, xZ_E, Phi_ZE, Phi_EZ, Phi_EHL, Phi_ZHL, Phi_HLE  = [par[1].value for par in self.params.items()]

        Phi_ZE = self.Phi([Phi_ZE], 400, self.wavelengths)
        Phi_EZ = self.Phi([Phi_EZ], 400, self.wavelengths)
        Phi_EHL = self.Phi([Phi_EHL], 400, self.wavelengths)
        Phi_ZHL = self.Phi([Phi_ZHL], 400, self.wavelengths)
        Phi_HLE = self.Phi([Phi_HLE], 400, self.wavelengths)

        _0 = np.zeros(self.wavelengths.shape) if isinstance(Phi_ZE, np.ndarray) else 0

        K = np.asarray([[-Phi_ZE - Phi_ZHL, Phi_EZ, _0],
                        [Phi_ZE, -Phi_EZ - Phi_EHL, Phi_HLE],
                        [Phi_ZHL, Phi_EHL, -Phi_HLE]])

        # n = K.shape[0]

        _overlap = np.trapz(self.Diode_q_rel * self.I_source, x=self.wavelengths)

        q_tot_Z, q_tot_E = IZ * _overlap, IE * _overlap


        #Z

        Z_start, Z_end = self.aug_matrix._C_indiv_range(0)
        times_Z = self.aug_matrix.matrices[0, 0].times

        x0 = np.linspace(0, times_Z[0], num=10)

        _init_x = self.simulate(q_tot_Z, V, [xZ_Z * c0Z, (1-xZ_Z)*c0Z, 0], self.ST, K, x0, self.wavelengths, l=1,
                               I_source=self.I_source, w_irr=w_irr)[-1, :]

        self.C[Z_start:Z_end, :] = self.simulate(q_tot_Z, V, _init_x, self.ST, K, times_Z, self.wavelengths, l=1,
                               I_source=self.I_source, w_irr=w_irr)

        # E

        E_start, E_end = self.aug_matrix._C_indiv_range(1)
        times_E = self.aug_matrix.matrices[1, 0].times

        x0 = np.linspace(0, times_E[0], num=10)

        _init_x = self.simulate(q_tot_E, V, [xZ_E * c0E, (1 - xZ_E) * c0E, 0], self.ST, K, x0, self.wavelengths, l=1,
                                I_source=self.I_source, w_irr=w_irr)[-1, :]

        self.C[E_start:E_end, :] = self.simulate(q_tot_E, V, _init_x, self.ST, K, times_E, self.wavelengths, l=1,
                               I_source=self.I_source, w_irr=w_irr)

        # self.C = np.heaviside(self.times, 1)[..., None] * result

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
