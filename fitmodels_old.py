import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from lmfit import Parameters
from abc import abstractmethod


# virtual class that every model must inherits
class _Model(object):
    n = 0  # number of visible species in model
    n_full = 0  # number of all possible species, eg. for model ABC n_full = 3, but if only A and B are visible, n = 2

    _visible = None
    params = None
    species_names = None

    name = 'AB Model'
    description = "..."

    _err = 1e-8

    def __init__(self, times=None, visible=None):
        self.times = times
        self.C = None
        self.C_full = None

        if visible is None:
            self.init_visible()
        else:
            self.visible = visible

        self.init_params()
        self.species_names = np.array(list('AB'), dtype=np.str)

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, value):
        if not isinstance(value, (list, tuple)):
            raise ValueError("Visible array must be type of list or tuple.")
        self.n = 0
        for val in value:
            if val:
                self.n += 1
        if self.n == 0:
            raise ValueError('Visible array must contain at least one item True.')

        # n could have changed, so we have to init again
        self.init_times(self.times)
        self._visible = value

    def init_times(self, times):
        if times is not None:
            self.C = np.zeros((times.shape[0], self.n))
            self.C_full = np.zeros((times.shape[0], self.n_full))
            self.times = times

    def calc_C(self, params=None):
        if params is not None:
            self.params = params

    # virtual method
    def init_visible(self):
        self.visible = (self.n_full - 1) * [True] + [False]

    @abstractmethod
    def init_params(self):
        self.params = []

    def get_conc_matrix(self):
        """Selects from full concentration matrix only species that are visible"""
        j = 0
        for i in range(self.n_full):
            if self._visible[i]:
                self.C[:, j] = self.C_full[:, i]
                j += 1

        return self.C

    def get_species_name(self, i):
        # i is index in range(n)
        vis_species = self.species_names[self._visible]
        return vis_species[i]

    def cA(self, c0, k, n):
        if n == 1:  # first order
            return np.heaviside(self.times, 1) * c0 * np.exp(-self.times * k)
        elif n == 2:  # second order
            return np.heaviside(self.times, 1) * c0 / (1 + c0 * k * self.times)
        else:  # n-th order, this wont work for negative times c(t) = 1-n root of (c^(1-n) + k*(n-1)*t)
            expr_in_root = np.power(float(c0), 1 - n) + k * (n - 1) * self.times
            expr_in_root = expr_in_root.clip(min=0)  # set to 0 all the negative values
            return np.heaviside(self.times, 1) * np.power(expr_in_root, 1.0 / (1 - n))

    def cB(self, t, c0, k1, k2):
        if np.abs(k1 - k2) < self._err:
            return np.heaviside(t, 1) * c0 * k1 * t * np.exp(-k1 * t)
        else:
            return np.heaviside(t, 1) * (k1 * c0 / (k2 - k1)) * (np.exp(-k1 * t) - np.exp(-k2 * t))


class AB_Model(_Model):
    """Simple A->B model, default is 1st order reaction, 2nd order and n-th order can be selected as well. Also,
    user can define whether both A and B are visible, or only A or B is visible. Default is [True, False] - only A is
     visible"""

    order = '1st'
    n_full = 2
    name = 'A→B (variable order)'

    def __init__(self, times=None, visible=None, order='1st'):
        """order == '1st' - 1st order kinetics (default)
        order == '2nd' - 2ns order reaction kinetics
        order == 'n-th' - n-th order reaction kinetics

        For 1st order, c0=1 in params is set to fixed as default."""
        self.order = order

        super(AB_Model, self).__init__(times, visible)

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

    def calc_C(self, params=None):
        super(AB_Model, self).calc_C(params)

        c0, k, n = self.params['c0'].value, self.params['k'].value, self.params['n'].value

        self.C_full[:, 0] = self.cA(c0, k, n)  # A
        self.C_full[:, 1] = np.heaviside(self.times, 1) * (c0 - self.C_full[:, 0])  # B

        return self.get_conc_matrix()



class AB_mixed12_Model(_Model):
    """Mixed first and second order kinetics, d[A]/dt = -k1[A] - k2[A]^2"""
    n_full = 2
    name = 'A→B (mixed 1st and 2nd order)'

    def __init__(self, times=None, visible=None):
        super(AB_mixed12_Model, self).__init__(times, visible)

        self.species_names = np.array(list('AB'), dtype=np.str)

        self.description = "A->B model of mixed first and second order. d[A]/dt = -k1[A] - k2[A]^2, [A]_0 = c_0"

    def init_params(self):
        self.params = Parameters()
        self.params.add('c0', value=1, min=0, max=np.inf, vary=False)
        self.params.add('k1', value=1, min=0, max=np.inf)
        self.params.add('k2', value=0.2, min=0, max=np.inf)

    def calc_C(self, params=None):
        super(AB_mixed12_Model, self).calc_C(params)

        c0, k1, k2 = self.params['c0'].value, self.params['k1'].value, self.params['k2'].value

        self.C_full[:, 0] = np.heaviside(self.times, 1) * c0 * k1 / (
                (c0 * k2 + k1) * np.exp(k1 * self.times) - c0 * k2)  # A
        self.C_full[:, 1] = np.heaviside(self.times, 1) * (c0 - self.C_full[:, 0])  # B

        # cut matrix accordding to
        return self.get_conc_matrix()


class ABC_Model(_Model):
    """ABC kinetic model, first order."""

    n_full = 3
    name = 'A→B→C (1st order)'

    def __init__(self, times=None, visible=None):
        super(ABC_Model, self).__init__(times, visible)

        self.species_names = np.array(list('ABC'), dtype=np.str)

        self.description = "Simple A->B->C model of 1st order. d[A]/dt = -k1[A] - k2[A]^2, [A]_0 = c_0"

    def init_params(self):
        self.params = Parameters()
        self.params.add('c0', value=1, min=0, max=np.inf, vary=False)
        self.params.add('k1', value=1, min=0, max=np.inf)
        self.params.add('k2', value=0.5, min=0, max=np.inf)

    def calc_C(self, params=None):
        super(ABC_Model, self).calc_C(params)

        c0, k1, k2 = self.params['c0'].value, self.params['k1'].value, self.params['k2'].value

        self.C_full[:, 0] = np.heaviside(self.times, 1) * c0 * np.exp(-self.times * k1)
        self.C_full[:, 1] = self.cB(self.times, c0, k1, k2)
        self.C_full[:, 2] = np.heaviside(self.times, 1) * (c0 - self.C_full[:, 0] - self.C_full[:, 1])

        return self.get_conc_matrix()


class ABCD_Model(_Model):
    """ABCD kinetic model, first order."""

    n_full = 5
    name = 'A→B→C→D (1st order, Z0)'

    def __init__(self, times=None, visible=None):
        super(ABCD_Model, self).__init__(times, visible)

        self.species_names = np.array(list('ABCDZ'), dtype=np.str)

        # self.description = "Simple A->B->C model of 1st order. d[A]/dt = -k1[A] - k2[A]^2, [A]_0 = c_0"

    def init_params(self):
        self.params = Parameters()
        self.params.add('c0', value=1, min=0, max=np.inf, vary=False)
        self.params.add('k1', value=1, min=0, max=np.inf)
        self.params.add('k2', value=0.5, min=0, max=np.inf)
        self.params.add('k3', value=0.2, min=0, max=np.inf)

    def calc_C(self, params=None):
        super(ABCD_Model, self).calc_C(params)

        c0, k1, k2, k3 = self.params['c0'].value, self.params['k1'].value, self.params['k2'].value, self.params[
            'k3'].value

        self.C_full[:, 0] = np.heaviside(self.times, 1) * c0 * np.exp(-self.times * k1)
        self.C_full[:, 1] = self.cB(self.times, c0, k1, k2)

        def dC_dt(cC, t):
            cB = self.cB(t, c0, k1, k2)
            return k2 * cB - k3 * cC  # d[C]/dt = k2[B] - k3[C]

        # initial condition, cC(t=0) = 0
        result = odeint(dC_dt, 0, self.times)
        self.C_full[:, 2] = np.heaviside(self.times, 1) * result.flatten()
        self.C_full[:, 3] = np.heaviside(self.times, 1) * (
                c0 - self.C_full[:, 0] - self.C_full[:, 1] - self.C_full[:, 2])

        self.C_full[:, 4] = np.ones(self.times.shape[0])

        return self.get_conc_matrix()


class CP_Model(_Model):
    """ABCD kinetic model, first order."""

    n_full = 4
    name = 'CP fitting model ABCZ'

    def __init__(self, times=None, visible=None):
        super(CP_Model, self).__init__(times, visible)

        self.species_names = np.array(list('ABCZ'), dtype=np.str)

    def init_params(self):
        self.params = Parameters()
        self.params.add('c0', value=1, min=0, max=np.inf, vary=False)
        self.params.add('kTT', value=1, min=0, max=np.inf)  # BP 2nd order decay
        self.params.add('kBP', value=1, min=0, max=np.inf)  # BP 1st order decay
        self.params.add('k1', value=1, min=0, max=np.inf)  # kq[Q]
        self.params.add('kT1T0', value=0.5, min=0, max=np.inf)  # k(T1->T0)
        self.params.add('qT1', value=0.5, min=0, max=1)  # kb, ka + kb = k1
        self.params.add('kd', value=0.3, min=0, max=np.inf)  # kd
        # ka = pT1 * k1
        # kb = (1 - pT1) * k1

    def calc_C(self, params=None):
        super(CP_Model, self).calc_C(params)

        def cA(t, c0, k, kTT):
            return np.heaviside(t, 1) * c0 * k / (
                    (c0 * kTT + k) * np.exp(k * t) - c0 * kTT)  # A

        kTT = self.params['kTT'].value
        kBP = self.params['kBP'].value
        c0, k1, kT1T0, qT1, kd = self.params['c0'].value, self.params['k1'].value, self.params['kT1T0'].value, \
                                 self.params[
                                     'qT1'].value, self.params['kd'].value

        self.C_full[:, 0] = cA(self.times, c0, k1 + kBP, kTT)
        self.C_full[:, 1] = self.cB(self.times, c0, qT1 * k1, kT1T0)

        def dC_dt(cC, t):
            cB = self.cB(t, c0, qT1 * k1, kT1T0)
            return (1 - qT1) * k1 * cA(t, c0, k1 + kBP, kTT) + kT1T0 * cB - kd * cC  # d[C]/dt = k3[A] + k2[B] - k4[C]

        # initial condition, cC(t=0) = 0
        result = odeint(dC_dt, 0, self.times)
        self.C_full[:, 2] = np.heaviside(self.times, 1) * result.flatten()

        self.C_full[:, 3] = np.ones(self.times.shape[0])

        return self.get_conc_matrix()


class ABCDE_Model(_Model):
    """ABCDE kinetic model, first order."""

    n_full = 5
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

    def calc_C(self, params=None):
        super(ABCDE_Model, self).calc_C(params)

        c0, k1, k2, k3, k4 = self.params['c0'].value, self.params['k1'].value, self.params['k2'].value, \
                             self.params['k3'].value, self.params['k4'].value

        self.C_full[:, 0] = np.heaviside(self.times, 1) * c0 * np.exp(-self.times * k1)
        self.C_full[:, 1] = self.cB(self.times, c0, k1, k2)

        def solve(conc, t):
            cC, cD = conc
            cB = self.cB(t, c0, k1, k2)
            dC_dt = k2 * cB - k3 * cC  # d[C]/dt = k2[B] - k3[C]
            dD_dt = k3 * cC - k4 * cD  # d[D]/dt = k3[C] - k4[D]
            return [dC_dt, dD_dt]

        # initial conditiona, cC(t=0) = 0, cD(t=0) = 0
        result = odeint(solve, [0, 0], self.times)
        self.C_full[:, 2] = np.heaviside(self.times, 1) * result[:, 0]
        self.C_full[:, 3] = np.heaviside(self.times, 1) * result[:, 1]

        self.C_full[:, 4] = np.heaviside(self.times, 1) * (
                c0 - self.C_full[:, 0] - self.C_full[:, 1] - self.C_full[:, 2] - self.C_full[:, 3])

        return self.get_conc_matrix()


class Delayed_Fl(_Model):
    """Delayed fluorescence"""

    n_full = 4
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
        self.params.add('kqTT', value=0.5, min=0, max=np.inf)  # rate constant of 3BP and 3DR annihilation (mixed TT annihilation)
        self.params.add('kISC', value=0.4, min=0, max=np.inf)  # shifting the maximum of the emission maxima
        self.params.add('kd', value=0.3, min=0, max=np.inf)  # 1st order decay rate constant of DR

    def calc_C(self, params=None):
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

        self.C_full[:, 0] = np.heaviside(self.times, 1) * result[:, 0]
        self.C_full[:, 1] = np.heaviside(self.times, 1) * result[:, 1]
        self.C_full[:, 2] = np.heaviside(self.times, 1) * result[:, 2]
        self.C_full[:, 3] = np.ones(self.times.shape[0])
        #
        # self.C_full[:, 4] = np.heaviside(self.times, 1) * (
        #         c0 - self.C_full[:, 0] - self.C_full[:, 1] - self.C_full[:, 2] - self.C_full[:, 3])

        return self.get_conc_matrix()


class ABC_DEF_Model(_Model):
    """Two independent ABC kinetics, first order."""

    n_full = 6
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

    def calc_C(self, params=None):
        super(ABC_DEF_Model, self).calc_C(params)

        c01, c02, k1, k2, k3, k4 = self.params['c01'].value, self.params['c02'].value, self.params['k1'].value, \
                                   self.params['k2'].value, self.params['k3'].value, self.params['k4'].value

        self.C_full[:, 0] = np.heaviside(self.times, 1) * c01 * np.exp(-self.times * k1)
        self.C_full[:, 1] = self.cB(self.times, c01, k1, k2)
        self.C_full[:, 2] = np.heaviside(self.times, 1) * (c01 - self.C_full[:, 0] - self.C_full[:, 1])

        self.C_full[:, 3] = np.heaviside(self.times, 1) * c02 * np.exp(-self.times * k3)
        self.C_full[:, 4] = self.cB(self.times, c02, k3, k4)
        self.C_full[:, 5] = np.heaviside(self.times, 1) * (c02 - self.C_full[:, 3] - self.C_full[:, 4])

        return self.get_conc_matrix()


# class ABC2nd_Model(Model):
#     n_full = 3
#
#     # bounds = ((0, np.inf), (0, np.inf))
#
#     def __init__(self, times=None, visible=None, p0=None):
#         super(ABC2nd_Model, self).__init__(times, visible, p0)
#
#     def init_visible(self):
#         self.visible = [True, True, False]
#
#     def init_params(self):
#         self.params = Parameters()
#         self.params.add('c0', value=1, min=-np.inf, max=np.inf)
#         self.params.add('k1', value=1, min=0, max=np.inf)
#         self.params.add('k2', value=0.5, min=0, max=np.inf)
#         self.params.add('n', value=1.2, min=1, max=10)
#
#     def calc_C(self, params=None):
#         super(ABC2nd_Model, self).calc_C(params)
#
#         c0, k1, k2, n = self.params['c0'].value, self.params['k1'].value, self.params['k2'].value, self.params[
#             'n'].value
#
#         self.C[:, 0] = np.heaviside(self.times, 1) * c0 * np.exp(-self.times * k1)
#
#         def dB_dt(cB, t):
#             cA = np.heaviside(t, 1) * c0 * np.exp(-t * k1)
#             return k1 * cA - k2 * np.sign(cB) * np.power(np.abs(cB), n)  # d[B]/dt = k2[A] - k3[B]^2
#
#         # initial condition, cB(t=0) = 0
#         ys = odeint(dB_dt, 0, self.times)
#         self.C[:, 1] = np.heaviside(self.times, 1) * ys.flatten()
#
#         self.C[:, 2] = np.heaviside(self.times, 1) * (c0 - self.C[:, 0] - self.C[:, 1])
#
#         return self.C



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
