import numpy as np
import lmfit
from scipy.linalg import lstsq
from copy import deepcopy
from scipy.optimize import nnls as _nnls
from numba import njit
# # from theano.ode import DifferentalEquation
# from pymc3.ode import DifferentialEquation
import math

def find_nearest_idx(array, value):
    if isinstance(value, (int, float)):
        value = np.asarray([value])
    else:
        value = np.asarray(value)

    result = np.empty_like(value, dtype=int)
    for i in range(value.shape[0]):
        idx = np.searchsorted(array, value[i], side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value[i] - array[idx - 1]) < math.fabs(value[i] - array[idx])):
            result[i] = idx - 1
        else:
            result[i] = idx
    return result if result.shape[0] > 1 else result[0]

from .constraints import ConstraintNonneg

# @njit(fastmath=True, parallel=True)
def _res_varpro(C, D):
    """Calculates residuals efficiently  by (I - CC+)D.
    Projector CC+ is calculated by SVD: CC+ = U @ U.T.

    Removal of the columns of U that does not correspond to data is needed
    (those columns whose corresponding singular values ar    e
    lower
    than
    tolerance)"""

    U, S, VT = np.linalg.svd(C, full_matrices=False)

    Sr = S[S > S[0] * 1e-10]
    Ur = U[:, :Sr.shape[0]]

    R = (np.eye(C.shape[0]) - Ur.dot(Ur.T)).dot(D)

    return R.flatten()

def mse(C, ST, D_actual, D_calculated):
    """ Mean square error """
    return ((D_actual - D_calculated) ** 2).sum() / D_actual.size


def NNLS(A, B):
    """solve least squares solution for X: AX=B by non-negative least squares"""

    if B.ndim == 2:
        N = B.shape[-1]
    else:
        N = 0

    X = np.zeros((A.shape[-1], N))
    residual = np.zeros((N))

    # nnls is Ax = b; thus, need to iterate along
    # columns of B
    if N == 0:
        X, residual = _nnls(A, B)
    else:
        for i in range(N):
            X[:, i], residual[i] = _nnls(A, B[:, i])

    return X, residual


def OLS(A, B):
    """solve least squares solution for X: AX=B by ordinary least squares"""

    X, residual, rank, svs = lstsq(A, B)
    return X, residual


class Fitter:
    """
    Multivariate Curve Resolution - Alternating Regression
    Purely soft modeling or hard modeling - fitting model applied to C or ST
    or combination of these methods

    D = CS^T

    """

    def __init__(self, times=None, wls=None, D: np.ndarray = None, **kwargs):

        self.times = times
        self.wls = wls
        self.D = D

        self.t_dim = None
        self.w_dim = None

        # number of components - target
        self.n = None

        # fitting algorithm - only for hard modeling
        self.fit_alg = 'leastsq'

        self.max_iter = 10
        self.c_model = None

        self.C_est = None  # initial C matrix
        self.ST_est = None  # initial ST matrix
        self.au = None  # for augmented matrix

        self.c_constraints = None  # constraints on C matrix profiles
        self.st_constraints = None  # constraints on ST matrix profiles

        self.C_matrix_constraints = None
        # for MCR algorithm
        self.C_regressor = 'ols'  # can be 'ols' for ordinary least squares or 'nnls' for non-negative LS
        self.S_regressor = 'ols'

        self.c_fix = None
        self.st_fix = None

        self.ST_opt = None
        self.C_opt = None
        self.last_result = None  # last fitting result
        self.minimizer = None
        self.lof = 0  # lack of fit

        self.kwds = None  # keywords args to pass to underlying fitting function

        self.update_options(**kwargs)

    def update_options(self, **kwargs):
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise TypeError(f'Argument {key} is not valid.')
            self.__setattr__(key, value)

        self.t_dim = self.times.shape[0] if self.times is not None else None  # number of points in time dimension
        self.w_dim = self.wls.shape[0] if self.wls is not None else None  # number of points in wavelength dimension

        self.ST_opt = self.ST_est
        self.C_opt = self.C_est

    def _C_regressor(self, A, B):
        return NNLS(A, B) if self.C_regressor.lower() == 'nnls' else OLS(A, B)

    def _S_regressor(self, A, B):
        return NNLS(A, B) if self.S_regressor.lower() == 'nnls' else OLS(A, B)

    # C MCR half fit
    def calc_C(self):
        if self.D is None or self.ST_opt is None:
            raise ValueError("Matrix D or spectra matrix S^T cannot be None.")

        self.C_opt = self._C_regressor(self.ST_opt.T, self.D.T)[0].T

        # Apply fixed C's
        if self.c_fix:
            self.C_opt[:, self.c_fix] = self.C_est[:, self.c_fix]

        if self.c_constraints is not None:
            # Apply c-constraints
            for i in range(self.n):
                for constr in self.c_constraints[i]:
                    self.C_opt[:, i] = constr.transform(self.C_opt[:, i])

        if self.C_matrix_constraints is not None:
            for constr in self.C_matrix_constraints:
                self.C_opt = constr.transform(self.C_opt)

        # Apply fixed C's
        if self.c_fix:
            self.C_opt[:, self.c_fix] = self.C_est[:, self.c_fix]

    # ST MCR half fit
    def calc_ST(self):
        if self.D is None or self.C_opt is None:
            raise ValueError("Matrix D or concentration matrix C cannot be None.")

        self.ST_opt = self._S_regressor(self.C_opt, self.D)[0]

        # Apply fixed ST's
        if self.st_fix:
            self.ST_opt[self.st_fix] = self.ST_est[self.st_fix]

        if self.st_constraints is not None:
            # Apply st-constraints
            for i in range(self.n):
                for constr in self.st_constraints[i]:
                    self.ST_opt[i] = constr.transform(self.ST_opt[i])

        # Apply fixed ST's
        if self.st_fix:
            self.ST_opt[self.st_fix] = self.ST_est[self.st_fix]

    # special HS-MCR-ALS fit for photokinetic model
    def H_fit_multiset(self, **kwargs):

        self.update_options(**kwargs)

        _C_opt = self.C_est.copy()
        _ST_opt = self.ST_est.copy()

        # C_est, ST_est = self.C_est, self.ST_est

        c_fix = self.c_fix
        st_fix = self.st_fix

        # self.fit_alg = fit_alg if fit_alg is not None else self.fit_alg
        # self.n = n if n is not None else self.n
        # self.st_fix = st_fix if st_fix is not None else self.st_fix
        # self.c_fix = c_fix if c_fix is not None else self.c_fix

        # _C_opt = C_est.copy() if C_est is not None else np.zeros_like(C_est)
        # _ST_opt = ST_est.copy() if ST_est is not None else np.zeros_like(ST_est)

        it_mcr = 5

        def residuals(params):
            # needed to use nonlocal because of nested functions, https://stackoverflow.com/questions/5218895/python-nested-functions-variable-scoping
            nonlocal _ST_opt, _C_opt

            for i in range(it_mcr):



                # # perform MCR of calculating C profiles from spectra
                # _C_opt = lstsq(_ST_opt.T, self.D.T)[0].T
                #
                _C_opt = self.c_model.calc_C(params, _C_opt)
                # apply nonzero constraint
                # _C_opt *= (_C_opt > 0)

                if not st_fix or len(st_fix) < self.n:
                    _ST_opt = lstsq(_C_opt, self.D)[0]
                # nonnegative constrain
                _ST_opt *= (_ST_opt > 0)

                _ST_opt[0] *= 26139.01 / _ST_opt[0].max()

                # apply fixed spectra
                if st_fix:
                    _ST_opt[st_fix] = self.ST_est[st_fix]

                setattr(self.c_model, 'ST', _ST_opt)

            # # # apply closure contstrain on C profiles
            # # _C_opt = 36e-6 * _C_opt / _C_opt.sum(axis=1, keepdims=True)
            #
            # # calculate kinetic profiles based ont the model
            # _C_opt = self.c_model.calc_C(params, _C_opt)
            #
            # # apply nonzero constraint
            # _C_opt *= (_C_opt > 0)
            #
            #             # _C_opt[_C_opt < 100] = 100
            #
            # if c_fix:
            #     _C_opt[:, c_fix] = C_est[:, c_fix]
            # # if we don't fix all spectral components or don't provide any spectra, calculate them by lstsq
            # if not st_fix or len(st_fix) < self.n:
            #     _ST_opt = lstsq(_C_opt, self.D)[0]
            # # apply fixed spectra
            # if st_fix:
            #     _ST_opt[st_fix] = ST_est[st_fix]
            #
            # # nonnegative constrain
            # _ST_opt *= (_ST_opt > 0)
            #
            # # apply spectra constraints
            # # normalize Z to 26139.01
            #
            # _ST_opt[0] *= 26139.01 / _ST_opt[0].max()
            #
            # # apply fixed spectra
            # if st_fix:
            #     _ST_opt[st_fix] = ST_est[st_fix]
            #
            # # set spectra matrix to model as a parameter
            # setattr(self.c_model, 'ST', _ST_opt)

            R = np.dot(_C_opt, _ST_opt) - self.D  # calculate residuals
            return R

        self.minimizer = lmfit.Minimizer(residuals, self.c_model.params)
        self.last_result = self.minimizer.minimize(method=self.fit_alg)  # minimize the residuals

        self.c_model.params = self.last_result.params

        self.C_opt = _C_opt
        self.ST_opt = _ST_opt

        return True

    def obj_func_fit(self, C_est=None, **kwargs):  # resolving factor analysis
        self.update_options(**kwargs)

        _C_opt = self.C_est.copy() if C_est is None else C_est

        w_idx = find_nearest_idx(self.wls, 260)
        weights = np.ones((self.D.shape[0] + 5, self.D.shape[1]))

        weights[:, :w_idx] = 0.1  # weights in the region of 230 - 260 are 0.1

        def residuals(params):
            nonlocal _C_opt

            T = self.c_model.get_T(params)
            ST = T.dot(self.c_model.VT)

            # C_basis = self.c_model.U @ self.c_model.Sigma @ np.linalg.inv(T)

            self.c_model.ST = ST

            t, w, n = self.D.shape[0], self.D.shape[1], T.shape[0]

            _C_opt = self.c_model.calc_C(params, _C_opt)

            R = np.zeros((t + n + 1, w), dtype=np.float64)  # residual matrix
            # R = np.zeros((n + 1, w), dtype=np.float64)  # residual matrix


            # nonnegativity of spectra
            # put negative values, positives will be zero, normalization to maximum of individual spectrum
            R_sp = ST * (ST < 0) / ST.max(axis=0, keepdims=True) * 2

            R[:n, :] = R_sp

            # fixed spectrum
            # R[n, :] = (self.c_model.Z_true - ST[0]) / self.c_model.Z_true.max()
            R[n, :] = np.ones_like(w) * (self.c_model.Z_true.max() - ST[0].max())  # norm to maximum
            #
            # _C_opt = self.c_model.calc_C(params, _C_opt)
            #
            R[n+1:, :] = (_C_opt @ ST - self.D) / self.D.max()

            # R_C = (_C_opt - C_basis) / _C_opt.max()

            # return np.hstack((R.flatten(), R_C.flatten()))
            return R * weights

        self.minimizer = lmfit.Minimizer(residuals, self.c_model.params)
        kws = {} if self.kwds is None else self.kwds
        self.last_result = self.minimizer.minimize(method=self.fit_alg, **kws)  # minimize the residuals

        self.c_model.params = self.last_result.params

        self.C_opt = _C_opt
        self.ST_opt = self.c_model.ST


    def var_pro(self, C_est=None, c_fix=None, **kwargs):
        self.update_options(**kwargs)

        "only conc. profiles can be fixed, but not spectra"

        _C_opt = C_est.copy() if C_est is not None else None #np.zeros_like(self.C_est)

        def residuals(params):
            # needed to use nonlocal because of nested functions, https://stackoverflow.com/questions/5218895/python-nested-functions-variable-scoping
            nonlocal _C_opt
            _C_opt = self.c_model.calc_C(params, _C_opt)
            if c_fix:
                _C_opt[:, c_fix] = C_est[:, c_fix]

            # _ST_calc = NNLS(_C_opt, self.D)[0]
            #
            # return _C_opt @ _ST_calc - self.D

            # calculate the residual matrix by varpro (I - CC+)D
            return _res_varpro(_C_opt, self.D)

        self.minimizer = lmfit.Minimizer(residuals, self.c_model.params)
        kws = {} if self.kwds is None else self.kwds
        self.last_result = self.minimizer.minimize(method=self.fit_alg, **kws)  # minimize the residuals

        self.c_model.params = self.last_result.params

        self.C_opt = _C_opt
        self.ST_opt = lstsq(_C_opt, self.D)[0]
        # self.ST_opt = NNLS(_C_opt, self.D)[0]

    def var_pro_femto(self, **kwargs):
        self.update_options(**kwargs)

        # _C_opt = C_est.copy() if C_est is not None else None  # np.zeros_like(self.C_est)
        D_fit = np.zeros_like(self.D)
        _C_tensor = None
        _ST = np.zeros((self.ST_opt.shape[0] + self.c_model.coh_spec_order + 1,
                        self.ST_opt.shape[1])) if self.c_model.coh_spec else np.zeros_like(self.ST_opt)

        w_idxs = find_nearest_idx(self.wls, [378, 393])
        weights = np.ones_like(self.D)

        weights[:, w_idxs[0]:w_idxs[1]] = 0.05  # weights in the region of 378 to 393 nm is set to 0.1

        coh_idx = find_nearest_idx(self.wls, 460)
        coh_scale = np.ones_like(self.wls)
        coh_scale[coh_idx:] = 0

        # coh_scale = None

        def residuals(params):
            # needed to use nonlocal because of nested functions, https://stackoverflow.com/questions/5218895/python-nested-functions-variable-scoping
            nonlocal _C_tensor, D_fit

            _C_tensor = self.c_model.calc_C(params)

            if self.c_model.coh_spec:
                _C_COH = self.c_model.simulate_coh_gaussian(params, coh_scale)
                _C_tensor = np.concatenate((_C_tensor, _C_COH), axis=-1)

            _C_tensor = np.nan_to_num(_C_tensor)

            for i in range(self.wls.shape[0]):
                _ST[:, i] = lstsq(_C_tensor[i], self.D[:, i])[0]

            if self.c_model.coh_spec:
                self.c_model.ST_COH = _ST[-self.c_model.coh_spec_order - 1:]

            D_fit = np.matmul(_C_tensor, _ST.T[..., None]).squeeze().T

            R = self.D - D_fit

            R = np.nan_to_num(R)

            return R * weights

        self.minimizer = lmfit.Minimizer(residuals, self.c_model.params)
        kws = {} if self.kwds is None else self.kwds
        self.last_result = self.minimizer.minimize(method=self.fit_alg, **kws)  # minimize the residuals

        self.c_model.params = self.last_result.params

        self.C_opt = _C_tensor[0, :, :-self.c_model.coh_spec_order - 1] if self.c_model.coh_spec else _C_tensor[0]
        self.ST_opt = _ST[:-self.c_model.coh_spec_order - 1] if self.c_model.coh_spec else _ST

        return D_fit


    def _set_C_indiv(self, Ci, i, j=0):
        assert self.au is not None
        idx_0, idx_1 = self.au._C_indiv_range(i, j=j)
        self.C_opt[idx_0:idx_1, :] = Ci


    # optimization of C profiles according to kinetic model in HS-MCR-AR
    def _C_fit_opt(self):
        # C optimized by kinetic model

        # i = 2  # for specific kinetic model
        # idx_0, idx_1 = self.au._C_indiv_range(i)

        # _C_est = self.C_opt[idx_0:idx_1, :] if self.au else self.C_opt
        # _C_fit = self.C_est[idx_0:idx_1, :].copy() if self.au else self.C_est.copy()

        _C_est = self.C_opt
        _C_fit = self.C_est.copy()

        def residuals(params):
            nonlocal _C_fit, _C_est
            _C_fit = self.c_model.calc_C(params, _C_fit)
            if self.c_fix:
                _C_fit[:, self.c_fix] = self.C_est[:, self.c_fix]

            R = _C_est - _C_fit

            return R

        self.minimizer = lmfit.Minimizer(residuals, self.c_model.params)
        kws = {} if self.kwds is None else self.kwds
        self.last_result = self.minimizer.minimize(method=self.fit_alg, **kws)  # minimize the residuals

        self.c_model.params = self.last_result.params
        #
        # if self.au:
        #     self.C_opt[idx_0:idx_1, :] = _C_fit
        # else:
        #     self.C_opt = _C_fit

        self.C_opt = _C_fit

    # general case that include both possibilities - pure MCR fit with constraints
    # and no hard modeling involved or HS-MCR - MCR with concentrations hard constraints
    def HS_MCR_fit(self, **kwargs):

        self.update_options(**kwargs)

        # Ensure only C or ST provided
        if (self.C_est is None) & (self.ST_est is None):
            raise TypeError('C or ST estimate must be provided')

        # if ST estimate is not provided, calculate ST from C estimate by lstsq

        if self.ST_est is None:
            self.ST_opt = lstsq(self.C_est, self.D)[0]

        assert self.n == self.ST_opt.shape[0]

        # st and c constraints are lists of lists of constraints for each C and ST component
        if self.st_constraints is not None:
            assert len(self.st_constraints) == self.n
        if self.c_constraints is not None:
            assert len(self.c_constraints) == self.n

        for i in range(self.max_iter):

            self.calc_C()

            # perform hard modeling for C if model is provided only to C_opt optimized in previous step

            if self.c_model:
                setattr(self.c_model, 'ST', self.ST_opt)
                self._C_fit_opt()

                # self.H_fit(self.c_model, self.ST_opt, self.C_est, [i for i in range(self.n)], self.c_fix)

            self.calc_ST()

            # normalize Z to constant value

            self.ST_opt[0] *= 29043 / self.ST_opt[0].max()

            # E must have the same epsilon at 444 nm as Z has

            idx = find_nearest_idx(self.wls, 443 - 230)

            self.ST_opt[1] *= self.ST_opt[0, idx] / self.ST_opt[1, idx]


        return True
