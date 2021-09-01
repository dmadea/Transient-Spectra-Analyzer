import numpy as np
import lmfit
from scipy.linalg import lstsq
from numpy.linalg import pinv
import scipy
from copy import deepcopy
from scipy.optimize import nnls as _nnls
from scipy.optimize import least_squares
from numba import njit
import sys
from misc import find_nearest_idx
import time

posv = scipy.linalg.get_lapack_funcs(('posv'))

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


def res_par_varpro(C, D, rcond=1e-10):
    """Calculates residuals efficiently  by partitioned variable projection.
    residuals are (I - CC+)D. C is tensor (n_w, n_t, k), D is matrix (n_t, n_w)
    Projector CC+ is calculated by batched pseudoinverse
    """

    # t0 = time.perf_counter()

    Cp = pinv(C, rcond=rcond)  # calculate batch pseudoinverse
    I = np.eye(C.shape[1])[None, ...]  # eye matrix
    P = I - np.matmul(C, Cp)  # calculate the projector
    residuals = np.matmul(P, D.T[..., None]).squeeze().T  # and final residuals

    # tdiff = time.perf_counter() - t0
    # print(f"res_par_varpro took {tdiff * 1e3} ms")

    return residuals


@njit(parallel=False, fastmath=False)
def res_par_varpro_nb(C, D, rcond=1e-10):
    """Calculates residuals efficiently  by partitioned variable projection.
    residuals are (I - CC+)D. C is tensor (n_w, n_t, k), D is matrix (n_t, n_w)
    Projector CC+ is calculated by batched pseudoinverse
    """

    residuals = np.empty_like(D)

    I = np.eye(C.shape[1])

    for i in range(C.shape[0]):
        Cpi = pinv(C[i], rcond=rcond)
        P = I - C[i].dot(Cpi)  # projector C @ C+

        #         P.flat[::P.shape[0] + 1] += 1  # add ones to main diagonal
        residuals[:, i] = P.dot(D[:, i])

    return residuals


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


def OLS_ridge(A, B, alpha=0.0001):
    """fast: solve least squares solution for X: AX=B by ordinary least squares, with direct solve,
    with optional Tikhonov regularization"""

    ATA = A.T.dot(A)
    ATB = A.T.dot(B)

    if alpha != 0:
        ATA.flat[::ATA.shape[-1] + 1] += alpha

    c, x, info = posv(ATA, ATB, lower=False,
                      overwrite_a=True,
                      overwrite_b=False)

    return x, None


class Fitter:
    """
    Multivariate Curve Resolution - Alternating Regression
    Purely soft modeling or hard modeling - fitting model applied to C or ST
    or combination of these methods

    D = CS^T

    """

    def __init__(self, times=None, wls=None, D: np.ndarray = None, **kwargs):

        self.fit_varpro = False

        self.times = times
        self.wls = wls
        self.D = D

        self.regressors = ['ols', 'ridge', 'nnls']

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
        self.C_regressor = self.regressors[1]
        self.S_regressor = self.regressors[1]
        self.regressor_alpha = 0.0001

        self.c_fix = None
        self.st_fix = None

        self.ST_opt = None
        self.C_opt = None
        self.last_result = None  # last fitting result
        self.minimizer = None
        self.lof = 0  # lack of fit
        self.verbose = 2  # 0 for non verbose, 2 for verbose, like for lmfit
        self.is_interruption_requested = lambda: False  # function that returns True or False, default no interruption

        # keywords args to pass to underlying fitting function - lmfit
        self.kwds = {'ftol': 1e-10, 'xtol': 1e-10, 'gtol': 1e-10, 'loss': 'linear', 'verbose': self.verbose}

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

        # self.kwds = {'ftol': 1e-10, 'xtol': 1e-10, 'gtol': 1e-10, 'loss': 'linear', 'verbose': self.verbose}

    def _regressor(self, A, B, method='ridge'):
        if method.lower() == self.regressors[0]:
            return OLS(A, B)
        elif method.lower() == self.regressors[1]:
            return OLS_ridge(A, B, self.regressor_alpha)
        else:
            return NNLS(A, B)

    # C MCR half fit
    def calc_C(self):
        if self.D is None or self.ST_opt is None:
            raise ValueError("Matrix D or spectra matrix S^T cannot be None.")

        self.C_opt = self._regressor(self.ST_opt.T, self.D.T, method=self.C_regressor)[0].T

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

        self.ST_opt = self._regressor(self.C_opt, self.D, method=self.S_regressor)[0]

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
        #
        # w_idx = find_nearest_idx(self.wls, 260)
        # weights = np.ones((self.D.shape[0] + 5, self.D.shape[1]))

        # weights[:, :w_idx] = 0.5  # weights in the region of 230 - 260 are 0.1

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
            # R[n, :] = np.ones_like(w) * (self.c_model.Z_true.max() - ST[0].max())  # norm to maximum
            Z_eps415 = 39033
            R[n, :] = np.ones_like(w) * (Z_eps415 - ST[0].max())  # norm to maximum

            #
            # _C_opt = self.c_model.calc_C(params, _C_opt)
            #
            R[n+1:, :] = (_C_opt @ ST - self.D) / self.D.max()

            # R_C = (_C_opt - C_basis) / _C_opt.max()

            # return np.hstack((R.flatten(), R_C.flatten()))
            # return R * weights
            return R

        self.minimizer = lmfit.Minimizer(residuals, self.c_model.params)
        # kws = {} if self.kwds is None else self.kwds
        kws = {'verbose': 2}
        self.last_result = self.minimizer.minimize(method=self.fit_alg, **kws)  # minimize the residuals

        self.c_model.params = self.last_result.params

        self.C_opt = _C_opt
        self.ST_opt = self.c_model.ST

    # fit full kinetic model with fix possibilities
    def fit_full_model(self, **kwargs):
        self.update_options(**kwargs)

        _C_opt = None
        _ST_opt = None

        def residuals(params):
            nonlocal _C_opt, _ST_opt
            _C_opt = self.c_model.calc_C(params, _C_opt)  # calculate conc. profiles based on kin. model
            if self.c_fix:  # replace C profiles to fixed ones if fix is defined
                _C_opt[:, self.c_fix] = self.C_est[:, self.c_fix]

            _ST_opt = self._regressor(_C_opt, self.D, method=self.S_regressor)[0]  # calculate spectra by S regressor (OLS or NNLS)

            if self.st_fix:  # replace spectra to fixed ones if fix is defined
                _ST_opt[self.st_fix, :] = self.ST_est[self.st_fix]

            R = _C_opt @ _ST_opt - self.D  # calculate residuals

            return np.nan_to_num(R)

        self.minimizer = lmfit.Minimizer(residuals, self.c_model.params,
                                         iter_cb=lambda params, iter, resid, *args, **kws: self.is_interruption_requested())
        self.kwds.update(kwargs)
        self.last_result = self.minimizer.minimize(method=self.fit_alg, **self.kwds)  # minimize the residuals

        self.c_model.params = self.last_result.params

        self.C_opt = _C_opt
        self.ST_opt = _ST_opt

    # def var_pro(self, C_est=None, c_fix=None, **kwargs):
    #     self.update_options(**kwargs)
    #
    #     "only conc. profiles can be fixed, but not spectra"
    #
    #     _C_opt = C_est.copy() if C_est is not None else None #np.zeros_like(self.C_est)
    #
    #     def residuals(params):
    #         # needed to use nonlocal because of nested functions, https://stackoverflow.com/questions/5218895/python-nested-functions-variable-scoping
    #         nonlocal _C_opt
    #         _C_opt = self.c_model.calc_C(params, _C_opt)
    #         if c_fix:
    #             _C_opt[:, c_fix] = C_est[:, c_fix]
    #
    #         # _ST_calc = NNLS(_C_opt, self.D)[0]
    #         #
    #         # return _C_opt @ _ST_calc - self.D
    #
    #         # calculate the residual matrix by varpro (I - CC+)D
    #         return _res_varpro(_C_opt, self.D)
    #
    #     self.minimizer = lmfit.Minimizer(residuals, self.c_model.params)
    #     # self.kwds = {'ftol': 1e-8, 'xtol': 1e-8, 'gtol': 1e-8, 'loss': 'linear', 'verbose': 0}
    #     self.last_result = self.minimizer.minimize(method=self.fit_alg, **self.kwds)  # minimize the residuals
    #
    #     self.c_model.params = self.last_result.params
    #
    #     self.C_opt = _C_opt
    #     self.ST_opt = lstsq(_C_opt, self.D)[0]
    #     # self.ST_opt = NNLS(_C_opt, self.D)[0]

    def var_pro_femto(self, **kwargs):
        self.update_options(**kwargs)

        def residuals(params):
            # needed to use nonlocal because of nested functions, https://stackoverflow.com/questions/5218895/python-nested-functions-variable-scoping
            # nonlocal D_fit

            t0 = time.perf_counter()

            if self.fit_varpro:
                C = self.c_model.simulate_C_tensor(params)
                R = res_par_varpro(C, self.D)
            else:
                # classical naive batched least squares solution is so far almost 1 order of magnitude faster than "fancy"
                # partitioned variable projection algorithm...
                # pinv could be improved by really partitioning, because the projector matrices are large
                # or to use GPUs
                D_fit, self.C_opt, self.ST_opt = self.c_model.simulate_mod(self.D, params)
                R = self.D - D_fit


            # R = np.nan_to_num(R)
            weights = self.c_model.get_weights(params)

            tdiff = time.perf_counter() - t0
            print(f"residuals took {tdiff * 1e3} ms")

            return R * weights

        # iter_cb - callback function
        self.minimizer = lmfit.Minimizer(residuals, self.c_model.params,
                                         iter_cb=lambda params, iter, resid, *args, **kws: self.is_interruption_requested())
        self.kwds.update(kwargs)
        self.last_result = self.minimizer.minimize(method=self.fit_alg, **self.kwds)  # minimize the residuals

        self.c_model.params = self.last_result.params

        D_fit, self.C_opt, self.ST_opt = self.c_model.simulate_mod(self.D, self.c_model.params)

        # self.C_opt = _C_tensor[0, :, :-self.c_model.coh_spec_order - 1] if self.c_model.coh_spec else _C_tensor[0]
        # self.ST_opt = _ST[:-self.c_model.coh_spec_order - 1] if self.c_model.coh_spec else _ST

        return D_fit


    def _set_C_indiv(self, Ci, i, j=0):
        assert self.au is not None
        idx_0, idx_1 = self.au._C_indiv_range(i, j=j)
        self.C_opt[idx_0:idx_1, :] = Ci


    # optimization of only C profiles according to kinetic model in HS-MCR-AR
    def _C_fit_opt(self):
        # C optimized by kinetic model

        _C_est = self.C_opt
        _C_fit = self.C_opt.copy()

        def residuals(params):
            nonlocal _C_fit, _C_est
            _C_fit = self.c_model.calc_C(params, _C_fit)
            if self.c_fix:
                _C_fit[:, self.c_fix] = self.C_est[:, self.c_fix]

            R = _C_est - _C_fit

            return np.nan_to_num(R)

        self.minimizer = lmfit.Minimizer(residuals, self.c_model.params)
        # self.kwds = {'ftol': 1e-8, 'xtol': 1e-8, 'gtol': 1e-8, 'loss': 'linear', 'verbose': 0}
        self.last_result = self.minimizer.minimize(method=self.fit_alg, **self.kwds)  # minimize the residuals

        self.c_model.params = self.last_result.params

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
            self.ST_opt = self._S_regressor(self.C_est, self.D)[0]

        assert self.n == self.ST_opt.shape[0]

        # st and c constraints are lists of lists of constraints for each C and ST component
        if self.st_constraints is not None:
            assert len(self.st_constraints) == self.n
        if self.c_constraints is not None:
            assert len(self.c_constraints) == self.n

        # if self.verbose == 2:
        #     print('Iteration\tSum of squares\tLack of Fit')

        for i in range(self.max_iter):

            self.calc_C()

            # perform hard modeling for C if model is provided only to C_opt optimized in previous step

            if self.c_model:
                if self.verbose == 2:
                    print("Optimization of C profiles:")
                setattr(self.c_model, 'ST', self.ST_opt)
                self._C_fit_opt()

            self.calc_ST()

            if self.verbose == 2:
                D_fit = self.C_opt @ self.ST_opt
                ssq = ((self.D - D_fit) ** 2).sum()
                lof = np.sqrt(ssq / (self.D ** 2).sum()) * 100
                print(f'\nIteration {i+1}.\tSum of squares {ssq:.4g}\tLack of Fit {lof:.4g}')
                if self.c_model:
                    print('------------------------------------------------------------------\n')

                # print(f'{i+1}.\t{ssq:.4g}\t{lof:.4g}')

            if self.is_interruption_requested():
                break

