# import numpy as np  # import numpy package (abbreviation stands for Numerical Python)
# import matplotlib.pyplot as plt  # we plot graphs with this library
# import math
#
# from matplotlib import cm
# import matplotlib as mpl
# from matplotlib import gridspec
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#
# from matplotlib.ticker import *
#
#
# # works for numbers and array of numbers
# def find_nearest_idx(array, value):
#     if isinstance(value, (int, float)):
#         value = np.asarray([value])
#     else:
#         value = np.asarray(value)
#
#     result = np.empty_like(value, dtype=int)
#     for i in range(value.shape[0]):
#         idx = np.searchsorted(array, value[i], side="left")
#         if idx > 0 and (idx == len(array) or math.fabs(value[i] - array[idx - 1]) < math.fabs(value[i] - array[idx])):
#             result[i] = idx - 1
#         else:
#             result[i] = idx
#     return result if result.shape[0] > 1 else result[0]
#
#
# def find_nearest(array, value):
#     idx = find_nearest_idx(array, value)
#     return array[idx]
#
#
# MAJOR_TICK_DIRECTION = 'in'  # in, out or inout
# MINOR_TICK_DIRECTION = 'in'  # in, out or inout
# MARKER_SIZE = 20
#
# Z_NUM = '$Z$-$\\bf{Xa}$'
# E_NUM = '$E$-$\\bf{Xa}$'
# HL_NUM = '$\\bf{Xb}$'
#
# # EPS_LABEL = '$\\varepsilon$ ($10^4$ M$^{-1}$ cm$^{-1}$)'
# WL_LABEL = 'Wavelength (nm)'
#
# def eps_label(factor):
#     num = np.log10(1/factor).astype(int)
#     return f'$\\varepsilon$ ($10^{num}$ M$^{-1}$ cm$^{-1}$)'
#
# X_SIZE, Y_SIZE = 5, 4
#
# COLORS = ['blue', 'red', 'green', 'black']
#
#
# def setup_twin_x_axis(ax, y_label="Molar spectral flux ($10^{-10}\\ mol\\ s^{-1}\\ nm^{-1}$)",
#                       x_label=None, ylim=(None, None), y_major_locator=None, y_minor_locator=None,
#                       keep_zero_aligned=True):
#     ax2 = ax.twinx()
#
#     ax2.tick_params(which='major', direction=MAJOR_TICK_DIRECTION)
#     ax2.tick_params(which='minor', direction=MINOR_TICK_DIRECTION)
#
#     if y_major_locator:
#         ax2.yaxis.set_major_locator(y_major_locator)
#
#     if y_minor_locator:
#         ax2.yaxis.set_minor_locator(y_minor_locator)
#
#     ax2.set_ylabel(y_label)
#
#     if keep_zero_aligned and ylim[0] is None and ylim[1] is not None:
#         # a = bx/(x-1)
#         ax1_ylim = ax1.get_ylim()
#         x = -ax1_ylim[0] / (ax1_ylim[1] - ax1_ylim[0])  # position of zero in ax1, from 0, to 1
#         a = ylim[1] * x / (x - 1)  # calculates the ylim[0] so that zero position is the same for both axes
#         ax2.set_ylim(a, ylim[1])
#
#     elif ylim[0] is not None:
#         ax2.set_ylim(ylim)
#
#     return ax2
#
#
# def setup_wavenumber_axis(ax, x_label="Wavenumber ($10^{4}$ cm$^{-1}$)",
#                           x_major_locator=None, x_minor_locator=AutoMinorLocator(5), factor=1e3):
#     secondary_ax = ax.secondary_xaxis('top', functions=(lambda x: factor / x, lambda x: 1 / (factor * x)))
#
#     secondary_ax.tick_params(which='major', direction=MAJOR_TICK_DIRECTION)
#     secondary_ax.tick_params(which='minor', direction=MINOR_TICK_DIRECTION)
#
#     if x_major_locator:
#         secondary_ax.xaxis.set_major_locator(x_major_locator)
#
#     if x_minor_locator:
#         secondary_ax.xaxis.set_minor_locator(x_minor_locator)
#
#     secondary_ax.set_xlabel(x_label)
#
#     return secondary_ax
#
#
# def set_main_axis(ax, x_label=WL_LABEL, y_label="Absorbance", xlim=(None, None), ylim=(None, None),
#                   x_major_locator=None, x_minor_locator=None, y_major_locator=None, y_minor_locator=None):
#     ax.set_ylabel(y_label)
#     ax.set_xlabel(x_label)
#     if xlim[0] is not None:
#         ax.set_xlim(xlim)
#     if ylim[0] is not None:
#         ax.set_ylim(ylim)
#
#     if x_major_locator:
#         ax.xaxis.set_major_locator(x_major_locator)
#
#     if x_minor_locator:
#         ax.xaxis.set_minor_locator(x_minor_locator)
#
#     if y_major_locator:
#         ax.yaxis.set_major_locator(y_major_locator)
#
#     if y_minor_locator:
#         ax.yaxis.set_minor_locator(y_minor_locator)
#
#     ax.tick_params(axis='both', which='major', direction=MAJOR_TICK_DIRECTION)
#     ax.tick_params(axis='both', which='minor', direction=MINOR_TICK_DIRECTION)
#
# path = r'C:\Users\Dominik\Documents\MUNI\Organic Photochemistry\Projects\2019-Bilirubin project\UV-VIS\QY measurement\Photodiode\new setup'
#
# fnames = [[r'\Z 350p\cut.txt', r'\E 350p\cut.txt',  r'\E 500p\cut.txt'],
#          [r'\reactor LEDs\HL 355\cut.txt', r'\reactor LEDs\Z 355\cut.txt', r'\reactor LEDs\Z 490\cut.txt']]
#
# if __name__ == '__main__':
#     fig, ax1 = plt.subplots(1, 1, figsize=(1 * X_SIZE, Y_SIZE))
#
#     data = np.genfromtxt(path + fnames[1][1], delimiter='\t')
#     #     data = data[:, :-10]
#     t = data[0, 1:]
#     x = data[1:, 0]
#     y = data[1:, 1:]
#
#     step = 2
#     cmap = cm.get_cmap('jet_r')
#     # cmap = cm.get_cmap('gist_rainbow')
#
#     # norm = mpl.colors.LogNorm(vmin=10,vmax=t[-1], clip=True)
#     # norm = mpl.colors.PowerNorm(gamma=0.3, vmin=0,vmax=t[-1], clip=True)
#     linthresh = 50
#     norm = mpl.colors.SymLogNorm(vmin=0, vmax=t[-1], linscale=0.5, linthresh=linthresh, base=10, clip=True)
#
#     set_main_axis(ax1, xlim=(230, 590), x_minor_locator=AutoMinorLocator(5), y_minor_locator=None)
#     secax = setup_wavenumber_axis(ax1)
#
#     ts_empf = (0, 40, 1500, t[-1])
#     tsb_idxs = find_nearest_idx(t, ts_empf)
#     ts_real = t[tsb_idxs]
#
#     n_spectra = 30
#     x_space = np.linspace(0, 1, n_spectra, endpoint=True, dtype=np.float64)
#
#     t_idx_space = find_nearest_idx(t, norm.inverse(x_space))
#     t_idx_space = np.sort(np.asarray(list(set(t_idx_space).union(set(tsb_idxs)))))
#
#     for i in t_idx_space:
#         x_real = norm(t[i])
#         x_real = 0 if np.ma.is_masked(x_real) else x_real
#         ax1.plot(x, y[:, i], color=cmap(x_real),
#                  lw=2 if i in tsb_idxs else 0.6,
#                  alpha=1 if i in tsb_idxs else 0.5,
#                  zorder=5 if i in tsb_idxs else 0)
#
#
#     # for j in range(0, t.shape[0], step):
#     #     idx = norm(t[j])
#     #     #     print(idx)
#     #     idx = 0 if np.ma.is_masked(idx) else idx
#     #     ax1.plot(x, y[:, j], color=cmap(idx),
#     #              lw=2 if j in tsb_idxs else 0.3,
#     #              alpha=1 if j in tsb_idxs else 0.6,
#     #              zorder=5 if j in tsb_idxs else 0)
#
#     # cbar, cbax = add_colorbar(ax1, cmap, norm, major_locator=FixedLocator(t[tsb_idxs]))
#     # cbar, cbax = add_colorbar(ax1, cmap, norm, major_locator=Fixe(linthresh=100, base=10))
#
#     cbaxes = inset_axes(ax1, width="3%", height="80%", loc='center right', borderpad=5.5)
#
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#     # sm.set_array([])
#     cbar = plt.colorbar(sm, cax=cbaxes, orientation='vertical',
#                         format=mpl.ticker.ScalarFormatter(),
#                         # ticks=FixedLocator(ticks),
#                         label="Time (s)")
#
#     cbaxes.invert_yaxis()
#     # cbaxes.minorticks_on()
#     # l = FixedLocator([0, 10, 100, 1000])
#     # cbaxes.yaxis.set_major_locator(l)
#
#     minor_ticks = cbar._locate(
#         [10, 20, 30, 40, 50, 60, 70, 80, 90, 200, 300, 400, 500, 600, 700, 800, 900, 2000, 3000, 4000, 5000])
#     cbar.ax.yaxis.set_ticks(minor_ticks, minor=True)
#
#     major_ticks = np.sort(np.hstack((np.asarray([1e2, 1e3]), ts_real)))
#
#     transformed = cbar._locate(major_ticks)
#
#     cbaxes.yaxis.set_ticks(transformed, minor=False)
#     cbaxes.set_yticklabels([f'{int(num)}' for num in major_ticks])
#
#     for ytick, ytick_label, t in zip(cbaxes.yaxis.get_major_ticks(), cbaxes.get_yticklabels(), major_ticks):
#         if t in ts_real:
#             color = cmap(norm(t))
#             ytick_label.set_color(color)
#             ytick_label.set_fontweight('bold')
#             ytick.tick2line.set_color(color)
#             ytick.tick2line.set_markersize(5)
#             # ytick.tick2line.set_markeredgewidth(2)
#
#     plt.savefig('test_plot.png', dpi=500, bbox_inches='tight')
#
#     plt.show()
#
#


# import multiprocessing
# import time
# import matplotlib.pyplot as plt
# import sklearn.linear_model as lm
# import numpy as np
# import timeit
#
#
# def menger(x1, y1, x2, y2, x3, y3):
#     # x - log R norm, y - log W norm
#
#     P1P2 = (x2 - x1) ** 2 + (y2 - y1) ** 2
#     P2P3 = (x3 - x2) ** 2 + (y3 - y2) ** 2
#     P3P1 = (x1 - x3) ** 2 + (y1 - y3) ** 2
#
#     C2 = 2 * (x1 * y2 + x2 * y3 + x3 * y1 - x2 * y1 - x3 * y2 - x1 * y3) / np.sqrt(P1P2 * P2P3 * P3P1)
#     return C2
#
#
# def L_corner_search(func, max_iter=60, treshold=1e-4, verbose=True, end_alphas=(1e-9, 1e3), run_parallel=False):
#     """Finds the regularization parameter alpha on the L-curve at maximum curvature.
#     Based on L-curve corner search algorithm described in https://arxiv.org/pdf/1608.04571.pdf
#
#     func is a function that takes alpha as an argument and returns the log_R norm and log_W norm tuple"""
#
#     def _shuffle(a, P):
#         a[3] = a[2]
#         P[3] = P[2]
#         a[2] = a[1]
#         P[2] = P[1]
#         a[1] = 10 ** ((np.log10(a[3]) + phi * np.log10(a[0])) / (1 + phi))
#         P[1, 0], P[1, 1] = func(a[1])
#
#     a = np.asarray([end_alphas[0], 0, 0, end_alphas[1]])
#
#     phi = (1 + np.sqrt(5)) / 2  # golden ratio
#
#     a[1] = 10 ** ((np.log10(a[3]) + phi * np.log10(a[0])) / (1 + phi))
#     a[2] = 10 ** (np.log10(a[0]) + np.log10(a[3]) - np.log10(a[1]))
#
#     # P = np.zeros((4, 2), dtype=np.float64)
#
#     if run_parallel:
#         with multiprocessing.Pool() as pool:
#             result = pool.map(func, a)
#             P = np.asarray(result, dtype=np.float64)
#     else:
#         P = np.asarray([func(ai) for ai in a], dtype=np.float64)
#
#     trajectory = np.hstack((a[:, None].copy(), P.copy()))
#
#     alpha_MC = a[0]
#     for i in range(max_iter):
#         C2 = menger(P[0, 0], P[0, 1], P[1, 0], P[1, 1], P[2, 0], P[2, 1])
#         C3 = menger(P[1, 0], P[1, 1], P[2, 0], P[2, 1], P[3, 0], P[3, 1])
#
#         while C3 < 0:
#             _shuffle(a, P)
#             C3 = menger(P[1, 0], P[1, 1], P[2, 0], P[2, 1], P[3, 0], P[3, 1])
#
#         if C2 > C3:
#             alpha_MC = a[1]
#             _shuffle(a, P)
#         else:
#             alpha_MC = a[2]
#             a[0] = a[1]
#             P[0] = P[1]
#             a[1] = a[2]
#             P[1] = P[2]
#             a[2] = 10 ** (np.log10(a[0]) + np.log10(a[3]) - np.log10(a[1]))
#             P[2, 0], P[2, 1] = func(a[2])
#
#         for j in range(4):
#             if a[j] not in trajectory[:, 0]:
#                 trajectory = np.vstack((trajectory, np.asarray([a[j], P[j, 0], P[j, 1]])))
#
#         cost = (a[3] - a[0]) / a[3]
#
#         if verbose:
#             print(f'Iteration {i + 1}:')
#             print(f'Alphas: {a[0]:.4g}, {a[1]:.4g}, {a[2]:.4g}, {a[3]:.4g}; Alpha_MC: {alpha_MC:.4g}')
#             print(f'Cost: {cost:.4g}\n')
#
#         if cost < treshold:
#             break
#
#     return alpha_MC, trajectory[np.argsort(trajectory[:, 0])]
#
#
# t = np.linspace(0, 30, 1000)
#
# alpha = 1e-2
#
# k1, k2, k3 = 2, 1, 0.1
# a1, a2, a3 = 1, 0.8, 0.5
#
# d1, d2, d3 = a1*np.exp(-t*k1),  a2*np.exp(-t*k2), a3*np.exp(-t*k3)
#
# # plt.plot(t, d1)
# # plt.plot(t, d2)
# # plt.plot(t, d3)
# # plt.plot(t, d1+d2+d3)
#
# # plt.show()
# data = d1+d2+d3
# data += np.random.normal(0, scale=0.05, size=data.shape)
#
# ks = np.logspace(-2, 1, 200)
# X = np.exp(-t[:, None] * ks[None, :])
# #
# # plt.plot(t, X)
# # plt.show()
#
# def calc_W_norms_Lasso(alpha):
#     # mod = lm.Ridge(alpha=alpha, fit_intercept=False)
#     mod = lm.Lasso(alpha=alpha, max_iter=1e6, warm_start=False, fit_intercept=False)
#     mod.fit(X.copy(), data.copy())
#     W = mod.coef_.T
#     fit = mod.predict(X)
#
#     log_R_norm = np.log10(((data - fit) * (data - fit)).sum())  # log10 residual norm
#     log_W_norm = np.log10(np.abs(W).sum())  # log10 smoothing norm - norm of parameters
#
#     return W, fit, log_R_norm, log_W_norm
#
# def calc_W_norms_Ridge(alpha):
#     # mod = lm.Ridge(alpha=alpha, fit_intercept=False)
#     mod = lm.Ridge(alpha=alpha, fit_intercept=False)
#     mod.fit(X.copy(), data.copy())
#     W = mod.coef_.T
#     fit = mod.predict(X)
#
#     log_R_norm = np.log10(((data - fit) * (data - fit)).sum())  # log10 residual norm
#     log_W_norm = np.log10((W*W).sum())  # log10 smoothing norm - norm of parameters
#
#     return W, fit, log_R_norm, log_W_norm
#
# def _func(alpha):
#     W, fit, log_R_norm, log_W_norm = calc_W_norms_Ridge(alpha)
#     return log_R_norm, log_W_norm
#
# # alpha_MC = L_corner_search(_func, end_alphas=(1e-10, 1e-1), run_parallel=False)
#
# # alphas = np.logspace(-7, -1, 40)
# # norms = np.zeros((alphas.shape[0], 2))
# # Ws = np.zeros((alphas.shape[0], ks.shape[0]))
# #
# # for i in range(alphas.shape[0]):
# #     W, fit, log_R_norm, log_W_norm = calc_W_norms_Lasso(alphas[i])
# #
# #     norms[i, 0] = log_R_norm
# #     norms[i, 1] = log_W_norm
# #     Ws[i] = W
# #     print(f'{alphas[i]:.3g}: {log_R_norm:.3g}')
#
# # print(norms)
# #
# # W, fit, log_R_norm, log_W_norm = calc_W_norms(alpha_MC)
# #
# # plt.semilogx(1/ks, W)
# # # plt.vlines(1/k1, W.min(), W.max())
# # # plt.vlines(1/k2, W.min(), W.max())
# # # plt.vlines(1/k3, W.min(), W.max())
# #
# # plt.show()
# # #
# # plt.scatter(norms[:, 0], norms[:, 1])
# # plt.show()
# #
# # plt.plot(t, data)
# # plt.plot(t, fit)
# # plt.show()
#
#
# # def f(x):
# #     time.sleep(1)
# #     return x + 1
# #
# def run_parallel():
#     with multiprocessing.Pool() as pool:
#         print(pool.map(_func, np.logspace(-9, -1, 40)))
# #
# def run_serial():
#     print([_func(a) for a in np.logspace(-9, -1, 40)])
#
# if __name__ == '__main__':
#     alpha_MC, trajectory = L_corner_search(_func, end_alphas=(1e-9, 1e-3), run_parallel=False)
# # #     alpha_MC = L_corner_search(_func, end_alphas=(1e-10, 1e-1), run_parallel=False)
# #
# #     print(trajectory.shape)
#     print(trajectory)
# #
#     plt.scatter(trajectory[:, 1], trajectory[:, 2])
#     plt.show()
#
#     # ret = timeit.timeit(lambda: run_serial(), number=1)
#     # print(f"serial: {ret}")
#
#     # ret = timeit.timeit(lambda: run_parallel(), number=1)
#     # print(f"parallel: {ret}")
#
#     # run_parallel()
#     # run_serial()


# import pyqtgraph.examples
# pyqtgraph.examples.run()

#
# # # import multiprocessing as mp
# # from multiprocessing import Pool
# # import timeit
# # import time
# #
# # def long_run(n):
# #     # s = ''
# #     # for i in range(int(n)):
# #     #     s += str(i)
# #     time.sleep(2)
# #     return n
# #
# #
# # def run_parallel():
# #     pool = Pool()
# #     results = [pool.apply(long_run, args=(x,)) for x in [5e5, 1e5, 7e5, 3e4, 2e4, 1e4, 2e5]]
# #     return results
# #
# # def run_serial():
# #     results = [long_run(x) for x in [5e5, 1e5, 7e5, 3e4, 2e4, 1e4, 2e5]]
# #     return results
# #
# # def simulate( times, K, eps, q_tot, c0, V, I_source):
# #
# #     const = np.log(10)
# #
# #     def dc_dt(c, t):
# #         c_eps = c[:, None] * eps  # hadamard product
# #         c_dot_eps = c_eps.sum(axis=0)
# #
# #         q = c_eps * np.where(c_dot_eps <= 0.001, const - c_dot_eps * const * const / 2,
# #                              (1 - np.exp(-c_dot_eps * const)) / c_dot_eps) * I_source
# #
# #         product = np.matmul(K, q.T[..., None]).squeeze()  # w x n x 1
# #
# #         return q_tot / V * (product.sum(0) - (product[0] + product[-1]) / 2)
# #
# #     return odeint(dc_dt, c0, times)
# #
# # def log_likelihood( params, n_MCR_iter=5):
# #
# #     # optimize spectra for curent C params by MCR-ALS style
# #
# #     phi = self.Phi([params[0], params[1]], lambda_C=400)
# #
# #     if any(phi < 0) or any(phi > 1):
# #         return -np.inf
# #
# #     sigma = 0.01
# #
# #     K = np.asarray([[-phi, self._0],
# #                     [+phi, self._0]])
# #
# #     K = np.transpose(K, (2, 0, 1))
# #     C = np.zeros((self.times.shape[0], K.shape[0]))
# #
# #     eps_est = self.eps_est.copy()
# #
# #     for i in range(n_MCR_iter):
# #         # calc C
# #         C = self.simulate(self.times, K, eps_est, self.q_tot, self.c0, self.V, self.I_source)
# #
# #         # calc ST by lstsq
# #         eps_est = lstsq(C, self.D)[0]
# #
# #         # apply non-negative contraints on spectra
# #         eps_est *= (eps_est > 0)
# #
# #     #         self.calls.append([params[0], self.eps_est])
# #     D_sim = C.dot(self.eps_est)
# #     residuals = self.D - D_sim
# #
# #     # calculate the log of gaussian likelihood
# #     N = 1  # D.size
# #     #         LL = -0.5*N*np.log(2*np.pi*sigma**2) - (0.5/sigma**2) * (residuals**2).sum()
# #
# #     LL = - (0.5 / sigma ** 2) * (residuals ** 2).sum()
# #     return LL
# #
# #
# #
# #
# #
#!/usr/bin/env python3
# rand.py

# import asyncio
# import random
#
# # ANSI colors
# c = (
#     "\033[0m",   # End of color
#     "\033[36m",  # Cyan
#     "\033[91m",  # Red
#     "\033[35m",  # Magenta
# )
#
# async def makerandom(idx: int, threshold: int = 6) -> int:
#     print(c[idx + 1] + f"Initiated makerandom({idx}).")
#     i = random.randint(0, 10)
#     while i <= threshold:
#         print(c[idx + 1] + f"makerandom({idx}) == {i} too low; retrying.")
#         await asyncio.sleep(idx + 1)
#         i = random.randint(0, 10)
#     print(c[idx + 1] + f"---> Finished: makerandom({idx}) == {i}" + c[0])
#     return i
#
# async def main():
#     res = await asyncio.gather(*(makerandom(i, 10 - i - 1) for i in range(3)))
#     return res
#
# if __name__ == "__main__":
#     random.seed(444)
#     r1, r2, r3 = asyncio.run(main())
#     print()
#     print(f"r1: {r1}, r2: {r2}, r3: {r3}")
# import asyncio
#
# from PyQt5 import QtGui, QtWidgets
# from quamash import QEventLoop
#
# app = QtWidgets.QApplication([])
# loop = QEventLoop(app)
# asyncio.set_event_loop(loop)
#
# display = QtWidgets.QLCDNumber()
# display.setWindowTitle('Stopwatch')
#
# display.show()
#
# async def update_time():
#     value = 10000
#     while True:
#         display.display(value)
#         await asyncio.sleep(0.01)
#         value += 1
#
# asyncio.ensure_future(update_time())
#
# loop.run_forever()


from concurrent.futures import ThreadPoolExecutor
from time import sleep
import threading

def return_after_5_secs(message):
    print('print from another task\n')
    sleep(1)
    print(threading.current_thread())
    # print('after sleep')
    return message


def done(future):
    if future.done():
        print('completed')
        print(threading.current_thread())
        print(future.result())

pool = ThreadPoolExecutor(3)
future = pool.submit(return_after_5_secs, ("hello"))
future.add_done_callback(done)

print('another program running')
print(threading.current_thread())

# print('asdasd', flush=True)
# print(future.done())
# sleep(2)
# print(future.done())
# print(future.result())


# #
# #
# #
# #
# # if __name__ == '__main__':
# #
# #     # res = run_parallel()
# #     # res = run_serial()
# #
# #
# #     ret = timeit.timeit(lambda: run_serial(), number=2) / 2
# #     print(f"serial: {ret}")
# #
# #     ret = timeit.timeit(lambda: run_parallel(), number=2) / 2
# #     print(f"parallel: {ret}")
# #
# #
# #
# #
# #
# #
# #
# #
# #
