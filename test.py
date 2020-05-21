import numpy as np  # import numpy package (abbreviation stands for Numerical Python)
import matplotlib.pyplot as plt  # we plot graphs with this library
import math

from matplotlib import cm
import matplotlib as mpl
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from matplotlib.ticker import *


# works for numbers and array of numbers
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


def find_nearest(array, value):
    idx = find_nearest_idx(array, value)
    return array[idx]


MAJOR_TICK_DIRECTION = 'in'  # in, out or inout
MINOR_TICK_DIRECTION = 'in'  # in, out or inout
MARKER_SIZE = 20

Z_NUM = '$Z$-$\\bf{Xa}$'
E_NUM = '$E$-$\\bf{Xa}$'
HL_NUM = '$\\bf{Xb}$'

# EPS_LABEL = '$\\varepsilon$ ($10^4$ M$^{-1}$ cm$^{-1}$)'
WL_LABEL = 'Wavelength (nm)'

def eps_label(factor):
    num = np.log10(1/factor).astype(int)
    return f'$\\varepsilon$ ($10^{num}$ M$^{-1}$ cm$^{-1}$)'

X_SIZE, Y_SIZE = 5, 4

COLORS = ['blue', 'red', 'green', 'black']


def setup_twin_x_axis(ax, y_label="Molar spectral flux ($10^{-10}\\ mol\\ s^{-1}\\ nm^{-1}$)",
                      x_label=None, ylim=(None, None), y_major_locator=None, y_minor_locator=None,
                      keep_zero_aligned=True):
    ax2 = ax.twinx()

    ax2.tick_params(which='major', direction=MAJOR_TICK_DIRECTION)
    ax2.tick_params(which='minor', direction=MINOR_TICK_DIRECTION)

    if y_major_locator:
        ax2.yaxis.set_major_locator(y_major_locator)

    if y_minor_locator:
        ax2.yaxis.set_minor_locator(y_minor_locator)

    ax2.set_ylabel(y_label)

    if keep_zero_aligned and ylim[0] is None and ylim[1] is not None:
        # a = bx/(x-1)
        ax1_ylim = ax1.get_ylim()
        x = -ax1_ylim[0] / (ax1_ylim[1] - ax1_ylim[0])  # position of zero in ax1, from 0, to 1
        a = ylim[1] * x / (x - 1)  # calculates the ylim[0] so that zero position is the same for both axes
        ax2.set_ylim(a, ylim[1])

    elif ylim[0] is not None:
        ax2.set_ylim(ylim)

    return ax2


def setup_wavenumber_axis(ax, x_label="Wavenumber ($10^{4}$ cm$^{-1}$)",
                          x_major_locator=None, x_minor_locator=AutoMinorLocator(5), factor=1e3):
    secondary_ax = ax.secondary_xaxis('top', functions=(lambda x: factor / x, lambda x: 1 / (factor * x)))

    secondary_ax.tick_params(which='major', direction=MAJOR_TICK_DIRECTION)
    secondary_ax.tick_params(which='minor', direction=MINOR_TICK_DIRECTION)

    if x_major_locator:
        secondary_ax.xaxis.set_major_locator(x_major_locator)

    if x_minor_locator:
        secondary_ax.xaxis.set_minor_locator(x_minor_locator)

    secondary_ax.set_xlabel(x_label)

    return secondary_ax


def set_main_axis(ax, x_label=WL_LABEL, y_label="Absorbance", xlim=(None, None), ylim=(None, None),
                  x_major_locator=None, x_minor_locator=None, y_major_locator=None, y_minor_locator=None):
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    if xlim[0] is not None:
        ax.set_xlim(xlim)
    if ylim[0] is not None:
        ax.set_ylim(ylim)

    if x_major_locator:
        ax.xaxis.set_major_locator(x_major_locator)

    if x_minor_locator:
        ax.xaxis.set_minor_locator(x_minor_locator)

    if y_major_locator:
        ax.yaxis.set_major_locator(y_major_locator)

    if y_minor_locator:
        ax.yaxis.set_minor_locator(y_minor_locator)

    ax.tick_params(axis='both', which='major', direction=MAJOR_TICK_DIRECTION)
    ax.tick_params(axis='both', which='minor', direction=MINOR_TICK_DIRECTION)

path = r'C:\Users\Dominik\Documents\MUNI\Organic Photochemistry\Projects\2019-Bilirubin project\UV-VIS\QY measurement\Photodiode\new setup'

fnames = [[r'\Z 350p\cut.txt', r'\E 350p\cut.txt',  r'\E 500p\cut.txt'],
         [r'\reactor LEDs\HL 355\cut.txt', r'\reactor LEDs\Z 355\cut.txt', r'\reactor LEDs\Z 490\cut.txt']]

if __name__ == '__main__':
    fig, ax1 = plt.subplots(1, 1, figsize=(1 * X_SIZE, Y_SIZE))

    data = np.genfromtxt(path + fnames[1][1], delimiter='\t')
    #     data = data[:, :-10]
    t = data[0, 1:]
    x = data[1:, 0]
    y = data[1:, 1:]

    step = 2
    cmap = cm.get_cmap('jet_r')
    # cmap = cm.get_cmap('gist_rainbow')

    # norm = mpl.colors.LogNorm(vmin=10,vmax=t[-1], clip=True)
    # norm = mpl.colors.PowerNorm(gamma=0.3, vmin=0,vmax=t[-1], clip=True)
    linthresh = 50
    norm = mpl.colors.SymLogNorm(vmin=0, vmax=t[-1], linscale=0.5, linthresh=linthresh, base=10, clip=True)

    set_main_axis(ax1, xlim=(230, 590), x_minor_locator=AutoMinorLocator(5), y_minor_locator=None)
    secax = setup_wavenumber_axis(ax1)

    ts_empf = (0, 40, 1500, t[-1])
    tsb_idxs = find_nearest_idx(t, ts_empf)
    ts_real = t[tsb_idxs]

    n_spectra = 30
    x_space = np.linspace(0, 1, n_spectra, endpoint=True, dtype=np.float64)

    t_idx_space = find_nearest_idx(t, norm.inverse(x_space))
    t_idx_space = np.sort(np.asarray(list(set(t_idx_space).union(set(tsb_idxs)))))

    for i in t_idx_space:
        x_real = norm(t[i])
        x_real = 0 if np.ma.is_masked(x_real) else x_real
        ax1.plot(x, y[:, i], color=cmap(x_real),
                 lw=2 if i in tsb_idxs else 0.6,
                 alpha=1 if i in tsb_idxs else 0.5,
                 zorder=5 if i in tsb_idxs else 0)


    # for j in range(0, t.shape[0], step):
    #     idx = norm(t[j])
    #     #     print(idx)
    #     idx = 0 if np.ma.is_masked(idx) else idx
    #     ax1.plot(x, y[:, j], color=cmap(idx),
    #              lw=2 if j in tsb_idxs else 0.3,
    #              alpha=1 if j in tsb_idxs else 0.6,
    #              zorder=5 if j in tsb_idxs else 0)

    # cbar, cbax = add_colorbar(ax1, cmap, norm, major_locator=FixedLocator(t[tsb_idxs]))
    # cbar, cbax = add_colorbar(ax1, cmap, norm, major_locator=Fixe(linthresh=100, base=10))

    cbaxes = inset_axes(ax1, width="3%", height="80%", loc='center right', borderpad=5.5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbaxes, orientation='vertical',
                        format=mpl.ticker.ScalarFormatter(),
                        # ticks=FixedLocator(ticks),
                        label="Time (s)")

    cbaxes.invert_yaxis()
    # cbaxes.minorticks_on()
    # l = FixedLocator([0, 10, 100, 1000])
    # cbaxes.yaxis.set_major_locator(l)

    minor_ticks = cbar._locate(
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 200, 300, 400, 500, 600, 700, 800, 900, 2000, 3000, 4000, 5000])
    cbar.ax.yaxis.set_ticks(minor_ticks, minor=True)

    major_ticks = np.sort(np.hstack((np.asarray([1e2, 1e3]), ts_real)))

    transformed = cbar._locate(major_ticks)

    cbaxes.yaxis.set_ticks(transformed, minor=False)
    cbaxes.set_yticklabels([f'{int(num)}' for num in major_ticks])

    for ytick, ytick_label, t in zip(cbaxes.yaxis.get_major_ticks(), cbaxes.get_yticklabels(), major_ticks):
        if t in ts_real:
            color = cmap(norm(t))
            ytick_label.set_color(color)
            ytick_label.set_fontweight('bold')
            ytick.tick2line.set_color(color)
            ytick.tick2line.set_markersize(5)
            # ytick.tick2line.set_markeredgewidth(2)

    plt.savefig('test_plot.png', dpi=500, bbox_inches='tight')

    plt.show()








# # import multiprocessing as mp
# from multiprocessing import Pool
# import timeit
# import time
#
# def long_run(n):
#     # s = ''
#     # for i in range(int(n)):
#     #     s += str(i)
#     time.sleep(2)
#     return n
#
#
# def run_parallel():
#     pool = Pool()
#     results = [pool.apply(long_run, args=(x,)) for x in [5e5, 1e5, 7e5, 3e4, 2e4, 1e4, 2e5]]
#     return results
#
# def run_serial():
#     results = [long_run(x) for x in [5e5, 1e5, 7e5, 3e4, 2e4, 1e4, 2e5]]
#     return results
#
# def simulate( times, K, eps, q_tot, c0, V, I_source):
#
#     const = np.log(10)
#
#     def dc_dt(c, t):
#         c_eps = c[:, None] * eps  # hadamard product
#         c_dot_eps = c_eps.sum(axis=0)
#
#         q = c_eps * np.where(c_dot_eps <= 0.001, const - c_dot_eps * const * const / 2,
#                              (1 - np.exp(-c_dot_eps * const)) / c_dot_eps) * I_source
#
#         product = np.matmul(K, q.T[..., None]).squeeze()  # w x n x 1
#
#         return q_tot / V * (product.sum(0) - (product[0] + product[-1]) / 2)
#
#     return odeint(dc_dt, c0, times)
#
# def log_likelihood( params, n_MCR_iter=5):
#
#     # optimize spectra for curent C params by MCR-ALS style
#
#     phi = self.Phi([params[0], params[1]], lambda_C=400)
#
#     if any(phi < 0) or any(phi > 1):
#         return -np.inf
#
#     sigma = 0.01
#
#     K = np.asarray([[-phi, self._0],
#                     [+phi, self._0]])
#
#     K = np.transpose(K, (2, 0, 1))
#     C = np.zeros((self.times.shape[0], K.shape[0]))
#
#     eps_est = self.eps_est.copy()
#
#     for i in range(n_MCR_iter):
#         # calc C
#         C = self.simulate(self.times, K, eps_est, self.q_tot, self.c0, self.V, self.I_source)
#
#         # calc ST by lstsq
#         eps_est = lstsq(C, self.D)[0]
#
#         # apply non-negative contraints on spectra
#         eps_est *= (eps_est > 0)
#
#     #         self.calls.append([params[0], self.eps_est])
#     D_sim = C.dot(self.eps_est)
#     residuals = self.D - D_sim
#
#     # calculate the log of gaussian likelihood
#     N = 1  # D.size
#     #         LL = -0.5*N*np.log(2*np.pi*sigma**2) - (0.5/sigma**2) * (residuals**2).sum()
#
#     LL = - (0.5 / sigma ** 2) * (residuals ** 2).sum()
#     return LL
#
#
#
#
#
#
#
#
#
# if __name__ == '__main__':
#
#     # res = run_parallel()
#     # res = run_serial()
#
#
#     ret = timeit.timeit(lambda: run_serial(), number=2) / 2
#     print(f"serial: {ret}")
#
#     ret = timeit.timeit(lambda: run_parallel(), number=2) / 2
#     print(f"parallel: {ret}")
#
#
#
#
#
#
#
#
#
