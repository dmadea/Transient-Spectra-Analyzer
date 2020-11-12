
import numpy as np
from scipy.linalg import svd

import os

import matplotlib.pyplot as plt

from misc import crop_data, find_nearest, find_nearest_idx
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from sklearn.decomposition import NMF
from sklearn.decomposition import FastICA
import matplotlib as mpl
import matplotlib.colors as c
from numpy import ma


WL_LABEL = 'Wavelength / nm'
WN_LABEL = "Wavenumber / $10^{4}$ cm$^{-1}$"
CONC_LABEL = 'Concentration / $\mu$mol L$^{-1}$'

plt.rcParams.update({'font.size': 12})

# default values
# plt.rcParams.update({'xtick.major.size': 3.5, 'ytick.major.size': 3.5})
# plt.rcParams.update({'xtick.minor.size': 2, 'ytick.minor.size': 2})
# plt.rcParams.update({'xtick.major.width': 0.8, 'ytick.major.width': 0.8})
# plt.rcParams.update({'xtick.minor.width': 0.6, 'ytick.minor.width': 0.6})

plt.rcParams.update({'xtick.major.size': 5, 'ytick.major.size': 5})
plt.rcParams.update({'xtick.minor.size': 2.5, 'ytick.minor.size': 2.5})
plt.rcParams.update({'xtick.major.width': 1, 'ytick.major.width': 1})
plt.rcParams.update({'xtick.minor.width': 0.8, 'ytick.minor.width': 0.8})
mpl.rcParams['hatch.linewidth'] = 0.8  # hatch linewidth

LEGEND_FONT_SIZE = 10
MAJOR_TICK_DIRECTION = 'in'  # in, out or inout
MINOR_TICK_DIRECTION = 'in'  # in, out or inout
from matplotlib.ticker import *

def eps_label(factor):
    num = np.log10(1 / factor).astype(int)
    return f'$\\varepsilon$ / $(10^{{{num}}}$ L mol$^{{-1}}$ cm$^{{-1}})$'


X_SIZE, Y_SIZE = 5, 4.5
dA_unit = '$\Delta A$ / $10^{-3}$'

COLORS = ['blue', 'red', 'green', 'orange', 'black', 'yellow']
COLORS_gradient = ['blue', 'lightblue']


def setup_wavenumber_axis(ax, x_label=WN_LABEL,
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


def plot_traces_onefig_ax(ax, D, D_fit, times, wavelengths, wls=[355, 400, 450, 500, 550], marker_size=10,
                          marker_linewidth=1,
                          marker_facecolor='white', alpha=0.8,
                          linthresh=1, linscale=1, colors=None, D_mul_factor=1e3, legend_spacing=0.2, lw=1.5,
                          legend_loc='lower right', y_label=dA_unit, x_label='Time / ps'):

    n = wls.__len__()
    t = times
    norm = mpl.colors.SymLogNorm(vmin=t[0], vmax=t[-1], linscale=linscale, linthresh=linthresh, base=10, clip=True)

    set_main_axis(ax, xlim=(t[0], t[-1]), y_label=y_label, x_label=x_label,
                  y_minor_locator=None, x_minor_locator=None)

    ax.plot(t, np.zeros_like(t), ls='--', color='black', lw=1)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] if colors is None else colors

    for i in range(n):
        color_points = c.to_rgb(colors[i])
        color_line = np.asarray(color_points) * 0.7

        idx = find_nearest_idx(wavelengths, wls[i])
        ax.scatter(t, D[:, idx] * D_mul_factor, edgecolor=color_points, facecolor=marker_facecolor,
                   alpha=alpha, marker='o', s=marker_size, linewidths=marker_linewidth)

        ax.scatter([], [], edgecolor=color_points, facecolor=marker_facecolor,
                   alpha=alpha, marker='o', label=f'{wls[i]} nm', s=marker_size * 2, linewidths=marker_linewidth)
        ax.plot(t, D_fit[:, idx] * D_mul_factor, lw=lw, color=color_line)

    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    ax.set_xscale('symlog', subs=[2, 3, 4, 5, 6, 7, 8, 9], linscale=linscale, linthresh=linthresh)

    ax.xaxis.set_minor_locator(MinorSymLogLocator(linthresh))

    ax.xaxis.set_major_formatter(ScalarFormatter())

    l = ax.legend(loc=legend_loc, frameon=False, labelspacing=legend_spacing)
    for text, color in zip(l.get_texts(), colors):
        text.set_color(color)

    d = 0.015
    tilt = 0.4
    ax.set_ylim(ax.get_ylim())

    sep = norm(linthresh)
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot([sep - d * tilt, sep + d * tilt], [-d * 0.8, +d * 0.8], **kwargs)
    ax.plot([sep - d * tilt, sep + d * tilt], [1 - d * 0.8, 1 + d * 0.8], **kwargs)
    ax.vlines(linthresh, ax.get_ylim()[0], ax.get_ylim()[1], ls='dotted', lw=1, color='k', zorder=10)

    ax.set_axisbelow(False)


def plot_spectra_ax(ax, data, times=[0, 100, 1, 5, 100, 500], zero_reg=None, z_unit=dA_unit, D_mul_factor=1e3,
                    legend_spacing=0.05, area_plot_data=(None, None), area_plot_data_red=(None, None),
                    colors=None, fill_alpha=0.3, lw=1.5, darkens_factor=1, cmap=None, columnspacing=2,
                    legend_loc='lower right',
                    legend_ncol=2, ylim=None):
    D = data.D * D_mul_factor
    if zero_reg is not None:
        cut_idxs = find_nearest_idx(data.wavelengths, zero_reg)
        D[:, cut_idxs[0]:cut_idxs[1]] = np.nan

    set_main_axis(ax, y_label=z_unit,
                  x_minor_locator=AutoMinorLocator(10), y_minor_locator=None,
                  xlim=(data.wavelengths[0], data.wavelengths[-1]))
    _ = setup_wavenumber_axis(ax, x_major_locator=MultipleLocator(0.5))

    t_idxs = find_nearest_idx(data.times, times)

    if cmap is 'redorange':
        cmap = mpl.colors.LinearSegmentedColormap.from_list('redorange', ['red', 'orange', (1, 1, 0, 1), (0, 1, 0, 1)],
                                                            N=len(times))
    elif cmap is 'blue':
        cmap = mpl.colors.LinearSegmentedColormap.from_list('ccc', [(0, 1, 0, 1), 'cyan', 'blue', 'darkblue'],
                                                            N=len(times))
    else:
        cmap = cm.get_cmap('gist_rainbow', t_idxs.shape[0] / 0.75) if cmap is None else cmap

    ax.plot(data.wavelengths, np.zeros_like(data.wavelengths), ls='--', color='black', lw=1)
    if ylim is not None:
        ax.set_ylim(ylim)

    for i in range(t_idxs.shape[0]):
        if colors is None:
            color = np.asarray(c.to_rgb(cmap(i))) * darkens_factor
            color[color > 1] = 1
        else:
            color = colors[i]

        ax.plot(data.wavelengths, D[t_idxs[i]], color=color, lw=lw, label=f'${times[i]:.3g}$ ps')

    if area_plot_data[0] is not None:
        ax.fill_between(area_plot_data[0], area_plot_data[1], color='blue', alpha=fill_alpha)

    if area_plot_data_red[0] is not None:
        ax.fill_between(area_plot_data_red[0], area_plot_data_red[1], color='red', alpha=fill_alpha)

    l = ax.legend(loc=legend_loc, frameon=False, labelspacing=legend_spacing, ncol=legend_ncol, handletextpad=0,
                  handlelength=0,
                  columnspacing=columnspacing)
    for i, text in enumerate(l.get_texts()):
        text.set_ha('right')
        text.set_color(cmap(i))

    ax.set_axisbelow(False)
    ax.yaxis.set_ticks_position('both')


def get_sym_space(vmin, vmax, n):
    """Return evenly spaced tics that are however symetric around zero (without zero!) so that tics are: (-t_i, t_i+1)"""
    raw_step = (vmax - vmin) / (n - 1)

    pos_steps = int(np.ceil((vmax - 0.5 * raw_step) / raw_step))
    neg_steps = int(np.ceil((abs(vmin) - 0.5 * raw_step) / raw_step))

    return np.linspace((neg_steps + 0.5) * -raw_step, (pos_steps + 0.5) * raw_step, pos_steps + neg_steps + 2)


def plot_data_ax(fig, ax, matrix, times, wavelengths, symlog=True, t_unit='ps',
                 z_unit=dA_unit, cmap='diverging', z_lim=(None, None),
                 t_lim=(None, None), w_lim=(None, None), linthresh=10, linscale=1, D_mul_factor=1e3,
                 y_major_formatter=ScalarFormatter(),
                 x_minor_locator=AutoMinorLocator(10), n_levels=30, plot_countours=True,
                 colorbar_locator=MultipleLocator(50),
                 diverging_white_cmap_tr=0.98, hatch='/////', colorbar_aspect=35, add_wn_axis=True):
    """data is individual dataset"""

    # assert type(data) == Data

    t_lim = (times[0] if t_lim[0] is None else t_lim[0], times[-1] if t_lim[1] is None else t_lim[1])
    w_lim = (
        wavelengths[0] if w_lim[0] is None else w_lim[0], wavelengths[-1] if w_lim[1] is None else w_lim[1])

    D = matrix.copy() * D_mul_factor

    zmin = np.min(D) if z_lim[0] is None else z_lim[0]
    zmax = np.max(D) if z_lim[1] is None else z_lim[1]

    if z_lim[0] is not None:
        D[D < zmin] = zmin

    if z_lim[1] is not None:
        D[D > zmax] = zmax

    register_div_cmap(zmin, zmax)
    register_div_white_cmap(zmin, zmax, diverging_white_cmap_tr)

    x, y = np.meshgrid(wavelengths, times)  # needed for pcolormesh to correctly scale the image

    # plot data matrix D

    set_main_axis(ax, xlim=w_lim, ylim=t_lim, x_label=r'Wavelength / nm', y_label=f'Time delay / {t_unit}',
                  x_minor_locator=x_minor_locator, y_minor_locator=None)
    if add_wn_axis:
        w_ax = setup_wavenumber_axis(ax, x_major_locator=MultipleLocator(0.5))
        w_ax.tick_params(which='minor', direction='out')
        w_ax.tick_params(which='major', direction='out')

    #     ax.set_facecolor((0.8, 0.8, 0.8, 1))
    if ma.is_masked(D):  # https://stackoverflow.com/questions/41664850/hatch-area-using-pcolormesh-in-basemap
        m_idxs = np.argwhere(D.mask[0] > 0).squeeze()
        wl_range = [wavelengths[m_idxs[0] - 1], wavelengths[m_idxs[-1] + 1]]
        ax.fill_between(wl_range, [t_lim[0], t_lim[0]], [t_lim[1], t_lim[1]], facecolor="none",
                        hatch=hatch, edgecolor="k", linewidth=0.0)

    #     mappable = ax.pcolormesh(x, y, D, cmap=cmap, vmin=zmin, vmax=zmax)
    levels = get_sym_space(zmin, zmax, n_levels)
    mappable = ax.contourf(x, y, D, cmap=cmap, vmin=zmin, vmax=zmax, levels=levels, antialiased=True)

    if plot_countours:
        cmap_colors = cm.get_cmap(cmap)
        colors = cmap_colors(np.linspace(0, 1, n_levels + 1))
        colors *= 0.45  # plot contours as darkens colors of colormap, blue -> darkblue, white -> gray ...
        ax.contour(x, y, D, colors=colors, levels=levels, antialiased=True, linewidths=0.2,
                   alpha=1, linestyles='-')

    ax.invert_yaxis()

    ax.tick_params(which='major', direction='out')
    ax.tick_params(which='minor', direction='out')

    ax.set_axisbelow(False)

    fig.colorbar(mappable, ax=ax, label=z_unit, orientation='vertical', aspect=colorbar_aspect, pad=0.025,
                 ticks=colorbar_locator)

    if symlog:
        ax.set_yscale('symlog', subs=[2, 3, 4, 5, 6, 7, 8, 9], linscale=linscale, linthresh=linthresh)
        ax.yaxis.set_minor_locator(MinorSymLogLocator(linthresh))

    if y_major_formatter:
        ax.yaxis.set_major_formatter(y_major_formatter)


def plot_SADS_ax(ax, wls, SADS, labels=None, zero_reg=(None, None), z_unit=dA_unit, D_mul_factor=1e3,
                 legend_spacing=0.2, legend_ncol=1, colors=None, lw=1.5, show_legend=False,
                 area_plot_data=(None, None), area_plot_color='violet', area_plot_data2=(None, None),
                 area_plot_color2='blue',
                 area_plot_alpha=0.2, area_plot_alpha2=0.1, w_lim=(None, None)):
    _SADS = SADS.copy() * D_mul_factor
    if zero_reg[0] is not None:
        cut_idxs = find_nearest_idx(wls, zero_reg)
        _SADS[cut_idxs[0]:cut_idxs[1]] = np.nan

    w_lim = (wls[0] if w_lim[0] is None else w_lim[0], wls[-1] if w_lim[1] is None else w_lim[1])

    set_main_axis(ax, y_label=z_unit, xlim=w_lim,
                  x_minor_locator=AutoMinorLocator(10), x_major_locator=MultipleLocator(100), y_minor_locator=None)
    _ = setup_wavenumber_axis(ax, x_major_locator=MultipleLocator(0.5))

    cmap = cm.get_cmap('gist_rainbow', _SADS.shape[1] / 0.75)

    ax.axhline(0, ls='--', color='black', lw=1)
    labels = list('ABCDEFGHIJ') if labels is None else labels

    for i in range(_SADS.shape[1]):
        if colors is None:
            color = np.asarray(c.to_rgb(cmap(i))) * 0.9
            color[color > 1] = 1
        else:
            color = colors[i]

        ax.plot(wls, _SADS[:, i], color=color, lw=lw, label=labels[i])

    if area_plot_data[0] is not None:
        ax.fill_between(area_plot_data[0], area_plot_data[1], color=area_plot_color, alpha=area_plot_alpha, zorder=0)
    if area_plot_data2[0] is not None:
        ax.fill_between(area_plot_data2[0], area_plot_data2[1], color=area_plot_color2, alpha=area_plot_alpha2,
                        zorder=-10)

    if show_legend:
        ax.legend(frameon=False, labelspacing=legend_spacing, ncol=legend_ncol)
    ax.set_axisbelow(False)
    ax.yaxis.set_ticks_position('both')


def register_div_white_cmap(zmin, zmax, treshold=0.98):
    """Registers `diverging` diverging color map just suited for data.

    With extra white space at around zero - for filled countours colormaps to ensure zero is white."""

    diff = zmax - zmin
    w = np.abs(zmin / diff)  # white color point set to zero z value

    tr = treshold

    _cdict = {'red': ((0.0, 0.0, 0.0),
                      (w / 2, 0.0, 0.0),
                      (w * tr, 1.0, 1.0),
                      (w + (1 - w) * (1 - tr), 1.0, 1.0),
                      (w + (1 - w) / 3, 1.0, 1.0),
                      (w + (1 - w) * 2 / 3, 1.0, 1.0),
                      (1.0, 0.3, 0.3)),

              'green': ((0.0, 0, 0),
                        (w / 2, 0.0, 0.0),
                        (w * tr, 1.0, 1.0),
                        (w + (1 - w) * (1 - tr), 1.0, 1.0),
                        (w + (1 - w) / 3, 1.0, 1.0),
                        (w + (1 - w) * 2 / 3, 0.0, 0.0),
                        (1.0, 0.0, 0.0)),

              'blue': ((0.0, 0.3, 0.3),
                       (w / 2, 1.0, 1.0),
                       (w * tr, 1.0, 1.0),
                       (w + (1 - w) * (1 - tr), 1.0, 1.0),
                       (w + (1 - w) / 3, 0.0, 0.0),
                       (w + (1 - w) * 2 / 3, 0.0, 0.0),
                       (1.0, 0.0, 0.0))
              }

    custom_cmap = LinearSegmentedColormap('diverging_white_tr', _cdict)
    cm.register_cmap('diverging_white_tr', custom_cmap)


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


class LFP_matrix(object):

    @classmethod
    def from_value_matrix(cls, value_matrix, times, wavelengths, filename='', name='', mask=[]):
        m = cls()
        m.Y = value_matrix
        m.times = times
        m.wavelengths = wavelengths
        m.filename = filename
        m.name = name
        m.mask = mask
        m.SVD()
        m._set_D()
        return m

    @property
    def Mask(self):
        return self._mask

    @Mask.setter
    def Mask(self, value):  # true or false
        self._mask = value
        self._set_D()

    @property
    def SVD_filter(self):
        return self._SVD_filter

    @SVD_filter.setter
    def SVD_filter(self, value):  # true or false
        self._SVD_filter = value
        self._set_D()

    @property
    def ICA_filter(self):
        return self._ICA_filter

    @ICA_filter.setter
    def ICA_filter(self, value):  # true or false
        self._ICA_filter = value
        self._set_D()

    def _set_D(self):
        if self.Yr is None:
            self.Yr = self.Y
        self.D = self.Yr.copy() if self._SVD_filter else self.Y.copy()
        if self._ICA_filter:
            self.D -= self.ICA_subtr_mat
        if self._mask:
            self.apply_mask(self.D)

    def get_factored_matrix(self):
        """Returns the current/factored matrix as a LFP_Matrix object."""

        if self._mask:
            self.Mask = False

        m = LFP_matrix.from_value_matrix(self.D.copy(), self.times.copy(), self.wavelengths.copy(),
                                         filename=self.filename,
                                         name=self.name, mask=self.mask.copy())
        return m

    def __init__(self, data=None, filename=None, name=None):

        self.wavelengths = None  # dim = w
        self.times = None  # dim = t
        self.Y = None  # dim = t x w   # original data
        self.original_data_matrix = None

        self._SVD_filter = False
        self._ICA_filter = False
        self._mask = False
        self.ICA_components = 5

        self.D = None  # matrix to be plotted - factored matrix if SVD filter or ICA filter is True, else self.Y

        if data is not None:
            self.wavelengths = data[0, 1:]  # first row without first column
            self.times = data[1:, 0]  # first column without the first row

            self.Y = data[1:, 1:]  # original matrix
            self.original_data_matrix = data

        self.D = self.Y

        self.filename = filename
        self.name = name

        # svd matrices k = min(t, w)
        self.U = None  # dim = (t x k)
        self.S = None  # !! this is only 1D array of singular values, not diagonal matrix
        self.V_T = None  # dim = (k x w)

        self.C_ICA = None
        self.ST_ICA = None

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

        self.Yr = self.Y  # reconstructed data matrix after data reduction from SVD
        self.ICA_subtr_mat = 0  # matrix for subtraction after ICA comp. removal

        self.Y_fit = None

        self.C_fit = None
        self.ST_fit = None
        self.D_fit = None
        self.C_COH = None
        self.ST_COH = None

        self.E = None  # residuals

        self.times_fine = None
        self.C_fine = None

        self.mask = []

        self.SVD()

    def add_masked_area(self, t0=None, t1=None, w0=None, w1=None):

        t0_idx = find_nearest_idx(self.times, t0) if t0 is not None else 0
        t1_idx = find_nearest_idx(self.times, t1) + 1 if t1 is not None else self.Y.shape[0]

        w0_idx = find_nearest_idx(self.wavelengths, w0) if w0 is not None else 0
        w1_idx = find_nearest_idx(self.wavelengths, w1) + 1 if w1 is not None else self.Y.shape[1]

        self.mask.append([t0_idx, t1_idx, w0_idx, w1_idx])

    def clear_mask(self):
        self.mask.clear()

    def apply_mask(self, matrix):
        if len(self.mask) == 0:
            return

        assert matrix.shape[0] == self.times.shape[0] and matrix.shape[1] == self.wavelengths.shape[0]

        for (t0_idx, t1_idx, w0_idx, w1_idx) in self.mask:
            matrix[t0_idx:t1_idx, w0_idx:w1_idx] = np.nan


    @classmethod
    def construct_test_matrix(cls, noise_intensity=0.1):

        def gauss(x, mu, sigma):
            return np.exp(- (x - mu) * (x - mu) / (2 * sigma * sigma))

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

        from gui_console import Console

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
        self.run_ICA()

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

    def save_factored_matrix(self, output_dir='.\\', delimiter='\t', encoding='utf8', t0=None, t1=None, w0=None, w1=None):
        if self.filename is None:
            return

        _, fname = os.path.split(self.filename)
        name, ext = os.path.splitext(fname)

        fpath = os.path.join(output_dir, f'{name}_factored{ext}')

        self.save_matrix(self.D, fname=fpath, delimiter=delimiter, encoding=encoding, t0=t0, t1=t1, w0=w0, w1=w1)

    def save_original_matrix(self, output_dir='.\\', delimiter='\t', encoding='utf8', t0=None, t1=None, w0=None, w1=None):
        if self.filename is None:
            return

        _, fname = os.path.split(self.filename)
        name, ext = os.path.splitext(fname)

        fpath = os.path.join(output_dir, f'{name}{ext}')

        self.save_matrix(self.Y, fname=fpath, delimiter=delimiter, encoding=encoding, t0=t0, t1=t1, w0=w0, w1=w1)

    def save_matrix(self, D=None, fname='output.txt', delimiter='\t', encoding='utf8', t0=None, t1=None, w0=None, w1=None):
        # cut data if necessary

        t_idx_start = find_nearest_idx(self.times, t0) if t0 is not None else 0
        t_idx_end = find_nearest_idx(self.times, t1) + 1 if t1 is not None else self.D.shape[0]

        wl_idx_start = find_nearest_idx(self.wavelengths, w0) if w0 is not None else 0
        wl_idx_end = find_nearest_idx(self.wavelengths, w1) + 1 if w1 is not None else self.D.shape[1]

        D = self.D if D is None else D

        # crop the data if necessary
        D_crop = D[t_idx_start:t_idx_end, wl_idx_start:wl_idx_end]
        times_crop = self.times[t_idx_start:t_idx_end]
        wavelengths_crop = self.wavelengths[wl_idx_start:wl_idx_end]

        mat = np.vstack((wavelengths_crop, D_crop))
        buffer = delimiter + delimiter.join(f"{num}" for num in times_crop) + '\n'
        buffer += '\n'.join(delimiter.join(f"{num}" for num in row) for row in mat.T)

        with open(fname, 'w', encoding=encoding) as f:
            f.write(buffer)

    # non-negative matrix factorization solution
    def get_NMF_solution(self, n_components=3, random_state=0):
        model = NMF(n_components=n_components, init='random', random_state=random_state)
        _D = self.Y.copy()
        _D[_D < 0] = 0
        C = model.fit_transform(_D)
        ST = model.components_
        return C, ST

    @staticmethod
    def _fEFA(matrix, sing_values_num=7, points=200):
        """Performs forward Evolving factor analysis over time domain on the current matrix."""

        t_idxs = np.linspace(int(matrix.shape[0] / points), matrix.shape[0] - 1, num=points).astype(int)
        sing_values = np.ones((points, sing_values_num), dtype=np.float64) * np.nan
        fEFA_VTs = np.ones((points, sing_values_num, matrix.shape[1])) * np.nan
        # self.fEFA_Us = np.ones((points, sing_values_num, self.D.shape[0])) * np.nan

        for i in range(points):
            U, S, V_T = svd(matrix[:t_idxs[i], :], full_matrices=False, lapack_driver='gesdd')
            n = int(min(sing_values_num, S.shape[0]))
            sing_values[i, :n] = S[:n]

            fEFA_VTs[i, :n] = V_T[:n, :]

        return sing_values, fEFA_VTs, t_idxs

    def fEFA(self, sing_values_num=7, points=200):
        """Performs forward Evolving factor analysis over time domain on the current matrix."""

        t_idxs = np.linspace(int(self.times.shape[0] / points), self.times.shape[0] - 1, num=points).astype(int)
        self.sing_values = np.ones((points, sing_values_num), dtype=np.float64) * np.nan
        self.fEFA_VTs = np.ones((points, sing_values_num, self.D.shape[1])) * np.nan
        # self.fEFA_Us = np.ones((points, sing_values_num, self.D.shape[0])) * np.nan

        for i in range(points):
            U, S, V_T = svd(self.D[:t_idxs[i], :], full_matrices=False, lapack_driver='gesdd')
            n = int(min(sing_values_num, S.shape[0]))
            self.sing_values[i, :n] = S[:n]

            self.fEFA_VTs[i, :n] = V_T[:n, :]
            # self.fEFA_Us[i, :n] = U[:, :n].T

        times = self.times[t_idxs]

        for i in range(sing_values_num):
            plt.plot(times, self.sing_values[:, i], label=f'{i+1}')
        plt.xlabel('Time / s')
        plt.ylabel('Singular value')
        plt.title('Evolving factor analysis')
        plt.yscale('log')
        plt.legend()

        plt.show()

    def fEFA_plot_VTs(self, component=1, norm=False):
        if not hasattr(self, 'fEFA_VTs'):
            return

        assert self.sing_values.shape[0] == self.fEFA_VTs.shape[0]

        n = self.sing_values.shape[0]

        cmap = cm.get_cmap('jet', n)

        for i in range(n):
            vector = self.fEFA_VTs[i, component - 1, :]
            if norm:
                vector /= vector.max()
            plt.plot(self.wavelengths, vector, label=f'SV={self.sing_values[i]}',
                     color=cmap(i), lw=0.5)
        plt.xlabel('Wavelength / nm')
        plt.ylabel('Amplitude')
        plt.title(f'{component}-th V_T vector')
        # plt.legend()

        plt.show()

    def run_ICA(self, random_state=0, max_iter=1e4):
        ica = FastICA(n_components=self.ICA_components, random_state=random_state, max_iter=int(max_iter))

        self.C_ICA = ica.fit_transform(self.Y)
        self.ST_ICA = ica.mixing_.T

    def set_ICA_filter(self, l_comp=(), n_components=5):
        """Subtracts components in l_comp = [0, 1, 5, ...] list/tuple."""

        if any(map(lambda item: item >= n_components, l_comp)):
            raise ValueError(f"Invalid input, l_comp cannot contain values larger than {n_components - 1}.")

        if n_components != self.ICA_components:
            self.ICA_components = n_components
            self.run_ICA()

        comps = np.zeros(n_components)
        comps[l_comp] = 1

        self.ICA_subtr_mat = self.C_ICA @ np.diag(comps) @ self.ST_ICA  # outer product

        # update D
        self._set_D()

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
        self._set_D()

    def crop_data(self, t0=None, t1=None, w0=None, w1=None):

        self.Y, self.times, self.wavelengths = crop_data(self.Y, self.times, self.wavelengths, t0, t1, w0, w1)

        self.SVD()
        # update matrix D
        self._set_D()

        return self

    def baseline_corr(self, t0=0, t1=200):
        """Subtracts a average of specified time range from all data.
        Deep copies the object and new averaged one is returned."""

        t_idx_start = find_nearest_idx(self.times, t0) if t0 is not None else 0
        t_idx_end = find_nearest_idx(self.times, t1) + 1 if t1 is not None else self.D.shape[0]

        D_selection = self.Y[t_idx_start:t_idx_end, :]
        self.Y -= D_selection.mean(axis=0)

        self.SVD()
        self._set_D()

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
        self._set_D()

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
        self._set_D()

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

    def plot_fit_femto(self, t_unit='ps', z_unit=dA_unit, cmap='diverging', z_lim=(None, None),
                  t_lim=(None, None), w_lim=(None, None), linthresh=1, linscale=1.5, D_mul_factor=1e3,
                  y_major_formatter=ScalarFormatter(),
                  x_minor_locator=AutoMinorLocator(10), n_levels=30, plot_countours=True,
                  colorbar_locator=AutoLocator(), hatch='/////', colorbar_aspect=35, add_wn_axis=True,
                  wls_fit=(355, 400, 450, 500, 550), marker_size=10, marker_linewidth=1,
                  marker_facecolor='none', alpha_traces=1, legend_spacing=0.2, lw_traces=1.5, lw_spectra=1.5,
                  legend_loc_traces='lower right',
                  fig_size=(15, 4.5), dpi=500, filepath=None, transparent=True, hatched_wls=(None, None),
                  plot_ST=True):

        if self.D_fit is None:
            raise ValueError("No fitting data available.")

        _D = self.D.copy()
        _D_fit = self.D_fit.copy()

        assert _D.shape == _D_fit.shape

        if hatched_wls[0] is not None:
            idx1, idx2 = find_nearest_idx(self.wavelengths, hatched_wls)

            mask = np.zeros_like(self.D)
            mask[:, idx1:idx2] = 1

            _D = ma.masked_array(_D, mask=mask)
            # _D_fit = ma.masked_array(_D_fit, mask=mask)

        COLORS = ['blue', 'red', 'green', 'orange', 'purple', 'black', 'gray']

        fig, axes = plt.subplots(1, 3 if plot_ST else 2, figsize=fig_size)

        plot_data_ax(fig, axes[0], _D, self.times, self.wavelengths, D_mul_factor=D_mul_factor, symlog=True,
                     plot_countours=plot_countours,
                     n_levels=n_levels, cmap=cmap, linthresh=linthresh, linscale=linscale,
                     t_unit=t_unit, z_unit=z_unit,
                     z_lim=z_lim, t_lim=t_lim, w_lim=w_lim, y_major_formatter=y_major_formatter,
                     x_minor_locator=x_minor_locator, colorbar_locator=colorbar_locator, hatch=hatch,
                     colorbar_aspect=colorbar_aspect, add_wn_axis=add_wn_axis)

        if plot_ST:
            ST = self.ST_COH if self.ST_fit.shape[0] == 0 else self.ST_fit
            plot_SADS_ax(axes[1], self.wavelengths, ST.T, zero_reg=hatched_wls, colors=COLORS,
                         D_mul_factor=D_mul_factor, z_unit=z_unit, lw=lw_spectra)

        plot_traces_onefig_ax(axes[-1], _D, _D_fit, self.times, self.wavelengths,
                              wls=wls_fit, marker_size=marker_size, alpha=alpha_traces,
                              marker_facecolor=marker_facecolor,
                              marker_linewidth=marker_linewidth, colors=COLORS,
                              linscale=linscale, linthresh=linthresh, x_label=f'Time / {t_unit}',
                              legend_spacing=legend_spacing, y_label=z_unit,
                              lw=lw_traces, legend_loc=legend_loc_traces, D_mul_factor=D_mul_factor)

        plt.tight_layout()

        if filepath:
            ext = os.path.splitext(filepath)[1].lower()[1:]
            plt.savefig(fname=filepath, format=ext, transparent=transparent, dpi=dpi)
        else:
            plt.show()

    def plot_data(self, symlog=False, t_unit='ps', z_unit=dA_unit, cmap='diverging', z_lim=(None, None),
                  t_lim=(None, None), w_lim=(None, None), linthresh=1, linscale=1.5, D_mul_factor=1e3,
                  y_major_formatter=ScalarFormatter(),
                  x_minor_locator=AutoMinorLocator(10), n_levels=30, plot_countours=True,
                  colorbar_locator=AutoLocator(), hatch='/////', colorbar_aspect=35, add_wn_axis=True,
                  fig_size=(5, 4.5), dpi=500, filepath=None, transparent=True, hatched_wls=(None, None)):

        fig, ax = plt.subplots(1, 1, figsize=fig_size)

        _D = self.D.copy()

        if hatched_wls[0] is not None:
            idx1, idx2 = find_nearest_idx(self.wavelengths, hatched_wls)

            mask = np.zeros_like(self.D)
            mask[:, idx1:idx2] = 1

            _D = ma.masked_array(_D, mask=mask)

        plot_data_ax(fig, ax, _D, self.times, self.wavelengths, D_mul_factor=D_mul_factor, symlog=symlog,
                     plot_countours=plot_countours,
                     n_levels=n_levels, cmap=cmap, linthresh=linthresh, linscale=linscale,
                     t_unit=t_unit, z_unit=z_unit,
                     z_lim=z_lim, t_lim=t_lim, w_lim=w_lim, y_major_formatter=y_major_formatter,
                     x_minor_locator=x_minor_locator, colorbar_locator=colorbar_locator, hatch=hatch,
                     colorbar_aspect=colorbar_aspect, add_wn_axis=add_wn_axis)

        plt.tight_layout()

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
            wl_idxs.append(find_nearest_idx(self.wavelengths, wl))

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
            t_idxs.append(find_nearest_idx(self.times, t))

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
            wl_idxs.append(find_nearest_idx(self.wavelengths, wl))

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


    @staticmethod
    def to_string(array, separator='\t', decimal_sep='.', new_line='\n'):
        list_array = array.tolist()

        list_array[0][0] = 'Wavelength'

        buffer = new_line.join(separator.join("{}".format(num) for num in row) for row in list_array)

        return buffer
