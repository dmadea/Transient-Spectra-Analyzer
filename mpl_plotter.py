
import numpy as np
import matplotlib.pyplot as plt

from misc import find_nearest_idx
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from matplotlib import colormaps
import matplotlib as mpl
import matplotlib.colors as c
from numpy import ma

from matplotlib.ticker import SymmetricalLogLocator, ScalarFormatter, AutoMinorLocator, MultipleLocator, Locator

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

X_SIZE, Y_SIZE = 5, 4.5
dA_unit = '$\Delta A$ / $10^{-3}$'

COLORS = ['blue', 'red', 'green', 'orange', 'black', 'yellow']
COLORS_gradient = ['blue', 'lightblue']


class MajorSymLogLocator(SymmetricalLogLocator):
    """
    Determine the tick locations for symmetric log axes.
    Slight modification .... TODO
    """

    def tick_values(self, vmin, vmax):
        base = self._base
        linthresh = self._linthresh

        if vmax < vmin:
            vmin, vmax = vmax, vmin

        # if -linthresh < vmin < vmax < linthresh:
        #     # only the linear range is present
        #     return [vmin, vmax]

        # Lower log range is present
        has_a = (vmin < -linthresh)
        # Upper log range is present
        has_c = (vmax > linthresh)

        # Check if linear range is present
        has_b = (has_a and vmax > -linthresh) or (has_c and vmin < linthresh) or -linthresh < vmin < vmax < linthresh

        def get_log_range(lo, hi):
            lo = np.floor(np.log(lo) / np.log(base))
            hi = np.ceil(np.log(hi) / np.log(base))
            return lo, hi

        # Calculate all the ranges, so we can determine striding
        a_lo, a_hi = (0, 0)
        if has_a:
            a_upper_lim = min(-linthresh, vmax)
            a_lo, a_hi = get_log_range(abs(a_upper_lim), abs(vmin) + 1)

        c_lo, c_hi = (0, 0)
        if has_c:
            c_lower_lim = max(linthresh, vmin)
            c_lo, c_hi = get_log_range(c_lower_lim, vmax + 1)

        # Calculate the total number of integer exponents in a and c ranges
        total_ticks = (a_hi - a_lo) + (c_hi - c_lo)
        if has_b:
            total_ticks += 1
        stride = max(total_ticks // (self.numticks - 1), 1)

        ticklocs = []  # places to put major ticks

        if has_a:
            ticklocs.extend(-1 * (base ** (np.arange(a_lo, a_hi,
                                                    stride)[::-1])))

        if has_c:
            ticklocs.extend(base ** (np.arange(c_lo, c_hi, stride)))

        if has_b:
            linthresh_base = base ** np.floor(np.log(linthresh) / np.log(base))

            major_ticks = np.arange(-linthresh, linthresh + linthresh_base, linthresh_base)

            for tick in major_ticks:
                if tick not in ticklocs:
                    ticklocs.append(tick)

        ret = np.array(ticklocs)
        ret.sort()

        return self.raise_if_exceeds(ret)


class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.... Modified, TODO
    """

    def __init__(self, linthresh=1, n_lin_ints=10, n_log_ints=10, base=10.):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically. nints gives the number of
        intervals that will be bounded by the minor ticks.
        """
        self.linthresh = linthresh
        self.nintervals = n_lin_ints
        self.n_log_intervals = n_log_ints
        self.base = base

    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
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
            majorlocs = np.insert(majorlocs, 0, majorlocs[0] * self.base)
        else:
            majorlocs = np.insert(majorlocs, 0, majorlocs[0] - self.linthresh)

        # add temporary major tick location at the upper end
        if majorlocs[-1] != 0. and ((np.abs(majorlocs[-1]) != self.linthresh and dmupper > self.linthresh) or (
                dmupper == self.linthresh and majorlocs[-1] > 0)):
            majorlocs = np.append(majorlocs, majorlocs[-1] * self.base)
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
                # ndivs = self.n_log_intervals - 1

                # if the difference between major locks is not full decade
                # log_ratio = np.sign(majorlocs[i - 1]) * np.log(majorlocs[i] / majorlocs[i - 1]) / np.log(self.base)
                # if log_ratio != 1:
                base_difference = majorstep / (self.base ** np.floor(np.log(majorstep) / np.log(self.base)))
                ndivs = (self.n_log_intervals - 1) * base_difference / (self.base - 1)

            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i - 1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))


def eps_label(factor):
    num = np.log10(1 / factor).astype(int)
    return f'$\\varepsilon$ / $(10^{{{num}}}$ L mol$^{{-1}}$ cm$^{{-1}})$'


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


def get_sym_space(vmin, vmax, n):
    """Return evenly spaced tics that are however symetric around zero (without zero!) so that tics are: (-t_i, t_i+1)"""
    raw_step = (vmax - vmin) / (n - 1)

    pos_steps = int(np.ceil((vmax - 0.5 * raw_step) / raw_step))
    neg_steps = int(np.ceil((abs(vmin) - 0.5 * raw_step) / raw_step))

    return np.linspace((neg_steps + 0.5) * -raw_step, (pos_steps + 0.5) * raw_step, pos_steps + neg_steps + 2)


def _plot_tilts(ax, norm, at_value, axis='y', inverted_axis=False):
    d = 0.015
    tilt = 0.4

    sep = 1 - norm(at_value) if inverted_axis else norm(at_value)
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)

    x_vals = [[-d * 0.8, +d * 0.8],
             [1 - d * 0.8, 1 + d * 0.8],
             [0, 1]]
    y_vals = [[sep - d * tilt, sep + d * tilt],
             [sep - d * tilt, sep + d * tilt],
             [sep, sep]]

    if axis == 'x':
        x_vals, y_vals = y_vals, x_vals

    ax.plot(x_vals[0], y_vals[0], **kwargs)
    ax.plot(x_vals[1], y_vals[1], **kwargs)
    ax.plot(x_vals[2], y_vals[2], ls='dotted', lw=1, **kwargs)


def plot_traces_onefig_ax(ax, D, D_fit, times, wavelengths, mu=None, wls=(355, 400, 450, 500, 550), marker_size=10,
                          marker_linewidth=1, n_lin_bins=10, n_log_bins=10, t_axis_formatter=ScalarFormatter(),
                          marker_facecolor='white', alpha=0.8, y_lim=(None, None), plot_tilts=True, wl_unit='nm',
                          linthresh=1, linscale=1, colors=None, D_mul_factor=1e3, legend_spacing=0.2, lw=1.5,
                          legend_loc='lower right', y_label=dA_unit, x_label='Time / ps', symlog=True,
                          t_lim=(None, None),
                          plot_zero_line=True):
    n = wls.__len__()
    t = times
    mu = np.zeros_like(wavelengths) if mu is None else mu

    t_lim = (times[0] if t_lim[0] is None else t_lim[0], times[-1] if t_lim[1] is None else t_lim[1])

    set_main_axis(ax, xlim=t_lim, ylim=y_lim, y_label=y_label, x_label=x_label,
                  y_minor_locator=None, x_minor_locator=None)

    if plot_zero_line:
        ax.plot(t - mu[0].mean(), np.zeros_like(t), ls='--', color='black', lw=1)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] if colors is None else colors

    for i in range(n):
        color_points = c.to_rgb(colors[i])
        color_line = np.asarray(color_points) * 0.7

        idx = find_nearest_idx(wavelengths, wls[i])
        tt = t - mu[idx]
        ax.scatter(tt, D[:, idx] * D_mul_factor, edgecolor=color_points, facecolor=marker_facecolor,
                   alpha=alpha, marker='o', s=marker_size, linewidths=marker_linewidth)

        ax.scatter([], [], edgecolor=color_points, facecolor=marker_facecolor,
                   alpha=alpha, marker='o', label=f'{wls[i]} {wl_unit}', s=marker_size * 2, linewidths=marker_linewidth)
        ax.plot(tt, D_fit[:, idx] * D_mul_factor, lw=lw, color=color_line)

    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    if symlog:
        ax.set_xscale('symlog', subs=[2, 3, 4, 5, 6, 7, 8, 9], linscale=linscale, linthresh=linthresh)

        ax.xaxis.set_major_locator(MajorSymLogLocator(base=10, linthresh=linthresh))
        ax.xaxis.set_minor_locator(MinorSymLogLocator(linthresh, n_lin_ints=n_lin_bins, n_log_ints=n_log_bins, base=10))

        if plot_tilts:
            norm = c.SymLogNorm(vmin=t_lim[0], vmax=t_lim[1], linscale=linscale, linthresh=linthresh, base=10,
                                clip=True)
            _plot_tilts(ax, norm, linthresh, 'x')

    if t_axis_formatter:
        ax.xaxis.set_major_formatter(t_axis_formatter)

    l = ax.legend(loc=legend_loc, frameon=False, labelspacing=legend_spacing)
    for text, color in zip(l.get_texts(), colors):
        text.set_color(color)

    ax.set_axisbelow(False)


# def plot_traces_onefig_ax(ax, D, D_fit, times, wavelengths, mu=None, wls=(355, 400, 450, 500, 550), marker_size=10,
#                           marker_linewidth=1, n_lin_bins=10, n_log_bins=10, t_axis_formatter=ScalarFormatter(),
#                           marker_facecolor='white', alpha=0.8, y_lim=(None, None),
#                           linthresh=1, linscale=1, colors=None, D_mul_factor=1e3, legend_spacing=0.2, lw=1.5,
#                           legend_loc='lower right', y_label=dA_unit, x_label='Time / ps', symlog=True, t_lim=(None, None)):
#
#     n = wls.__len__()
#     t = times
#     mu = np.zeros_like(t) if mu is None else mu
#     norm = mpl.colors.SymLogNorm(vmin=t[0], vmax=t[-1], linscale=linscale, linthresh=linthresh, base=10, clip=True)
#
#     t_lim = (times[0] if t_lim[0] is None else t_lim[0], times[-1] if t_lim[1] is None else t_lim[1])
#     set_main_axis(ax, xlim=t_lim, ylim=y_lim, y_label=y_label, x_label=x_label,
#                   y_minor_locator=None, x_minor_locator=None)
#
#     ax.plot(t - mu[0].mean(), np.zeros_like(t), ls='--', color='black', lw=1)
#
#     colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] if colors is None else colors
#
#     for i in range(n):
#         color_points = c.to_rgb(colors[i])
#         color_line = np.asarray(color_points) * 0.7
#
#         idx = find_nearest_idx(wavelengths, wls[i])
#         tt = t - mu[idx]
#         ax.scatter(tt, D[:, idx] * D_mul_factor, edgecolor=color_points, facecolor=marker_facecolor,
#                    alpha=alpha, marker='o', s=marker_size, linewidths=marker_linewidth)
#
#         ax.scatter([], [], edgecolor=color_points, facecolor=marker_facecolor,
#                    alpha=alpha, marker='o', label=f'{wls[i]} nm', s=marker_size * 2, linewidths=marker_linewidth)
#         ax.plot(tt, D_fit[:, idx] * D_mul_factor, lw=lw, color=color_line)
#
#     ax.xaxis.set_ticks_position('both')
#     ax.yaxis.set_ticks_position('both')
#
#     if symlog:
#         ax.set_xscale('symlog', subs=[2, 3, 4, 5, 6, 7, 8, 9], linscale=linscale, linthresh=linthresh)
#
#         ax.xaxis.set_major_locator(MajorSymLogLocator(base=10, linthresh=linthresh))
#         ax.xaxis.set_minor_locator(MinorSymLogLocator(linthresh, n_lin_ints=n_lin_bins, n_log_ints=n_log_bins, base=10))
#
#         d = 0.015
#         tilt = 0.4
#         ax.set_ylim(ax.get_ylim())
#
#         sep = norm(linthresh)
#         kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
#         ax.plot([sep - d * tilt, sep + d * tilt], [-d * 0.8, +d * 0.8], **kwargs)
#         ax.plot([sep - d * tilt, sep + d * tilt], [1 - d * 0.8, 1 + d * 0.8], **kwargs)
#         ax.vlines(linthresh, ax.get_ylim()[0], ax.get_ylim()[1], ls='dotted', lw=1, color='k', zorder=10)
#
#     if t_axis_formatter:
#         ax.xaxis.set_major_formatter(t_axis_formatter)
#
#     l = ax.legend(loc=legend_loc, frameon=False, labelspacing=legend_spacing)
#     for text, color in zip(l.get_texts(), colors):
#         text.set_color(color)
#
#     ax.set_axisbelow(False)


def plot_spectra_ax(ax, D, times, wavelengths, selected_times=(0, 100), zero_reg=None, z_unit=dA_unit, D_mul_factor=1,
                    legend_spacing=0.05, colors=None, lw=1.5, darkens_factor_cmap=1, cmap='jet', columnspacing=2,
                    legend_loc='lower right', legend_ncol=2, ylim=None, label_prefix='t = ', time_unit='ps'):

    _D = D * D_mul_factor

    if zero_reg[0] is not None:
        cut_idxs = find_nearest_idx(wavelengths, zero_reg)
        _D[:, cut_idxs[0]:cut_idxs[1]] = np.nan

    set_main_axis(ax, y_label=z_unit, xlim=(wavelengths[0], wavelengths[-1]),
                  x_minor_locator=AutoMinorLocator(10), x_major_locator=MultipleLocator(100), y_minor_locator=None)
    _ = setup_wavenumber_axis(ax, x_major_locator=MultipleLocator(0.5))

    t_idxs = find_nearest_idx(times, selected_times)

    _cmap = cm.get_cmap(cmap, t_idxs.shape[0])
    ax.axhline(0, wavelengths[0], wavelengths[-1], ls='--', color='black', lw=1)

    for i in range(t_idxs.shape[0]):
        if colors is None:
            color = np.asarray(c.to_rgb(_cmap(i))) * darkens_factor_cmap
            color[color > 1] = 1
        else:
            color = colors[i]

        ax.plot(wavelengths, _D[t_idxs[i]], color=color, lw=lw, label=f'{label_prefix}${selected_times[i]:.3g}$ {time_unit}')

    l = ax.legend(loc=legend_loc, frameon=False, labelspacing=legend_spacing, ncol=legend_ncol,
                  # handlelength=0, handletextpad=0,
                  columnspacing=columnspacing)
    for i, text in enumerate(l.get_texts()):
        # text.set_ha('right')
        text.set_color(_cmap(i))

    ax.set_axisbelow(False)
    ax.yaxis.set_ticks_position('both')


def plot_data_ax(fig, ax, matrix, times, wavelengths, symlog=True, t_unit='ps',
                 z_unit=dA_unit, cmap='diverging', z_lim=(None, None),
                 t_lim=(None, None), w_lim=(None, None), linthresh=1, linscale=1, D_mul_factor=1e3,
                 n_lin_bins=10, n_log_bins=10, plot_tilts=True,
                 y_major_formatter=ScalarFormatter(),
                 x_minor_locator=AutoMinorLocator(10), x_major_locator=None, n_levels=30, plot_countours=True,
                 colorbar_locator=MultipleLocator(50), colorbarpad=0.04,
                 diverging_white_cmap_tr=0.98, hatch='/////', colorbar_aspect=35, add_wn_axis=True,
                 x_label="Wavelength / nm"):
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

    set_main_axis(ax, xlim=w_lim, ylim=t_lim, x_label=x_label, y_label=f'Time delay / {t_unit}',
                  x_minor_locator=x_minor_locator, x_major_locator=x_major_locator, y_minor_locator=None)
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
        ax.contour(x, y, D, colors=colors, levels=levels, antialiased=True, linewidths=0.1,
                   alpha=1, linestyles='-')

    ax.invert_yaxis()

    ax.tick_params(which='major', direction='out')
    ax.tick_params(which='minor', direction='out')
    ax.yaxis.set_ticks_position('both')

    ax.set_axisbelow(False)

    fig.colorbar(mappable, ax=ax, label=z_unit, orientation='vertical', aspect=colorbar_aspect, pad=colorbarpad,
                 ticks=colorbar_locator)

    if symlog:
        ax.set_yscale('symlog', subs=[2, 3, 4, 5, 6, 7, 8, 9], linscale=linscale, linthresh=linthresh)
        ax.yaxis.set_major_locator(MajorSymLogLocator(base=10, linthresh=linthresh))
        ax.yaxis.set_minor_locator(MinorSymLogLocator(linthresh, n_lin_ints=n_lin_bins, n_log_ints=n_log_bins, base=10))

        if plot_tilts:
            norm = c.SymLogNorm(vmin=t_lim[0], vmax=t_lim[1], linscale=linscale, linthresh=linthresh, base=10,
                                clip=True)
            _plot_tilts(ax, norm, linthresh, 'y', inverted_axis=True)

    if y_major_formatter:
        ax.yaxis.set_major_formatter(y_major_formatter)

#
# def plot_data_ax(fig, ax, matrix, times, wavelengths, symlog=True, t_unit='ps',
#                  z_unit=dA_unit, cmap='diverging', z_lim=(None, None),
#                  t_lim=(None, None), w_lim=(None, None), linthresh=1, linscale=1, D_mul_factor=1e3,
#                  n_lin_bins=10, n_log_bins=10,
#                  y_major_formatter=ScalarFormatter(),
#                  x_minor_locator=AutoMinorLocator(10), n_levels=30, plot_countours=True,
#                  colorbar_locator=MultipleLocator(50),
#                  diverging_white_cmap_tr=0.98, hatch='/////', colorbar_aspect=35, add_wn_axis=True, x_label="Wavelength / nm"):
#     """data is individual dataset"""
#
#     # assert type(data) == Data
#
#     t_lim = (times[0] if t_lim[0] is None else t_lim[0], times[-1] if t_lim[1] is None else t_lim[1])
#     w_lim = (
#         wavelengths[0] if w_lim[0] is None else w_lim[0], wavelengths[-1] if w_lim[1] is None else w_lim[1])
#
#     D = matrix.copy() * D_mul_factor
#
#     zmin = np.min(D) if z_lim[0] is None else z_lim[0]
#     zmax = np.max(D) if z_lim[1] is None else z_lim[1]
#
#     if z_lim[0] is not None:
#         D[D < zmin] = zmin
#
#     if z_lim[1] is not None:
#         D[D > zmax] = zmax
#
#     register_div_cmap(zmin, zmax)
#     register_div_white_cmap(zmin, zmax, diverging_white_cmap_tr)
#
#     x, y = np.meshgrid(wavelengths, times)  # needed for pcolormesh to correctly scale the image
#
#     # plot data matrix D
#
#     set_main_axis(ax, xlim=w_lim, ylim=t_lim, x_label=x_label, y_label=f'Time delay / {t_unit}',
#                   x_minor_locator=x_minor_locator, y_minor_locator=None)
#     if add_wn_axis:
#         w_ax = setup_wavenumber_axis(ax, x_major_locator=MultipleLocator(0.5))
#         w_ax.tick_params(which='minor', direction='out')
#         w_ax.tick_params(which='major', direction='out')
#
#     #     ax.set_facecolor((0.8, 0.8, 0.8, 1))
#     if ma.is_masked(D):  # https://stackoverflow.com/questions/41664850/hatch-area-using-pcolormesh-in-basemap
#         m_idxs = np.argwhere(D.mask[0] > 0).squeeze()
#         wl_range = [wavelengths[m_idxs[0] - 1], wavelengths[m_idxs[-1] + 1]]
#         ax.fill_between(wl_range, [t_lim[0], t_lim[0]], [t_lim[1], t_lim[1]], facecolor="none",
#                         hatch=hatch, edgecolor="k", linewidth=0.0)
#
#     #     mappable = ax.pcolormesh(x, y, D, cmap=cmap, vmin=zmin, vmax=zmax)
#     levels = get_sym_space(zmin, zmax, n_levels)
#     mappable = ax.contourf(x, y, D, cmap=cmap, vmin=zmin, vmax=zmax, levels=levels, antialiased=True)
#
#     if plot_countours:
#         cmap_colors = cm.get_cmap(cmap)
#         colors = cmap_colors(np.linspace(0, 1, n_levels + 1))
#         colors *= 0.45  # plot contours as darkens colors of colormap, blue -> darkblue, white -> gray ...
#         ax.contour(x, y, D, colors=colors, levels=levels, antialiased=True, linewidths=0.2,
#                    alpha=1, linestyles='-')
#
#     ax.invert_yaxis()
#
#     ax.tick_params(which='major', direction='out')
#     ax.tick_params(which='minor', direction='out')
#
#     ax.set_axisbelow(False)
#
#     fig.colorbar(mappable, ax=ax, label=z_unit, orientation='vertical', aspect=colorbar_aspect, pad=0.025,
#                  ticks=colorbar_locator)
#
#     if symlog:
#         ax.set_yscale('symlog', subs=[2, 3, 4, 5, 6, 7, 8, 9], linscale=linscale, linthresh=linthresh)
#         ax.yaxis.set_major_locator(MajorSymLogLocator(base=10, linthresh=linthresh))
#         ax.yaxis.set_minor_locator(MinorSymLogLocator(linthresh, n_lin_ints=n_lin_bins, n_log_ints=n_log_bins, base=10))
#
#     if y_major_formatter:
#         ax.yaxis.set_major_formatter(y_major_formatter)


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
    w1, w2 = find_nearest_idx(wls, w_lim)
    _SADS = _SADS[w1:w2 + 1]
    wls = wls[w1:w2 + 1]

    # fctr = 1.1
    # _min, _max = abs(np.nanmin(_SADS)) * fctr * np.sign(np.nanmin(_SADS)), abs(np.nanmax(_SADS)) * fctr * np.sign(np.nanmax(_SADS))

    set_main_axis(ax, y_label=z_unit, xlim=w_lim, #, ylim=(_min, _max),
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
    # colormaps.register(custom_cmap, 'diverging_white_tr', force=True)
    cm.unregister_cmap('diverging_white_tr')
    cm.register_cmap('diverging_white_tr', custom_cmap)


def register_div_cmap(zmin, zmax):  # colors for femto TA heat maps: dark blue, blue, white, yellow, red, dark red
    """c map suited for the data so that zero will be always in white color."""

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
    cm.unregister_cmap('diverging')
    cm.register_cmap('diverging', custom_cmap)
    # colormaps.register(custom_cmap, 'diverging', force=True)



