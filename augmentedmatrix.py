from lfp_parser import parse_file
import numpy as np
from LFP_matrix import LFP_matrix, MinorSymLogLocator

from Widgets.fit_widget import FitWidget
from plotwidget import PlotWidget

from spectrum import Spectrum
from gui_console import Console

import pyqtgraph as pg
from matplotlib import cm

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

class AugmentedMatrix(object):

    def __init__(self, rows=2, columns=1):
        self.r = rows
        self.c = columns

        # self.matrices = [[None] * columns] * rows
        self.matrices = np.empty((rows, columns), dtype=LFP_matrix)

        self.aug_mat = None
        self.C_aug = None
        self.ST_aug = None

        # self.char_info = np.zeros((rows, columns), dtype=np.float64)

        # just number of stacked dimensions, times and wavelengths would not make any sense
        self.aug_times = None
        self.aug_wavelengths = None

    def load_matrix(self, row, column, filename):
        self.matrices[row, column] = parse_file(filename)

    def get_shapes(self):
        return [[self.matrices[i, j].Y.shape for j in range(self.c)] for i in range(self.r)]

    def construct_aug_matrix(self):
        if self.aug_mat is not None:
            return self.aug_mat

        self.aug_mat = np.hstack([mat.Y for mat in self.matrices[0]])
        # self.aug_wavelengths = np.hstack([mat.wavelengths for mat in self.matrices[0]])
        # self.aug_times = np.hstack([mat.times for mat in self.matrices[:, 0]])

        for r in range(1, self.r):
            stacked_row = np.hstack([mat.Y for mat in self.matrices[r]])
            self.aug_mat = np.vstack([self.aug_mat, stacked_row])

        self.aug_times = np.linspace(0, self.aug_mat.shape[0] - 1, self.aug_mat.shape[0])
        self.aug_wavelengths = np.linspace(0, self.aug_mat.shape[1] - 1, self.aug_mat.shape[1])

        return self.aug_mat

    def get_aug_LFP_matrix(self):
        if self.aug_mat is None:
            self.construct_aug_matrix()

        m = LFP_matrix.from_value_matrix(self.aug_mat, self.aug_times, self.aug_wavelengths)
        m.SVD_filter = False

        return m

    def _C_indiv_range(self, i, j=0):
        if self.C_aug is None:
            return

        idx_0 = 0
        for _i in range(i):
            idx_0 += self[_i, j].times.shape[0]

        n = self[i, j].times.shape[0]

        return idx_0, idx_0 + n

    def _C_indiv(self, i, j=0):
        if self.C_aug is None:
            return
        idx_0, idx_1 = self._C_indiv_range(i, j)

        return self.C_aug[idx_0:idx_1, :]

    def _calc_q_rel(self, eps, c, wls, l=1, I_source=None):
        # c is t x n matrix, eps is n x w matrix

        assert I_source.shape[0] == wls.shape[0]

        I_source /= np.trapz(I_source, x=wls)  # normalize irr source spectrum
        ln10 = np.log(10)
        # c_eps is   t x n x w   matrix
        c_eps = c[..., None] * eps[None, ...]
        # sum in second dimension to produce   t x 1 x w   matrix
        c_dot_eps = c_eps.sum(axis=1, keepdims=True)

        x_abs = c_eps * (1 - np.exp(-l * c_dot_eps * ln10)) / c_dot_eps

        # integrate along wavelengths (the third) dimension
        q_rel = np.trapz(x_abs * I_source, x=wls, axis=2)

        return q_rel.squeeze()

    def plot_fit_photochem(self, symlog=False, c_model=None, z_label='Absorbance', wl_label='Wavelength (nm)', t_label='Time (s)',
                           dpi=500, figsize=(50, 8), cmap='inferno', time_linthresh=200, time_linscale=1, transparent=False,
                           fname=r'C:\Users\dominik\Documents\Projects\Bilirubin\Results\test-fit_photochem.png', step=2):

        # if c_model is not None:
            # # fname = r'C:\Users\Dominik\Documents\MUNI\Organic Photochemistry\Projects\2019-Bilirubin project\UV-VIS\led sources.txt'
            # path = r"C:\Users\dominik\Documents\Projects\Bilirubin\UV-Vis data"
            #
            # data = np.loadtxt(path + r"\em sources.txt", dtype=np.float32, delimiter='\t', skiprows=1)
            #
            # # 355, 375, 375, 400, 450, 490 nm
            # # I_sources = [data[:, 1].copy(), data[:, 2].copy(), data[:, 3].copy(), data[:, 4].copy(), data[:, 5]]
            #
            # I_330 = data[:, 1].copy()
            # I_400 = data[:, 3].copy()
            # I_480 = data[:, 5].copy()

        I_sources = [c_model.I_330.copy(), c_model.I_330.copy(), c_model.I_375.copy(),
                     c_model.I_400.copy(), c_model.I_400.copy(), c_model.I_450.copy(),
                     c_model.I_480.copy(), c_model.I_480.copy(), c_model.LED_355.copy(),
                     c_model.LED_375.copy(), c_model.LED_405.copy(), c_model.LED_490.copy(),
                     c_model.LED_355.copy()]


        # fig, ax = plt.subplots(3, self.r, figsize=figsize)
        fig, ax = plt.subplots(2, self.r, figsize=figsize)

        comp = ['Z', 'E', 'HL', 'Photoproducts']
        cmap = cm.get_cmap(cmap)

        for i in range(self.r):
            wls = self[i, 0].wavelengths
            ts = self[i, 0].times

            D = self[i, 0].Y

            # plot spectrum with I source
            ax[0, i].set_title("Time-Dependent Spectra")

            for j in range(0, ts.shape[0], step):
                ax[0, i].plot(wls, D[j], color=cmap(j/(ts.shape[0]-1)), lw=0.3)

            # plot I_source
            ax[0, i].plot(wls, D.max() * I_sources[i] / I_sources[i].max(), color='black', linestyle='--', lw=1)

            ax[0, i].set_ylabel(z_label)
            ax[0, i].set_xlabel(wl_label)

            # plot conc profiles
            ax[1, i].set_title("Concentration profiles")
            C = self._C_indiv(i)

            for j in range(C.shape[1]):
                ax[1, i].plot(ts, C[:, j], label=comp[j])

            ax[1, i].legend()
            ax[1, i].set_ylabel('Concentration (M)')
            ax[1, i].set_xlabel(t_label)

            if symlog:
                ax[1, i].set_xscale('symlog', subsx=[2, 3, 4, 5, 6, 7, 8, 9], linscalex=time_linscale, linthreshx=time_linthresh)
                ax[1, i].xaxis.set_minor_locator(MinorSymLogLocator(time_linthresh))

            # # plot q rel
            # if I_sources is None:
            #     continue

            # ax[2, i].set_title("Part of absorbed light")
            # q_rel = self._calc_q_rel(self.ST_aug, C, wls, I_source=I_sources[i])
            #
            # for j in range(C.shape[1]):
            #     ax[2, i].plot(ts, q_rel[:, j], label=comp[j])
            #
            # ax[2, i].legend()
            # ax[2, i].set_ylabel('$q(t)/q^0_{mol}$')
            # ax[2, i].set_xlabel(t_label)
            #
            # if symlog:
            #     ax[2, i].set_xscale('symlog', subsx=[2, 3, 4, 5, 6, 7, 8, 9], linscalex=time_linscale, linthreshx=time_linthresh)
            #     ax[2, i].xaxis.set_minor_locator(MinorSymLogLocator(time_linthresh))

        plt.tight_layout()
        # fig.subplots_adjust(hspace=0)

        if fname is not None:
            plt.savefig(fname=fname, format='png', transparent=transparent, dpi=dpi)

        # plt.show()


    def plot_fit(self, t_unit='$\mu$s', z_unit='$\Delta A$', tmin=0, tmax=9.5, ylim0=0, ylim1=0.99,  zmin=None, zmax=None, dpi=100, figsize=(12, 15)):

        fig, ax = plt.subplots(self.r, 3, figsize=figsize)

        # plot fits

        # tmin = 0# self[0, 0].times[0]
        # tmax = 10

        comp = ['Xan', 'DR2', 'Component 3']

        # plot C matrices

        ax[0, 0].set_title("Conc. augmented matrix C")

        i0 = 0

        for r in range(self.r):

            n = self.C_aug.shape[1]
            t = self[r, 0].times

            i1 = i0 + t.shape[0]

            for i in range(n):
                ax[r, 0].plot(t, self.C_aug[i0:i1, i], label=f"{comp[i]}", lw=1.5)

            ax[r, 0].set_ylabel('Rel. population')
            ax[r, 0].legend()
            ax[r, 0].set_xlim(tmin, tmax)
            ax[r, 0].set_ylim(ylim0, ylim1)

            if r < self.r - 1:
                ax[r, 0].xaxis.set_ticks([])

            i0 = i1

        ax[-1, 0].set_xlabel(f'Time ({t_unit})')

        # plot ST matrix

        ax[0, 1].set_title("Spectra aug. matrix ST")

        w = self[0, 0].wavelengths
        assert self.ST_aug.shape[1] == w.shape[0]

        for i in range(self.ST_aug.shape[0]):
            ax[0, 1].plot(w, self.ST_aug[i], label=f"{comp[i]}", lw=1.5)

        ax[0, 1].legend()
        ax[0, 1].set_xlabel('Wavelength (nm)')
        ax[0, 1].set_ylabel(z_unit)

        # remove plots

        for r in range(1, self.r):
            ax[r, 1].set_visible(False)

        # plot residuals

        E_aug = self.C_aug @ self.ST_aug - self.aug_mat

        R2 = (1 - (E_aug * E_aug).sum() / (self.aug_mat * self.aug_mat).sum())
        # E2 = (self.E * self.E).sum()
        title = "Residuals aug. $E=CS^T - D$"
        title += f", $R^2$={R2:.4g}"

        ax[0, 2].set_title(title)

        i0 = 0

        vmin, vmax = -np.abs(np.max(E_aug)), np.abs(np.max(E_aug))

        for r in range(self.r):
            x, y = np.meshgrid(self[r, 0].wavelengths,
                               self[r, 0].times)  # needed for pcolormesh to correctly scale the image

            t = self[r, 0].times
            i1 = i0 + t.shape[0]

            E = E_aug[i0:i1, :]

            mappable = ax[r, 2].pcolormesh(x, y, E, cmap='seismic', vmin=vmin, vmax=vmax)

            fig.colorbar(mappable, ax=ax[r, 2], label=z_unit)
            ax[r, 2].set_ylabel(f'Time ({t_unit})')
            ax[r, 2].invert_yaxis()

            if r < self.r - 1:
                ax[r, 2].xaxis.set_ticks([])

            i0 = i1

        ax[-1, 2].set_xlabel('Wavelength (nm)')

        plt.tight_layout()
        fig.subplots_adjust(hspace=0)

        plt.savefig(fname=r'C:\Users\Dominik\Desktop\snth\test-fit.png', format='png', transparent=True, dpi=dpi)


        # plt.show()


    def plot_data(self, t_unit='$\mu$s', z_unit='$\Delta A$', c_map='inferno_r', zmin=None, zmax=None, dpi=500,
                  figsize=(8, 12), ts=(0.2, 0.5, 0.7, 1, 1.5, 2), ylim0=0, ylim1=0.2):

        fig, ax = plt.subplots(self.r, 2, figsize=figsize)

        ax[0, 0].set_title("Augmented matrix D")

        for r in range(self.r):
            x, y = np.meshgrid(self[r, 0].wavelengths,
                               self[r, 0].times)  # needed for pcolormesh to correctly scale the image

            D = self[r, 0].Y

            mappable = ax[r, 0].pcolormesh(x, y, D, cmap=c_map, vmin=np.min(D) if zmin is None else zmin,
                                        vmax=np.max(D) if zmax is None else zmax)

            fig.colorbar(mappable, ax=ax[r, 0], label=z_unit)

            ax[r, 0].set_ylabel(f'Time ({t_unit})')

            ax[r, 0].invert_yaxis()

            if r < self.r - 1:
                ax[r, 0].xaxis.set_ticks([])

        ax[-1, 0].set_xlabel('Wavelength (nm)')

        # plot spectra at different times

        for r in range(self.r):
            t_idxs = []
            for t in ts:
                t_idxs.append(Spectrum.find_nearest_idx(self[r, 0].times, t))

            for i, idx in enumerate(t_idxs):
                qc = pg.intColor(i, len(ts))
                color = (qc.red() / 255.0, qc.green() / 255.0, qc.blue() / 255.0)
                ax[r, 1].plot(self[r, 0].wavelengths, self[r, 0].Y[idx], label=f"{ts[i]} {t_unit}", lw=1, color=color)

            ax[r, 1].legend()
            ax[r, 1].set_ylabel(z_unit)
            ax[r, 1].set_ylim(ylim0, ylim1)

            if r < self.r - 1:
                ax[r, 1].xaxis.set_ticks([])

        ax[-1, 1].set_xlabel('Wavelength (nm)')

        # plt.tight_layout()
        # plt.subplots_adjust(hspace=0, wspace=0.4)
        plt.tight_layout()
        fig.subplots_adjust(hspace=0)

        plt.savefig(fname=r'C:\Users\Dominik\Desktop\snth\data.png', format='png', transparent=True, dpi=dpi)

        # plt.show()

    def __getitem__(self, item):
        return self.matrices[item]


def setup(t0=0.15, w0=460, w1=700):
    fw = FitWidget.instance
    pw = PlotWidget.instance

    path = r"C:\Users\Dominik\Documents\MUNI\Organic Photochemistry\Projects\2018-19_Japan-C-C bond homolysis\LFP\2019-07-11 Xan + CP2"

    paths = []

    paths.append(path + r"\p=0 T=20\20us-2sk-450-710-spectra.CSV")
    paths.append(path + r"\p=0,02 T=20\10us-2sk-450-710-spectra.CSV")
    paths.append(path + r"\p=0,05 T=20\10us-2sk-450-710-spectra.CSV")
    paths.append(path + r"\p=0,1 T=20\10us-2sk-450-710-spectra.CSV")
    paths.append(path + r"\p=0,2 T=20\10us-2sk-450-710-spectra.CSV")
    paths.append(path + r"\p=0,3 T=20\10us-2sk-450-710-spectra.CSV")
    paths.append(path + r"\p=2,5 T=20\10us-2sk-450-710-spectra.CSV")

    au = AugmentedMatrix(len(paths), 1)

    for i in range(len(paths)):
        au.load_matrix(i, 0, paths[i])

    t_dim_reduce = 10

    au[0, 0].crop_data(t0, 18, w0, w1)#.reduce(t_dim=t_dim_reduce)
    au[1, 0].crop_data(t0, 9, w0, w1)#.reduce(t_dim=t_dim_reduce)
    au[2, 0].crop_data(t0, 9, w0, w1)#.reduce(t_dim=t_dim_reduce)
    au[3, 0].crop_data(t0, 9, w0, w1)#.reduce(t_dim=t_dim_reduce)
    au[4, 0].crop_data(t0, 9, w0, w1)#.reduce(t_dim=t_dim_reduce)
    au[5, 0].crop_data(t0, 9, w0, w1)#.reduce(t_dim=t_dim_reduce)
    au[6, 0].crop_data(t0, 9, w0, w1)#.reduce(t_dim=t_dim_reduce)

    m = au.get_aug_LFP_matrix()

    pw.plot_matrix(m)

    fw.matrix = m
    return au


def setup2(t1=1000):
    fw = FitWidget.instance
    pw = PlotWidget.instance

    path = r"C:\Users\Dominik\Documents\MUNI\Organic Photochemistry\Projects\2019-Bilirubin project\UV-VIS\Irradiation kinetics\for MCR\Kinetics Z"
    paths = []

    paths.append(path + r"\355 nm MeOH aerated 2 LED modules\cut.txt")
    paths.append(path + r"\375 nm MeOH aerated 2 LED modules\cut.txt")
    paths.append(r'C:\Users\Dominik\Documents\MUNI\Organic Photochemistry\Projects\2019-Bilirubin project\UV-VIS\QY measurement\Z start\365 nm LED irr, pinhole 3\kin.txt')

    # paths.append(path + r"\375 nm MeOH aerated\cut.txt")
    paths.append(path + r"\400 nm MeOH aerated\cut.txt")
    paths.append(path + r"\450 nm MeOH aerated\cut.txt")
    paths.append(path + r"\490 nm MeOH aerated\cut.txt")
    # paths.append(path + r"\01_1000 uL MeOH\1000 uL bc cut.txt")

    au = AugmentedMatrix(len(paths), 1)

    for i in range(len(paths)):
        au.load_matrix(i, 0, paths[i])
        au[i, 0].crop_data(0, t1)

    # au[0, 0].crop_data(0, t1)
    # au[1, 0].crop_data(0, t1)
    # au[2, 0].crop_data(0, t1)

    m = au.get_aug_LFP_matrix()

    pw.plot_matrix(m)
    fw.matrix = m
    fw._au = au
    Console.push_variables({'matrix': m})

    return au


def setup3():
    fw = FitWidget.instance
    pw = PlotWidget.instance

    # path = r"C:\Users\Dominik\Documents\MUNI\Organic Photochemistry\Projects\2019-Bilirubin project\UV-VIS\QY measurement\Photodiode\new setup"
    path = r"C:\Users\dominik\Documents\Projects\Bilirubin\UV-Vis data"

    paths = []

    paths.append(path + r"\Z 330 nm\cut.txt")
    paths.append(path + r"\E 330 nm\cut.txt")

    paths.append(path + r"\Z 375 nm\cut.txt")

    paths.append(path + r"\Z 400 nm\cut.txt")
    paths.append(path + r"\E 400 nm\cut.txt")

    paths.append(path + r"\Z 450 nm\cut.txt")

    paths.append(path + r"\Z 480 nm\cut.txt")
    paths.append(path + r"\E 480 nm\cut.txt")

    path_reactors = r"C:\Users\dominik\Documents\Projects\Bilirubin\UV-Vis data\Kinetics Z degassed"

    paths.append(path_reactors + r"\355 nm (both modules)\cut.txt")
    paths.append(path_reactors + r"\375 nm (both modules)\cut.txt")
    paths.append(path_reactors + r"\405 nm (both modules)\cut.txt")

    paths.append(path_reactors + r"\490 nm, then switched 355 nm (both modules)\cut490.txt")
    paths.append(path_reactors + r"\490 nm, then switched 355 nm (both modules)\cut355.txt")


    # paths.append(path_reactors + r"\450 nm (one module)\cut.txt")
    # paths.append(path_reactors + r"\470 nm (one module)\cut.txt")


    au = AugmentedMatrix(len(paths), 1)

    for i in range(len(paths)):
        au.load_matrix(i, 0, paths[i])
        # au[i, 0].reduce(t_dim=2)

    au[-5, 0].crop_data(t1=2000)
    au[-4, 0].crop_data(t1=2000)
    au[-3, 0].crop_data(t1=2000)


    m = au.get_aug_LFP_matrix()

    pw.plot_matrix(m)
    fw.matrix = m
    fw._au = au
    Console.push_variables({'matrix': m})

    return au


def setup4():
    fw = FitWidget.instance
    pw = PlotWidget.instance

    # path = r"C:\Users\Dominik\Documents\MUNI\Organic Photochemistry\Projects\2019-Bilirubin project\UV-VIS\QY measurement\Photodiode\new setup"

    paths = []

    path_reactors = r"C:\Users\dominik\Documents\Projects\Bilirubin\UV-Vis data\Kinetics Z degassed"

    paths.append(path_reactors + r"\355 nm (both modules)\cut.txt")
    paths.append(path_reactors + r"\355 nm (both modules)\cut.txt")
    paths.append(path_reactors + r"\405 nm (both modules)\cut.txt")
    paths.append(path_reactors + r"\450 nm (one module)\cut.txt")
    paths.append(path_reactors + r"\470 nm (one module)\cut.txt")
    paths.append(path_reactors + r"\490 nm, then switched 355 nm (both modules)\cut490.txt")
    paths.append(path_reactors + r"\490 nm, then switched 355 nm (both modules)\cut355.txt")


    au = AugmentedMatrix(len(paths), 1)

    for i in range(len(paths)):
        au.load_matrix(i, 0, paths[i])
        au[i, 0].reduce(t_dim=2)

    m = au.get_aug_LFP_matrix()

    pw.plot_matrix(m)
    fw.matrix = m
    fw._au = au
    Console.push_variables({'matrix': m})

    return au



if __name__ == "__main__":
    m = AugmentedMatrix(2, 1)

    path = r"C:\Users\Dominik\Documents\MUNI\Organic Photochemistry\Projects\2018-19_Japan-C-C bond homolysis\LFP\2019-07-11 Xan + CP2"

    fname0 = path + r"\p=0 T=20\20us-2sk-450-710-spectra.CSV"
    fname1 = path + r"\p=0,1 T=20\10us-2sk-450-710-spectra.CSV"

    m.load_matrix(0, 0, fname0)
    m.load_matrix(1, 0, fname1)

    print(m.get_shapes())

    m.construct_aug_matrix()

    print(m.aug_mat.shape, m.aug_wavelengths, m.aug_times)

    # print(m.matrices[0][0])
