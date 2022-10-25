from PyQt6 import QtCore, QtGui, QtWidgets
import numpy as np
import pyqtgraph as pg
from pyqtgraph.exporters import SVGExporter, ImageExporter
from pyqtgraph.dockarea import Dock, DockArea
from PyQt6 import QtCore, QtGui

from PyQt6.QtGui import *
# from PyQt6.QtWidgets import *
from misc import find_nearest_idx, str_is_integer, crop_data

# from PyQt6.QtCore import *

from Widgets.heatmap import HeatMapWidget
from Widgets.plotwidget import PlotWidget
from Widgets.datapanel import DataPanel
from Widgets.svddockarea import SVDDockArea

from pyqtgraphmodif.dock_modif import DockLabel, DockDisplayMode

from scipy.linalg import lstsq
from scipy.integrate import cumtrapz

from scipy.interpolate import interp2d


class CommonDimension:
    Not = None
    First = 0
    Second = 1
    Both = 2


# from https://math.stackexchange.com/questions/1428566/fit-sum-of-exponentials/3808325#3808325
def fit_sum_exp(x, y, n=2, fit_intercept=False):
    """Fits the data with the sum of exponential function and returns

    if fit_intercept is True, last multiplier will be the intercept, also the 0 will be added
    at the end of lambda vector"""

    assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
    assert x.shape[0] == y.shape[0]
    assert x.shape[0] >= 2 * n

    Y_size = 2 * n + 1 if fit_intercept else 2 * n
    Y = np.empty((x.shape[0], Y_size))

    Y[:, 0] = cumtrapz(y, x, initial=0)
    for i in range(1, n):
        Y[:, i] = cumtrapz(Y[:, i - 1], x, initial=0)

    Y[:, -1] = 1
    for i in reversed(range(n, Y_size - 1)):
        Y[:, i] = Y[:, i + 1] * x

    A = lstsq(Y, y)[0]
    Ahat = np.diag(np.ones(n - 1), -1)
    Ahat[0] = A[:n]

    lambdas = np.linalg.eigvals(Ahat)
    # remove imaginary values
    if any(np.iscomplex(lambdas)):
        lambdas = lambdas.real

    X = np.exp(lambdas[None, :] * x[:, None])
    if fit_intercept:
        X = np.hstack((X, np.ones_like(x)[:, None]))
        lambdas = np.insert(lambdas, n, 0)
    multipliers = lstsq(X, y)[0]

    return multipliers, lambdas


def fit_polynomial_coefs(x, y, n=3):
    X = np.ones((x.shape[0], n))  # polynomial regression matrix

    for i in range(1, n):
        X[:, i:] *= x[:, None] / 100

    parmu = lstsq(X, y)[0]
    return parmu


class MainDisplayDockArea(DockArea):
    instance = None

    n_spectra = 20

    def __init__(self, parent=None):
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOptions(antialias=True)

        super(MainDisplayDockArea, self).__init__(parent)

        MainDisplayDockArea.instance = self

        self.heatmap_range_lock = False
        self.heatmap_line_lock = False

        # self.smooth_count = 0

        self.fit_matrix = None
        self.matrices = []

        # axes of the equal dimensions (same start and end points) of all matrices
        # 0 for first dimensions, 1 for second, 2 for both
        self.same_dimension = CommonDimension.Not

        self.plotted_spectra = []
        self.plotted_traces = []

        self.heat_map_levels = None
        self.selected_range_idxs = None

        #  heat map
        default_mode = DockDisplayMode.Matrix
        self.heat_map_widget = HeatMapWidget(self, default_mode=default_mode)
        self.heat_map_dock_label = DockLabel("Heat Map", self.heat_map_widget, display_mode=default_mode, show_setting_option=True)
        self.heat_map_dock = Dock("Heat Map", widget=self.heat_map_widget, size=(50, 7), label=self.heat_map_dock_label)

        # Spectra plot

        default_mode = DockDisplayMode.Column
        self.spectra_widget = PlotWidget(self, default_mode=default_mode)
        self.spectra_dock_label = DockLabel("Spectra", self.spectra_widget, closable=True)
        self.spectra_dock = Dock("Spectra", widget=self.spectra_widget, size=(40, 7), label=self.spectra_dock_label)

        self.spectrum_widget = PlotWidget(self, default_mode=default_mode)
        self.spectrum_dock_label = DockLabel("Spectrum", self.spectrum_widget, display_mode=default_mode)
        self.spectrum_dock = Dock("Spectrum", widget=self.spectrum_widget, label=self.spectrum_dock_label)

        default_mode = DockDisplayMode.Row
        self.trace_widget = PlotWidget(self, default_mode=default_mode)
        self.trace_dock_label = DockLabel("Trace", self.trace_widget, display_mode=default_mode)
        self.trace_dock = Dock("Trace", widget=self.trace_widget, label=self.trace_dock_label)

        # data panel

        self.data_panel = DataPanel()
        self.settings_dock = Dock("Properties", widget=self.data_panel, size=(1, 1))

        # self.data_panel.txb_t0.focus_lost.connect(self.update_range)
        # self.data_panel.txb_t0.returnPressed.connect(self.update_range)
        # self.data_panel.txb_t1.focus_lost.connect(self.update_range)
        # self.data_panel.txb_t1.returnPressed.connect(self.update_range)
        # self.data_panel.txb_w0.focus_lost.connect(self.update_range)
        # self.data_panel.txb_w0.returnPressed.connect(self.update_range)
        # self.data_panel.txb_w1.focus_lost.connect(self.update_range)
        # self.data_panel.txb_w1.returnPressed.connect(self.update_range)
        # self.data_panel.txb_z0.focus_lost.connect(self.update_levels)
        # self.data_panel.txb_z0.returnPressed.connect(self.update_levels)
        # self.data_panel.txb_z1.focus_lost.connect(self.update_levels)
        # self.data_panel.txb_z1.returnPressed.connect(self.update_levels)
        #
        # self.data_panel.txb_n_spectra.setText(str(self.n_spectra))
        #
        # # self.data_panel.btn_crop_matrix.clicked.connect(self.btn_crop_matrix_clicked)
        # # self.data_panel.btn_restore_matrix.clicked.connect(self.btn_restore_matrix_clicked)
        # self.data_panel.txb_n_spectra.focus_lost.connect(self.txb_n_spectra_focus_lost)
        # self.data_panel.txb_n_spectra.returnPressed.connect(self.txb_n_spectra_focus_lost)
        # self.data_panel.btn_redraw_spectra.clicked.connect(self.update_spectra)
        #
        # self.data_panel.txb_SVD_filter.focus_lost.connect(self.txb_SVD_filter_changed)
        # self.data_panel.txb_SVD_filter.returnPressed.connect(self.txb_SVD_filter_changed)
        # self.data_panel.cb_SVD_filter.toggled.connect(self.cb_SVD_filter_toggled)
        #
        # self.data_panel.txb_ICA_filter.focus_lost.connect(self.txb_ICA_filter_changed)
        # self.data_panel.txb_ICA_filter.returnPressed.connect(self.txb_ICA_filter_changed)
        # self.data_panel.cb_ICA_filter.toggled.connect(self.cb_ICA_filter_toggled)
        #
        # # self.data_panel.btn_center_levels.clicked.connect(self.btn_center_levels_clicked)
        # self.data_panel.txb_SVD_filter.setText("1-5")
        # self.data_panel.btn_fit_chirp_params.clicked.connect(self.fit_chirp_params)
        # self.data_panel.cb_show_chirp_points.toggled.connect(self.cb_show_roi_checkstate_changed)

        # addition of docs

        self.addDock(self.heat_map_dock, 'left')
        self.addDock(self.spectra_dock, 'right')
        self.addDock(self.spectrum_dock, 'bottom')
        self.addDock(self.trace_dock, 'right', self.spectrum_dock)
        self.addDock(self.settings_dock, 'left', self.heat_map_dock)

        # self.roi = None
        # self.chirp = self.heat_map_widget.heat_map_plot.heatmap_pi.plot([])

        # self.heat_map_vline.sigPositionChangeFinished.connect(self.update_trace_and_spectrum)
        # self.heat_map_hline.sigPositionChangeFinished.connect(self.update_trace_and_spectrum)
        # self.spectrum_vline.sigPositionChangeFinished.connect(self.update_trace_and_spectrum)
        # self.trace_vline.sigPositionChangeFinished.connect(self.update_trace_and_spectrum)

    def set_axis_label(self, index: int, x_label='Wavelength / nm', y_label='Time / ps', z_label='\u0394A'):
        self.heat_map_widget.plots[index].heatmap_pi.setLabel('left', y_label)
        self.heat_map_widget.plots[index].heatmap_pi.setLabel('bottom', x_label)
        self.heat_map_widget.plots[index].hist.axis.setLabel(z_label)

        self.trace_widget.plots[index].plot_item.setLabel('left', z_label)
        self.trace_widget.plots[index].plot_item.setLabel('bottom', y_label)

        self.spectrum_widget.plots[index].plot_item.setLabel('left', z_label)
        self.spectrum_widget.plots[index].plot_item.setLabel('bottom', x_label)

    def HPLCMS_baseline_corr(self, MS_index=0, UV_index=1, threshold_TWC=300):
        uv = self.matrices[UV_index]
        ms = self.matrices[MS_index]
        twc = uv.get_TWC(axis=1)

        # downscale twc to match the MS data
        twc = np.interp(ms.times, uv.times, twc)

        idxs = twc < threshold_TWC  # indexes where to interpolate the MS data and then subtract from all of them
        y = ms.times[idxs]
        mat2interp = ms.Y[idxs, :]

        f = interp2d(ms.wavelengths, y, mat2interp, kind='linear', copy=True)
        D_interp = f(ms.wavelengths, ms.times)

        # baseline correct
        # D_new = ms.Y - D_interp

        # replace the value
        ms.Y -= D_interp
        ms.Yr = ms.Y

        self.heat_map_widget.plots[MS_index].set_matrix(ms.Y, ms.times, ms.wavelengths, center_lines=False)

    def set_common_dim(self, dim: CommonDimension):
        self.same_dimension = dim

    def get_roi_pos(self):
        """This shit took me half a day to figure out."""
        if self.roi is None:
            return

        hs = self.roi.getHandles()
        n = len(hs)

        positions = np.zeros((n, 2))
        for i, h in enumerate(self.roi.getHandles()):
            qPoint = self.roi.mapSceneToParent(h.scenePos())

            positions[i, 0] = self.heat_map_widget.transform_wl_pos(qPoint.x())
            positions[i, 1] = self.heat_map_widget.transform_t_pos(qPoint.y())

        return positions

    def cb_show_roi_checkstate_changed(self):
        if self.roi is None:
            return

        val = 1000 if self.data_panel.cb_show_chirp_points.isChecked() else -1000
        self.roi.setZValue(val)

    def plot_chirp_points(self):
        if self.roi is None:

            matrix = self.matrices[0]

            t_mid = (matrix.times[-1] - matrix.times[0]) / 2
            n_w = matrix.wavelengths.shape[0] - 1
            wls = matrix.wavelengths[int(n_w / 5)], matrix.wavelengths[int(2 * n_w / 5)], \
                  matrix.wavelengths[int(3 * n_w / 5)], matrix.wavelengths[int(4 * n_w / 5)]
            self.roi = pg.PolyLineROI([[wls[0], t_mid], [wls[1], t_mid], [wls[2], t_mid], [wls[3], t_mid]], closed=False,
                           handlePen=pg.mkPen(color=(0, 255, 0), width=5),
                           hoverPen=pg.mkPen(color=(0, 150, 0), width=2),
                           handleHoverPen=pg.mkPen(color=(0, 150, 0), width=3))

            self.heat_map_widget.plots[0].addItem(self.roi)

    def add_chirp(self, wls,  mu):  # plots the chirp
        pen = pg.mkPen(color=QColor('black'), width=2)
        mu_tr = self.heat_map_widget.inv_transform_t_pos(mu)
        wls_tr = self.heat_map_widget.inv_transform_wl_pos(wls)
        self.chirp.setData(wls_tr, mu_tr, pen=pen)

    def fit_chirp_params(self):
        from Widgets.fitwidget import FitWidget as _fw

        if _fw.instance is None:
            return

        fw = _fw.instance

        if fw.current_model._class != 'Femto':
            return

        roi_pos = self.get_roi_pos()
        x, y = roi_pos[:, 0], roi_pos[:, 1]
        lambda_c = fw.current_model.get_lambda_c()

        if fw.current_model.chirp_type == 'exp':
            mul, lam = fit_sum_exp(x - lambda_c, y, fw.current_model.n_exp_chirp, fit_intercept=True)
            parmu = [mul[-1]] + [entry for tup in zip(mul[:-1], lam[:-1]) for entry in tup]
            fw.current_model.set_parmu(parmu, 'exp')
        else:
            n = fw.current_model.n_poly_chirp + 1
            parmu = fit_polynomial_coefs(x - lambda_c, y, n)
            fw.current_model.set_parmu(parmu, 'poly')

        fw.update_model_par_count(update_after_fit=True)

    def use_mask(self):
        if self.matrix is None:
            return

        self.matrix.Mask = not self.matrix.Mask
        self.plot_matrices(self.matrix, center_lines=False, keep_range=True, keep_fits=True)

    def cb_SVD_filter_toggled(self):
        if self.matrix is None:
            return

        self.matrix.SVD_filter = self.data_panel.cb_SVD_filter.isChecked()
        # self.plot_matrix(self.matrix, center_lines=False, keep_range=True)
        if self.data_panel.cb_SVD_filter.isChecked():
            self.txb_SVD_filter_changed()
        else:
            self.plot_matrices(self.matrix, center_lines=False, keep_range=True, keep_fits=True)

    def cb_ICA_filter_toggled(self):
        if self.matrix is None:
            return

        self.matrix.ICA_filter = self.data_panel.cb_ICA_filter.isChecked()
        if self.data_panel.cb_ICA_filter.isChecked():
            self.txb_ICA_filter_changed()
        else:
            self.plot_matrices(self.matrix, center_lines=False, keep_range=True, keep_fits=True)

    def txb_ICA_filter_changed(self):
        if self.matrix is None:
            return
        r_text = self.data_panel.txb_ICA_filter.text()
        vals = list(filter(None, r_text.split(',')))  # splits by comma and removes empty entries
        int_vals = []

        for val in vals:
            try:
                if str_is_integer(val):
                    int_vals.append(int(val) - 1)
                else:
                    # we dont have a single number, but in a format of eg. '1-3'

                    if '-' not in val:
                        continue

                    split = val.split('-')
                    x0 = int(split[0])
                    x1 = int(split[1])

                    int_vals += [i - 1 for i in range(x0, x1 + 1)]
            except:
                continue

        n_comp = int(SVDDockArea.instance.data_panel.sb_n_ICA.value())

        result = sorted(list(filter(lambda item: 0 <= item < n_comp, int_vals)))
        self.matrix.set_ICA_filter(result, n_components=n_comp)

        if self.data_panel.cb_ICA_filter.isChecked():
            self.plot_matrices(self.matrix, center_lines=False, keep_range=True, keep_fits=True)

    def txb_SVD_filter_changed(self):

        if self.matrix is None:
            return

        r_text = self.data_panel.txb_SVD_filter.text()

        # format - values separated by comma, eg. '1, 2, 3', '1-4, -3'

        vals = list(filter(None, r_text.split(',')))  # splits by comma and removes empty entries

        int_vals = []
        remove_vals = []

        for val in vals:
            try:
                if str_is_integer(val):
                    int_val = int(val)
                    if int_val > 0:
                        int_vals.append(int_val - 1)
                    else:
                        remove_vals.append(-1 * int_val - 1)  #put negative values into different list as positives
                else:
                    # we dont have a single number, but in a format of eg. '1-3'

                    if '-' not in val:
                        continue

                    split = val.split('-')
                    x0 = int(split[0])
                    x1 = int(split[1])

                    int_vals += [i - 1 for i in range(x0, x1 + 1)]
            except:
                continue

        result = sorted(list(set(int_vals) - set(remove_vals)))

        if not(result is None or len(result) == 0):
            self.matrix.set_SVD_filter(result)

        if self.data_panel.cb_SVD_filter.isChecked():
            self.plot_matrices(self.matrix, center_lines=False, keep_range=True, keep_fits=True)

    def txb_n_spectra_focus_lost(self):
        try:
            n = int(self.data_panel.txb_n_spectra.text())
            self.n_spectra = max(2, n)

            self.data_panel.txb_n_spectra.setText(str(self.n_spectra))

            self.update_spectra()

        except ValueError:
            pass

    def trace_spectrum_range_changed(self, vb, range):
        if self.heatmap_range_lock:
            return
        self.heatmap_range_lock = True

        if vb == self.spectrum.getViewBox():
            w0 = self.heat_map_widget.inv_transform_wl_pos(range[0][0])
            w1 = self.heat_map_widget.inv_transform_wl_pos(range[0][1])

            self.heat_map_widget.heat_map_plot.getViewBox().setXRange(w0, w1, padding=0)
            self.set_txb_ranges(w0=range[0][0], w1=range[0][1])
        else:
            t0 = self.heat_map_widget.inv_transform_t_pos(range[0][0])
            t1 = self.heat_map_widget.inv_transform_t_pos(range[0][1])

            self.heat_map_widget.heat_map_plot.getViewBox().setYRange(t0, t1, padding=0)
            self.set_txb_ranges(t0=range[0][0], t1=range[0][1])

        self.heatmap_range_lock = False

    def get_selected_range(self):
        if self.matrix is None:
            return

        try:
            t0, t1 = float(self.data_panel.txb_t0.text()), float(self.data_panel.txb_t1.text())
            w0, w1 = float(self.data_panel.txb_w0.text()), float(self.data_panel.txb_w1.text())

            if t0 > t1 or w0 > w1:
                return
        except ValueError:
            return

        return w0, w1, t0, t1

    def set_txb_ranges(self, w0=None, w1=None, t0=None, t1=None):
        if w0 is not None:
            self.data_panel.txb_w0.setText(f'{w0:.4g}')
        if w1 is not None:
            self.data_panel.txb_w1.setText(f'{w1:.4g}')
        if t0 is not None:
            self.data_panel.txb_t0.setText(f'{t0:.4g}')
        if t1 is not None:
            self.data_panel.txb_t1.setText(f'{t1:.4g}')

    # def heat_map_range_changed(self, vb, range):
    #     if self.heatmap_range_lock or self.matrix is None:
    #         return
    #     self.heatmap_range_lock = True
    #
    #     w0, w1, t0, t1 = range[0][0], range[0][1], range[1][0], range[1][1]
    #
    #     t0 = self.heat_map_widget.transform_t_pos(t0)  # transform t positions
    #     t1 = self.heat_map_widget.transform_t_pos(t1)
    #
    #     w0 = self.heat_map_widget.transform_wl_pos(w0)  # transform t positions
    #     w1 = self.heat_map_widget.transform_wl_pos(w1)
    #
    #     self.set_txb_ranges(w0, w1, t0, t1)
    #
    #     self.spectrum.getViewBox().setXRange(w0, w1, padding=0)
    #     self.trace.getViewBox().setXRange(t0, t1, padding=0)
    #
    #     # keep all the v and h lines inside the visible area
    #
    #     v_pos = self.heat_map_vline.pos()[0]
    #     h_pos = self.heat_map_hline.pos()[1]
    #
    #     if not range[0][0] <= v_pos <= range[0][1]:
    #         self.heat_map_vline.setPos(range[0][0] if np.abs(v_pos - range[0][0]) < np.abs(v_pos - range[0][1]) else range[0][1])
    #
    #     if not range[1][0] <= h_pos <= range[1][1]:
    #         self.heat_map_hline.setPos(range[1][0] if np.abs(h_pos - range[1][0]) < np.abs(h_pos - range[1][1]) else range[1][1])
    #
    #     it0, it1 = find_nearest_idx(self.matrix.times, t0), find_nearest_idx(self.matrix.times, t1) + 1
    #     iw0, iw1 = find_nearest_idx(self.matrix.wavelengths, w0), find_nearest_idx(self.matrix.wavelengths, w1) + 1
    #
    #     self.selected_range_idxs = (it0, it1, iw0, iw1)
    #
    #     self.data_panel.lbl_visible_area_msize.setText(f'{it1 - it0} x {iw1 - iw0}')
    #
    #     self.heatmap_range_lock = False

    def heat_map_levels_changed(self, hist):
        z_levels = self.heat_map_widget.get_z_range()
        self.data_panel.txb_z0.setText(f'{z_levels[0]:.4g}')
        self.data_panel.txb_z1.setText(f'{z_levels[1]:.4g}')

    def update_range(self):
        if self.heatmap_range_lock or self.matrix is None:
            return

        try:
            t0, t1 = float(self.data_panel.txb_t0.text()), float(self.data_panel.txb_t1.text())
            w0, w1 = float(self.data_panel.txb_w0.text()), float(self.data_panel.txb_w1.text())

            if t0 <= t1 and w0 <= w1:
                self.heat_map_widget.set_xy_range(w0, w1, t0, t1, 0)
        except ValueError:
            pass
        except AttributeError:
            pass

    def update_levels(self):
        try:
            z0, z1 = float(self.data_panel.txb_z0.text()), float(self.data_panel.txb_z1.text())

            if z0 <= z1:
                self.heat_map_widget.hist.setLevels(z0, z1)

        except ValueError:
            pass

    def btn_center_levels_clicked(self):
        z0, z1 = self.heat_map_widget.get_z_range()
        diff = z1 - z0
        self.heat_map_widget.hist.setLevels(-diff / 2, diff / 2)

    def btn_crop_matrix_clicked(self):
        if self.matrix is None:
            return

        w0, w1, t0, t1 = self.get_selected_range()

        self.matrix.crop_data(t0, t1, w0, w1)

        SVDDockArea.instance.set_data(self.matrix)
        self.cb_SVD_filter_toggled()

        # self.plot_matrix(self.matrix, False)

    def crop_matrix(self, t0=None, t1=None, w0=None, w1=None):
        if self.matrix is None:
            return

        self.matrix.crop_data(t0, t1, w0, w1)

        SVDDockArea.instance.set_data(self.matrix)
        self.cb_SVD_filter_toggled()

    def baseline_correct(self, t0=0, t1=0.2):
        if self.matrix is None:
            return

        self.matrix.baseline_corr(t0, t1)

        SVDDockArea.instance.set_data(self.matrix)
        self.cb_SVD_filter_toggled()

    def dimension_mul(self, t_mul=1, w_mul=1):
        if self.matrix is None:
            return

        self.matrix.times *= t_mul
        self.matrix.wavelengths *= w_mul

        SVDDockArea.instance.set_data(self.matrix)
        self.cb_SVD_filter_toggled()

    def btn_restore_matrix_clicked(self):
        if self.matrix is None:
            return
        self.matrix.restore_original_data()
        self.plot_matrices(self.matrix, False)

    # def init_trace_and_spectrum(self):
    #
    #     self.trace_plot_item = self.trace.plot([])
    #     self.spectrum_plot_item = self.spectrum.plot([])
    #
    #     self.trace_orig_plot_item = self.trace.plot([])
    #     self.spectrum_orig_plot_item = self.spectrum.plot([])
    #
    # def init_fit_trace_sp(self):
    #     self.trace_plot_item_fit = self.trace.plot([])
    #     self.spectrum_plot_item_fit = self.spectrum.plot([])

    def update_trace_and_spectrum(self, sender=None):
        if self.matrices is None or self.heatmap_line_lock or self.heatmap_range_lock:
            return

        x_positions, y_positions = self.heat_map_widget.get_positions()

        pen = pg.mkPen(color=QColor('black'), width=1)

        for i, m in enumerate(self.matrices):
            x_idx = find_nearest_idx(m.wavelengths, x_positions[i])
            y_idx = find_nearest_idx(m.times, y_positions[i])

            try:
                self.plotted_traces[i].setData(m.times, m.D[:, x_idx], pen=pen)
                self.plotted_spectra[i].setData(m.wavelengths, m.D[y_idx], pen=pen)

            except Exception as e:
                print(x_idx, y_idx, e)

        # time_pos = self.heat_map_hline.pos()[1]
        # time_pos = self.heat_map_widget.transform_t_pos(time_pos)
        # wl_pos = self.heat_map_vline.pos()[0]
        # wl_pos = self.heat_map_widget.transform_wl_pos(wl_pos)

        # wavelengths = self.matrix.wavelengths
        # times = self.matrix.times

        # t_idx = find_nearest_idx(times, time_pos)
        # wl_idx = find_nearest_idx(wavelengths, wl_pos)
        #
        # trace_y_data = self.matrix.D[:, wl_idx]
        # trace_y_datorig = self.matrix.Y[:, wl_idx]
        #
        # pen = pg.mkPen(color=QColor('black'), width=1)
        # pen_orig_data = pg.mkPen(color=QColor('blue'), width=1)
        #
        # spectrum_y_data = self.matrix.D[t_idx, :]
        # spectrum_y_data_orig = self.matrix.Y[t_idx, :]

        # if self.smooth_count == 0:
        #     spectrum_y_data = self.matrix.D[t_idx, :]
        #     spectrum_y_data_orig = self.matrix.Y[t_idx, :]
        # else:
        #     avrg_slice_matrix = self.matrix.D[t_idx - self.smooth_count:t_idx + self.smooth_count, :]
        #     spectrum_y_data = np.average(avrg_slice_matrix, axis=0)

        # if self.trace_plot_item is not None:
        #     self.trace_plot_item.setData(times, trace_y_data, pen=pen)
        #     self.spectrum_plot_item.setData(wavelengths, spectrum_y_data, pen=pen)
        #     if not np.allclose(trace_y_data, trace_y_datorig):
        #         self.trace_orig_plot_item.setData(times, trace_y_datorig, pen=pen_orig_data)
        #         self.spectrum_orig_plot_item.setData(wavelengths, spectrum_y_data_orig, pen=pen_orig_data)
        #     else:
        #         self.trace_orig_plot_item.setData([])
        #         self.spectrum_orig_plot_item.setData([])

        # plotting fit
        # if self.spectrum_plot_item_fit is not None and self.trace_plot_item_fit is not None and self.fit_matrix is not None:
        #     trace_y_data_fit = self.fit_matrix.Y[:, wl_idx]
        #
        #     if self.smooth_count == 0:
        #         spectrum_y_data_fit = self.fit_matrix.Y[t_idx, :]
        #     else:
        #         avrg_slice_matrix_fit = self.fit_matrix.Y[t_idx - self.smooth_count:t_idx + self.smooth_count, :]
        #         spectrum_y_data_fit = np.average(avrg_slice_matrix_fit, axis=0)
        #
        #     pen_fit = pg.mkPen(color=QColor('red'), width=1)
        #     self.trace_plot_item_fit.setData(times, trace_y_data_fit, pen=pen_fit)
        #     self.spectrum_plot_item_fit.setData(wavelengths, spectrum_y_data_fit, pen=pen_fit)

    def set_fit_matrix(self, fit_matrix):
        self.fit_matrix = fit_matrix

        self.spectrum.clearPlots()
        self.trace.clearPlots()

        self.init_trace_and_spectrum()
        self.init_fit_trace_sp()
        self.update_trace_and_spectrum()

        self.spectrum.autoBtnClicked()
        self.trace.autoBtnClicked()

    def update_spectra(self):
        if self.matrix is None:
            return

        self.spectra_plot.clearPlots()

        tup = self.get_selected_range()
        if tup is None:
            return

        w0, w1, t0, t1 = tup

        D_crop, times, wavelengths = crop_data(self.matrix.D, self.matrix.times, self.matrix.wavelengths, t0,
                                               t1, w0, w1)

        for i in range(self.n_spectra):
            sp = D_crop[int(i * D_crop.shape[0] / self.n_spectra)]
            color = pg.intColor(i, hues=self.n_spectra, values=1, maxHue=360, minHue=0)
            pen = pg.mkPen(color=color, width=1)
            self.spectra_plot.plot(wavelengths, sp, pen=pen)

    def heatmap_v_line_changed(self, sender_heatmap):

        if self.heatmap_line_lock:
            return

        self.heatmap_line_lock = True

        x_pos_orig = sender_heatmap.get_xpos()
        x_pos = sender_heatmap.transform_wl_pos(x_pos_orig)

        i = self.heat_map_widget.plots.index(sender_heatmap)
        self.spectrum_widget.plots[i].set_xpos(x_pos)

        if self.same_dimension in (CommonDimension.Second, CommonDimension.Both):
            for h, spectrum in zip(self.heat_map_widget.plots, self.spectrum_widget.plots):
                h.set_xpos(x_pos_orig)  # set the original value without transformation
                spectrum.set_xpos(x_pos)

        self.heatmap_line_lock = False

    def heatmap_h_line_changed(self, sender_heatmap):

        if self.heatmap_line_lock:
            return

        self.heatmap_line_lock = True

        y_pos_orig = sender_heatmap.get_ypos()
        y_pos = sender_heatmap.transform_t_pos(y_pos_orig)

        i = self.heat_map_widget.plots.index(sender_heatmap)
        self.trace_widget.plots[i].set_xpos(y_pos)

        if self.same_dimension in (CommonDimension.First, CommonDimension.Both):
            for h, trace in zip(self.heat_map_widget.plots, self.trace_widget.plots):
                h.set_ypos(y_pos_orig)  # set the original value without transformation
                trace.set_xpos(y_pos)

        self.heatmap_line_lock = False

    # def set_txb_ranges(self, index, w0=None, w1=None, t0=None, t1=None):
    #     if w0 is not None:
    #         self.data_panel.txb_w0.setText(f'{w0:.4g}')
    #     if w1 is not None:
    #         self.data_panel.txb_w1.setText(f'{w1:.4g}')
    #     if t0 is not None:
    #         self.data_panel.txb_t0.setText(f'{t0:.4g}')
    #     if t1 is not None:
    #         self.data_panel.txb_t1.setText(f'{t1:.4g}')

    def heatmap_range_changed(self, sender_heatmap, sender_vb, ranges, changes):
        """ params: sender heatmap, sender viewbox, [[x0, x1], [y0, y1]], [change in x, change in y]"""

        if self.heatmap_range_lock or self.matrices is None:
            return

        self.heatmap_range_lock = True

        x0, x1, y0, y1 = ranges[0][0], ranges[0][1], ranges[1][0], ranges[1][1]
        change_x, change_y = changes

        i = self.heat_map_widget.plots.index(sender_heatmap)

        if change_x:
            x0_tr = sender_heatmap.transform_wl_pos(x0)
            x1_tr = sender_heatmap.transform_wl_pos(x1)

            self.spectrum_widget.plots[i].set_x_range(x0_tr, x1_tr)
            self.data_panel.set_range(i, x0=x0_tr, x1=x1_tr)

            if self.same_dimension in (CommonDimension.Second, CommonDimension.Both):
                for i, h in enumerate(self.heat_map_widget.plots):
                    h.set_x_range(x0, x1)
                    self.spectrum_widget.plots[i].set_x_range(x0_tr, x1_tr)
                    self.data_panel.set_range(i, x0=x0_tr, x1=x1_tr)

        if change_y:
            y0_tr = sender_heatmap.transform_t_pos(y0)
            y1_tr = sender_heatmap.transform_t_pos(y1)

            self.trace_widget.plots[i].set_x_range(y0_tr, y1_tr)
            self.data_panel.set_range(i, y0=y0_tr, y1=y1_tr)

            if self.same_dimension in (CommonDimension.First, CommonDimension.Both):
                for i, h in enumerate(self.heat_map_widget.plots):
                    h.set_y_range(y0, y1)
                    self.trace_widget.plots[i].set_x_range(y0_tr, y1_tr)
                    self.data_panel.set_range(i, y0=y0_tr, y1=y1_tr)

        # # keep all the v and h lines inside the visible area
        # untransformed positions
        x_pos, y_pos = sender_heatmap.get_xpos(), sender_heatmap.get_ypos()

        if not x0 <= x_pos <= x1:
            new_xpos = x0 if np.abs(x_pos - x0) < np.abs(x_pos - x1) else x1
            sender_heatmap.set_xpos(new_xpos)

        if not y0 <= y_pos <= y1:
            new_ypos = y0 if np.abs(y_pos - y0) < np.abs(y_pos - y1) else y1
            sender_heatmap.set_ypos(new_ypos)

        # it0, it1 = find_nearest_idx(self.matrix.times, t0), find_nearest_idx(self.matrix.times, t1) + 1
        # iw0, iw1 = find_nearest_idx(self.matrix.wavelengths, w0), find_nearest_idx(self.matrix.wavelengths, w1) + 1
        #
        # self.selected_range_idxs = (it0, it1, iw0, iw1)
        #
        # self.data_panel.lbl_visible_area_msize.setText(f'{it1 - it0} x {iw1 - iw0}')
        #
        self.heatmap_range_lock = False

    def trace_range_changed(self, sender_trace, sender_vb, ranges, changes):
        x0, x1 = ranges[0][0], ranges[0][1]
        change_x, _ = changes

        if change_x:
            i = self.trace_widget.plots.index(sender_trace)
            h = self.heat_map_widget.plots[i]
            h.set_y_range(h.inv_transform_t_pos(x0), h.inv_transform_t_pos(x1))

    def spectrum_range_changed(self, sender_spectrum, sender_vb, ranges, changes):
        x0, x1 = ranges[0][0], ranges[0][1]
        change_x, _ = changes

        if change_x:
            i = self.spectrum_widget.plots.index(sender_spectrum)
            h = self.heat_map_widget.plots[i]
            h.set_x_range(h.inv_transform_wl_pos(x0), h.inv_transform_wl_pos(x1))

    def trace_v_line_changed(self, sender_trace):
        y_pos = sender_trace.get_xpos()
        i = self.trace_widget.plots.index(sender_trace)
        h = self.heat_map_widget.plots[i]
        h.set_ypos(h.inv_transform_t_pos(y_pos))

    def spectrum_v_line_changed(self, sender_spectrum):
        x_pos = sender_spectrum.get_xpos()
        i = self.spectrum_widget.plots.index(sender_spectrum)
        h = self.heat_map_widget.plots[i]
        h.set_xpos(h.inv_transform_wl_pos(x_pos))

    def plot_matrices(self, matrices, center_lines=True, keep_ranges=False, keep_fits=False):

        # w_range, t_range = self.heat_map_widget.heat_map_plot.getViewBox().viewRange()
        # z_range = self.heat_map_widget.get_z_range()

        # self.spectrum.clearPlots()
        # self.trace.clearPlots()
        # self.spectra_plot.clearPlots()

        # self.plotted_spectra = []
        # self.plotted_traces = []

        # self.heat_map_widget.clear_plots()
        # self.spectrum_widget.clear_plots()
        # self.trace_widget.clear_plots()
        self.same_dimension = CommonDimension.Not

        self.matrices = matrices

        n = len(self.matrices)

        atol = 0.1
        rtol = 0.1

        # determine the same dimension of the loaded matrices
        if n > 1:
            # ax0_shapes = [m.D.shape[0] for m in self.matrices]
            # ax1_shapes = [m.D.shape[1] for m in self.matrices]

            ax_0_starts = np.asarray([m.times[0] for m in self.matrices])
            ax_0_ends = np.asarray([m.times[-1] for m in self.matrices])

            ax_1_starts = np.asarray([m.wavelengths[0] for m in self.matrices])
            ax_1_ends = np.asarray([m.wavelengths[-1] for m in self.matrices])

            if np.allclose(ax_0_starts, ax_0_starts[0], rtol, atol) and \
               np.allclose(ax_0_ends, ax_0_ends[0], rtol, atol):
                self.same_dimension = CommonDimension.First

            if np.allclose(ax_1_starts, ax_1_starts[0], rtol, atol) and \
               np.allclose(ax_1_ends, ax_1_ends[0], rtol, atol):
                self.same_dimension = CommonDimension.First if self.same_dimension is None else CommonDimension.Both  # both are the same

            # # test if matrices has common dimensions
            # if np.all(ax0_shapes == ax0_shapes[0]):
            #     ax0_all_close = [np.allclose(m.times, self.matrices[0].times) for m in self.matrices[1:]]
            #     if np.all(ax0_all_close == ax0_all_close[0]):
            #         self.same_dimension = 0
            #
            # if np.all(ax1_shapes == ax1_shapes[0]):
            #     ax1_all_close = [np.allclose(m.wavelengths, self.matrices[0].wavelengths) for m in self.matrices[1:]]
            #     if np.all(ax1_all_close == ax1_all_close[0]):
            #         self.same_dimension = 1

        # if self.matrix.original_data_matrix is not None:
        #     self.data_panel.lbl_matrix_size.setText(f'{matrix.original_data_matrix.shape[0] - 1} x {matrix.original_data_matrix.shape[1] - 1}')
        #
        # self.data_panel.lbl_cr_matrix_size.setText(f'{matrix.D.shape[0]} x {matrix.D.shape[1]}')

        self.heat_map_widget.set_heatmaps(self.matrices, center_lines=center_lines, keep_ranges=keep_ranges)

        self.spectrum_widget.set_data(self.matrices, axis=1, title_prefix='Spectrum [')
        self.trace_widget.set_data(self.matrices, axis=0, title_prefix='Trace [')

        self.plotted_spectra = self.spectrum_widget.get_plotted_data()
        self.plotted_traces = self.trace_widget.get_plotted_data()

        self.heat_map_widget.connect_v_lines_position_changed(self.heatmap_v_line_changed)
        self.heat_map_widget.connect_h_lines_position_changed(self.heatmap_h_line_changed)

        self.heat_map_widget.connect_v_lines_position_changed(self.update_trace_and_spectrum)
        self.heat_map_widget.connect_h_lines_position_changed(self.update_trace_and_spectrum)

        # not necessary because the positions are taken from heatmap lines
        # self.trace_widget.connect_v_lines_position_changed(self.update_trace_and_spectrum)
        # self.spectrum_widget.connect_v_lines_position_changed(self.update_trace_and_spectrum)

        self.trace_widget.connect_v_lines_position_changed(self.trace_v_line_changed)
        self.spectrum_widget.connect_v_lines_position_changed(self.spectrum_v_line_changed)

        self.heat_map_widget.connect_ranges_changed(self.heatmap_range_changed)
        self.trace_widget.connect_ranges_changed(self.trace_range_changed)
        self.spectrum_widget.connect_ranges_changed(self.spectrum_range_changed)

        for i in range(len(self.matrices)):
            self.set_axis_label(i)

        # self.spectra_vline.sigPositionChanged.connect(update_heat_lines_spectra)
        # self.plotted_spectra.connect_v_lines_position_changed(self.update_trace_and_spectrum)
        # self.plotted_traces.connect_v_lines_position_changed(self.update_trace_and_spectrum)

        # update lines
        for h in self.heat_map_widget.plots:
            self.heatmap_v_line_changed(h)
            self.heatmap_h_line_changed(h)

        self.data_panel.initialize(self.matrices)

        self.heat_map_widget.autoscale()
        self.update_trace_and_spectrum()

        self.heat_map_widget.set_lines_color()
        self.spectrum_widget.set_lines_color()
        self.trace_widget.set_lines_color()

        # if keep_fits:
        #     self.init_fit_trace_sp()
        #
        # # redraw trace and spectrum figures
        # self.update_trace_and_spectrum()
        #
        # self.update_spectra()
        #
        # self.plot_chirp_points()
        # self.cb_show_roi_checkstate_changed()

    # def save_plot_to_clipboard_as_png(self, plot_item):
    #     self.img_exporter = ImageExporter(plot_item)
    #     self.img_exporter.export(copy=True)
    #
    # def save_plot_to_clipboard_as_svg(self, plot_item):
    #     self.svg_exporter = SVGExporter(plot_item)
    #     self.svg_exporter.export(copy=True)

