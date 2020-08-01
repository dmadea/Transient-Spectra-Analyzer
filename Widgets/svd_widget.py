import numpy as np
import pyqtgraph as pg
from pyqtgraph.dockarea import Dock, DockArea
from .datapanel_svd import DataPanelSVD
from .fit_layout import FitLayout
from scipy.linalg import svd
from misc import crop_data, set_axes
from LFP_matrix import LFP_matrix

# from PyQt5.QtCore import Qt


class SVDWidget(DockArea):
    instance = None
    t_unit = 'ps'
    w_unit = 'nm'

    def __init__(self, parent=None):
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOptions(antialias=True)

        super(SVDWidget, self).__init__(parent)

        SVDWidget.instance = self

        self.matrix = None
        self.times = None
        self.wavelengths = None
        self.U = None
        self.V_T = None
        self.S = None
        self.D = None

        self.f_EFA_log_sing_vals = None
        self.t_idxs_fEFA = None
        self.EFA_vectors = None

        # SVD data panel

        self.data_panel = DataPanelSVD()
        self.data_panel_dock = Dock("Properties", widget=self.data_panel, size=(1, 10))

        # log singular values

        w_log_singvals = pg.PlotWidget()
        self.singvals_plot = w_log_singvals.plotItem
        set_axes(self.singvals_plot, title="Singular values",
                 x_label='Singular value index',
                 y_label='log<sub>10</sub>( Magnitude )')

        self.singvals_dock = Dock("Singular values", widget=w_log_singvals, size=(5, 10))

        # EFA plot

        w_EFA = pg.PlotWidget()
        self.EFA_plot = w_EFA.plotItem
        set_axes(self.EFA_plot, title="Evolving factor analysis",
                 x_label=f'Time / {self.t_unit}',
                 y_label='log<sub>10</sub>( Magnitude )')

        self.EFA_plot_dock = Dock("EFA", widget=w_EFA, size=(5, 10))

        # Left singular vectors plot

        w_left_sv = pg.PlotWidget()
        self.left_sv = w_left_sv.plotItem
        set_axes(self.left_sv, title="Left singular vectors",
                 x_label=f'Time / {self.t_unit}',
                 y_label='')

        self.left_sv_dock = Dock("Left singular vectors", widget=w_left_sv)

        # Right singular vectors plot

        w_right_sv = pg.PlotWidget()
        self.right_sv = w_right_sv.plotItem
        set_axes(self.right_sv, title="Right singular vectors",
                 x_label=f'Wavelength / {self.w_unit}',
                 y_label='')

        self.right_sv_dock = Dock("Right singular vectors", widget=w_right_sv)

        # addition of docs

        self.addDock(self.data_panel_dock, 'left')
        self.addDock(self.singvals_dock, 'right')
        self.addDock(self.EFA_plot_dock, 'right')
        self.addDock(self.left_sv_dock, 'bottom')
        self.addDock(self.right_sv_dock, 'right', self.left_sv_dock)

        self.data_panel.sb_n_vectors.valueChanged.connect(self.redraw_plots)
        self.data_panel.cb_show_all.toggled.connect(self.redraw_plots)
        self.data_panel.sb_n_svals.valueChanged.connect(self.redraw_fEFA_vals)

        self.data_panel.cb_SVD.toggled.connect(self.SVD_from_selection_toggled)
        self.data_panel.btn_fEFA.clicked.connect(self.fEFA_clicked)

    def fEFA_clicked(self):
        if self.matrix is None:
            return

        points = min(int(self.data_panel.sb_n_t_points.value()), self.D.shape[0])
        s_nums = min(self.D.shape[0], self.D.shape[1])

        s_vals, self.EFA_vectors, self.t_idxs_fEFA = LFP_matrix._fEFA(self.D, s_nums, points)
        self.f_EFA_log_sing_vals = np.log10(s_vals)

        self.redraw_fEFA_vals()

    def redraw_fEFA_vals(self):
        if self.matrix is None or self.f_EFA_log_sing_vals is None:
            return

        _fl = self.f_EFA_log_sing_vals.flatten()
        _fl = _fl[~np.isnan(_fl)]  # remove nan values to calcuate min and max values

        _min = _fl.min()
        _max = _fl.max()

        idxs = self.t_idxs_fEFA
        s_nums = min(int(self.data_panel.sb_n_svals.value()), self.f_EFA_log_sing_vals.shape[1])

        self.EFA_plot.clearPlots()
        self.EFA_plot.getViewBox().setLimits(xMin=self.times[idxs[0]], xMax=self.times[idxs[-1]],
                                             yMin=1.2 ** np.sign(-_min) * _min,
                                             yMax=1.2 ** np.sign(_max) * _max)

        for i in range(s_nums):
            pen = pg.mkPen(color=FitLayout.int_default_color(i), width=1)
            idx_not_nan = np.count_nonzero(np.isnan(self.f_EFA_log_sing_vals[:, i]))
            self.EFA_plot.plot(self.times[idxs][idx_not_nan:], self.f_EFA_log_sing_vals[idx_not_nan:, i], pen=pen)

    def SVD_from_selection_toggled(self):
        if self.matrix is None:
            return

        from plotwidget import PlotWidget
        wwtt = PlotWidget.instance.get_selected_range() if self.data_panel.cb_SVD.isChecked() else None
        self.set_data(self.matrix, wwtt=wwtt)

    def redraw_plots(self):
        if self.matrix is None:
            return

        self.left_sv.clearPlots()
        self.right_sv.clearPlots()

        n_vectors = min(int(self.data_panel.sb_n_vectors.value()), self.S.shape[0])
        show_all = self.data_panel.cb_show_all.isChecked()

        if show_all:
            for i in range(n_vectors):
                pen_vector = pg.mkPen(color=FitLayout.int_default_color(i), width=1)
                self.left_sv.plot(self.times, self.U[:, i], pen=pen_vector)
                self.right_sv.plot(self.wavelengths, self.V_T[i, :], pen=pen_vector)
        else:
            pen_vector = pg.mkPen(color=FitLayout.int_default_color(0), width=1)
            self.left_sv.plot(self.times, self.U[:, n_vectors - 1], pen=pen_vector)
            self.right_sv.plot(self.wavelengths, self.V_T[n_vectors - 1, :], pen=pen_vector)

    def set_data(self, matrix, wwtt=None):
        if matrix is None:
            return
        self.matrix = matrix

        self.singvals_plot.clearPlots()
        self.left_sv.clearPlots()
        self.right_sv.clearPlots()

        if wwtt is not None:
            w0, w1, t0, t1 = wwtt
            self.D, self.times, self.wavelengths = crop_data(self.matrix.Y, self.matrix.times, self.matrix.wavelengths, t0, t1, w0, w1)
            self.U, self.S, self.V_T = svd(self.D, full_matrices=False, lapack_driver='gesdd')
        else:
            self.times = self.matrix.times
            self.wavelengths = self.matrix.wavelengths
            self.U = self.matrix.U
            self.V_T = self.matrix.V_T
            self.S = self.matrix.S
            self.D = self.matrix.Y

        self.data_panel.sb_n_vectors.setMaximum(self.S.shape[0])

        _min = np.log10(self.S.min())  # shit... log scale does not work in pyqtgraph or only partially....
        _max = np.log10(self.S.max())

        self.singvals_plot.getViewBox().setLimits(xMin=0, xMax=self.S.shape[0] + 1,
                                             yMin=1.2 ** np.sign(-_min) * _min,
                                              yMax=1.2 ** np.sign(_max) * _max)

        self.left_sv.getViewBox().setLimits(xMin=self.times[0], xMax=self.times[-1],
                                                 yMin=self.U.min(), yMax=self.U.max())

        self.right_sv.getViewBox().setLimits(xMin=self.wavelengths[0], xMax=self.wavelengths[-1],
                                          yMin=self.V_T.min(), yMax=self.V_T.max())

        self.singvals_plot.plot(np.arange(self.S.shape[0]) + 1, np.log10(self.S), symbol='o',
                                pen=None)

        # redraw data
        self.redraw_plots()
















