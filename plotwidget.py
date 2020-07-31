from PyQt5 import QtCore, QtGui, QtWidgets

import pyqtgraph as pg
from pyqtgraph.exporters import SVGExporter, ImageExporter
from pyqtgraph.dockarea import Dock, DockArea
from settings import Settings

from plot.legend_item import LegendItem
# from plot.img_exporter import ImgExporter

from PyQt5 import QtCore, QtGui

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from misc import find_nearest_idx, find_nearest, crop_data


from PyQt5.QtCore import *

from Widgets.fit_layout import HeatMapPlot

from PyQt5.QtCore import Qt

# from pyqtgraph.graphicsItems.LegendItem import LegendItem

import numpy as np

from Widgets.datapanel import DataPanel
from Widgets.svd_widget import SVDWidget


class PlotWidget(DockArea):
    instance = None

    n_spectra = 20

    # dark = 150
    # sym_grad = {'ticks': [(0.0, (75, 0, 130, 255)), (1.0, (dark, 0, 0, 255)), (0.333, (0, 0, 255, 255)),
    #                       (0.5, (255, 255, 255, 100)), (0.666, (255, 255, 0, 255)), (0.833, (255, 0, 0, 255))],
    #             'mode': 'rgb'}

    def __init__(self, set_coordinate_func=None, parent=None):
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOptions(antialias=True)

        super(PlotWidget, self).__init__(parent)

        PlotWidget.instance = self

        self.set_coordinate_func = set_coordinate_func

        self.change_range_lock = False

        self.smooth_count = 0

        self.fit_matrix = None

        self.matrix = None
        self.matrix_min = None
        self.matrix_max = None
        self.trace_plot_item = None
        self.spectrum_plot_item = None

        self.trace_plot_item_fit = None
        self.spectrum_plot_item_fit = None

        self.heat_map_levels = None

        # self.surface_widget = SurfacePlot(self)

        #  heat map

        self.heat_map_dock = Dock("Heat Map", size=(50, 7))
        self.heat_map_plot = HeatMapPlot(title="Heat Map")
        self.heat_map_plot.range_changed.connect(self.heat_map_range_changed)
        # # updates the spectra when y range of heatmap was changed
        # self.heat_map_plot.Y_range_changed.connect(self.update_spectra)
        self.heat_map_plot.levels_changed.connect(self.heat_map_levels_changed)
        heatmap_w = pg.GraphicsLayoutWidget()
        heatmap_w.ci.addItem(self.heat_map_plot)

        self.heat_map_dock.addWidget(heatmap_w)

        # self.ci.addItem(self.heat_map_plot)

        self.heat_map_vline = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen((0, 0, 0)))
        self.heat_map_hline = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen((0, 0, 0)))

        self.heat_map_plot.heat_map_plot.addItem(self.heat_map_vline, ignoreBounds=True)
        self.heat_map_plot.heat_map_plot.addItem(self.heat_map_hline, ignoreBounds=True)

        # self.spectra_plot = self.ci.addPlot(title="Spectra")

        # Spectra plot

        w_spectra = pg.PlotWidget(title="Spectra")
        self.spectra_plot = w_spectra.plotItem

        self.spectra_vline = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen((0, 0, 0)))
        self.spectra_plot.addItem(self.spectra_vline, ignoreBounds=True)

        self.spectra_plot.showAxis('top', show=True)
        self.spectra_plot.showAxis('right', show=True)

        self.spectra_plot.setLabel('left', text='\u0394A')
        self.spectra_plot.setLabel('bottom', text='Wavelength (nm)')
        self.spectra_plot.showGrid(x=True, y=True, alpha=0.1)

        self.spectra_dock = Dock("Spectra", widget=w_spectra, size=(40, 7))

        # Spectrum plot

        w_spectrum = pg.PlotWidget(title="Spectrum")
        self.spectrum = w_spectrum.plotItem

        self.spectrum_vline = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('b'))
        self.spectrum.addItem(self.spectrum_vline, ignoreBounds=True)

        self.spectrum.showAxis('top', show=True)
        self.spectrum.showAxis('right', show=True)

        self.spectrum.setLabel('left', text='\u0394A')
        self.spectrum.setLabel('bottom', text='Wavelength (nm)')

        self.spectrum_dock = Dock("Spectrum", widget=w_spectrum)
        self.spectrum.getViewBox().sigRangeChanged.connect(self.trace_spectrum_range_changed)

        # Trace plot

        w_trace = pg.PlotWidget(title="Trace")
        self.trace = w_trace.plotItem

        # self.trace = self.ci.addPlot(title="Trace")

        self.trace_vline = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('b'))
        self.trace.addItem(self.trace_vline, ignoreBounds=True)

        self.trace.showAxis('top', show=True)
        self.trace.showAxis('right', show=True)

        self.trace.setLabel('left', text='\u0394A')
        self.trace.setLabel('bottom', text='Time (us)')

        self.spectrum.showGrid(x=True, y=True, alpha=0.1)
        self.trace.showGrid(x=True, y=True, alpha=0.1)

        self.trace_dock = Dock("Trace", widget=w_trace)
        self.trace.getViewBox().sigRangeChanged.connect(self.trace_spectrum_range_changed)


        # data panel

        self.data_panel = DataPanel()
        self.settings_dock = Dock("Properties", widget=self.data_panel, size=(1, 1))

        self.data_panel.txb_t0.focus_lost.connect(self.update_range)
        self.data_panel.txb_t0.returnPressed.connect(self.update_range)
        self.data_panel.txb_t1.focus_lost.connect(self.update_range)
        self.data_panel.txb_t1.returnPressed.connect(self.update_range)
        self.data_panel.txb_w0.focus_lost.connect(self.update_range)
        self.data_panel.txb_w0.returnPressed.connect(self.update_range)
        self.data_panel.txb_w1.focus_lost.connect(self.update_range)
        self.data_panel.txb_w1.returnPressed.connect(self.update_range)
        self.data_panel.txb_z0.focus_lost.connect(self.update_levels)
        self.data_panel.txb_z0.returnPressed.connect(self.update_levels)
        self.data_panel.txb_z1.focus_lost.connect(self.update_levels)
        self.data_panel.txb_z1.returnPressed.connect(self.update_levels)

        self.data_panel.txb_n_spectra.setText(str(self.n_spectra))

        self.data_panel.btn_crop_matrix.clicked.connect(self.btn_crop_matrix_clicked)
        self.data_panel.btn_restore_matrix.clicked.connect(self.btn_restore_matrix_clicked)
        self.data_panel.txb_n_spectra.focus_lost.connect(self.txb_n_spectra_focus_lost)
        self.data_panel.txb_n_spectra.returnPressed.connect(self.txb_n_spectra_focus_lost)
        self.data_panel.btn_redraw_spectra.clicked.connect(self.update_spectra)

        self.data_panel.txb_SVD_filter.focus_lost.connect(self.txb_SVD_filter_changed)
        self.data_panel.txb_SVD_filter.returnPressed.connect(self.txb_SVD_filter_changed)
        self.data_panel.cb_SVD_filter.toggled.connect(self.cb_SVD_filter_toggled)

        self.data_panel.btn_center_levels.clicked.connect(self.btn_center_levels_clicked)
        self.data_panel.txb_SVD_filter.setText("1-5")

        # addition of docs

        self.addDock(self.heat_map_dock, 'left')
        self.addDock(self.spectra_dock, 'right')
        self.addDock(self.spectrum_dock, 'bottom')
        self.addDock(self.trace_dock, 'right', self.spectrum_dock)
        self.addDock(self.settings_dock, 'left', self.heat_map_dock)

        def update_v_lines():
            time_pos = self.heat_map_hline.pos()
            wl_pos = self.heat_map_vline.pos()

            self.spectrum_vline.setPos(wl_pos[0])
            self.spectra_vline.setPos(wl_pos[0])
            self.trace_vline.setPos(time_pos[1])

        def update_heat_lines():
            time_pos = self.trace_vline.pos()
            wl_pos = self.spectrum_vline.pos()

            self.heat_map_hline.setPos(time_pos[0])
            self.heat_map_vline.setPos(wl_pos[0])

        def update_heat_lines_spectra():
            wl_pos = self.spectra_vline.pos()
            self.heat_map_vline.setPos(wl_pos[0])

        self.heat_map_vline.sigPositionChanged.connect(update_v_lines)
        self.heat_map_hline.sigPositionChanged.connect(update_v_lines)
        self.spectrum_vline.sigPositionChanged.connect(update_heat_lines)
        self.spectra_vline.sigPositionChanged.connect(update_heat_lines_spectra)
        self.trace_vline.sigPositionChanged.connect(update_heat_lines)

        self.heat_map_vline.sigPositionChanged.connect(self.update_trace_and_spectrum)
        self.heat_map_hline.sigPositionChanged.connect(self.update_trace_and_spectrum)
        self.spectrum_vline.sigPositionChanged.connect(self.update_trace_and_spectrum)
        self.spectra_vline.sigPositionChanged.connect(self.update_trace_and_spectrum)
        self.trace_vline.sigPositionChanged.connect(self.update_trace_and_spectrum)

        self.roi = None
        self.chirp = self.heat_map_plot.heat_map_plot.plot([])

        # self.heat_map_vline.sigPositionChangeFinished.connect(self.update_trace_and_spectrum)
        # self.heat_map_hline.sigPositionChangeFinished.connect(self.update_trace_and_spectrum)
        # self.spectrum_vline.sigPositionChangeFinished.connect(self.update_trace_and_spectrum)
        # self.trace_vline.sigPositionChangeFinished.connect(self.update_trace_and_spectrum)

    def get_roi_pos(self):
        """This shit took me half a day to figure out."""
        if self.roi is None:
            return

        hs = self.roi.getHandles()
        n = len(hs)

        positions = np.zeros((n, 2))
        for i, h in enumerate(self.roi.getHandles()):
            qPoint = self.roi.mapSceneToParent(h.scenePos())

            positions[i, 0] = qPoint.x()
            positions[i, 1] = qPoint.y()

        return positions

    def plot_chirp_points(self):
        if self.roi is None:
            self.roi = pg.PolyLineROI([[350, 1], [500, 1], [650, 1]], closed=False,
                           handlePen=pg.mkPen(color=(0, 255, 0), width=5),
                           hoverPen=pg.mkPen(color=(0, 150, 0), width=2),
                           handleHoverPen=pg.mkPen(color=(0, 150, 0), width=3))

            self.heat_map_plot.heat_map_plot.addItem(self.roi)

    def add_chirp(self, wls,  mu):
        pen = pg.mkPen(color=QColor('black'), width=2)
        self.chirp.setData(wls, mu, pen=pen)

    @staticmethod
    def is_int(num):
        try:
            int(num)
            return True
        except ValueError:
            return False

    def cb_SVD_filter_toggled(self):
        self.matrix.SVD_filter = self.data_panel.cb_SVD_filter.isChecked()
        # self.plot_matrix(self.matrix, center_lines=False, keep_range=True)
        if self.data_panel.cb_SVD_filter.isChecked():
            self.txb_SVD_filter_changed()
        else:
            self.plot_matrix(self.matrix, center_lines=False, keep_range=True, keep_fits=True)

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
                if self.is_int(val):
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
            self.plot_matrix(self.matrix, center_lines=False, keep_range=True, keep_fits=True)

    def txb_n_spectra_focus_lost(self):
        try:
            n = int(self.data_panel.txb_n_spectra.text())
            self.n_spectra = max(2, n)

            self.data_panel.txb_n_spectra.setText(str(self.n_spectra))

            self.update_spectra()

        except ValueError:
            pass

    def trace_spectrum_range_changed(self, vb, range):
        if self.change_range_lock:
            return
        self.change_range_lock = True
        if vb == self.spectrum.getViewBox():
            self.heat_map_plot.heat_map_plot.getViewBox().setXRange(range[0][0], range[0][1], padding=0)
        else:
            self.heat_map_plot.heat_map_plot.getViewBox().setYRange(range[0][0], range[0][1], padding=0)

        self.change_range_lock = False

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

    def heat_map_range_changed(self, vb, range):
        if self.change_range_lock:
            return
        self.change_range_lock = True
        self.data_panel.txb_w0.setText(f'{range[0][0]:.4g}')
        self.data_panel.txb_w1.setText(f'{range[0][1]:.4g}')
        self.data_panel.txb_t0.setText(f'{range[1][0]:.4g}')
        self.data_panel.txb_t1.setText(f'{range[1][1]:.4g}')

        self.spectrum.getViewBox().setXRange(range[0][0], range[0][1], padding=0)
        self.trace.getViewBox().setXRange(range[1][0], range[1][1], padding=0)

        # keep all the v and h lines inside the visible area

        v_pos = self.heat_map_vline.pos()[0]
        h_pos = self.heat_map_hline.pos()[1]

        if not range[0][0] <= v_pos <= range[0][1]:
            self.heat_map_vline.setPos(range[0][0] if np.abs(v_pos - range[0][0]) < np.abs(v_pos - range[0][1]) else range[0][1])

        if not range[1][0] <= h_pos <= range[1][1]:
            self.heat_map_hline.setPos(range[1][0] if np.abs(h_pos - range[1][0]) < np.abs(h_pos - range[1][1]) else range[1][1])

        self.change_range_lock = False
    def heat_map_levels_changed(self, hist):
        z_levels = self.heat_map_plot.get_z_range()
        self.data_panel.txb_z0.setText(f'{z_levels[0]:.4g}')
        self.data_panel.txb_z1.setText(f'{z_levels[1]:.4g}')

    def update_range(self):
        try:
            t0, t1 = float(self.data_panel.txb_t0.text()), float(self.data_panel.txb_t1.text())
            w0, w1 = float(self.data_panel.txb_w0.text()), float(self.data_panel.txb_w1.text())

            if t0 <= t1 and w0 <= w1:
                # find true values that correspond to data
                _t0, _t1 = find_nearest(self.matrix.times, t0), find_nearest(self.matrix.times, t1)
                _w0, _w1 = find_nearest(self.matrix.wavelengths, w0), find_nearest(
                    self.matrix.wavelengths, w1)

                self.heat_map_plot.set_xy_range(_w0, _w1, _t0, _t1, 0)

        except ValueError:
            pass
        except AttributeError:
            pass

    def update_levels(self):
        try:
            z0, z1 = float(self.data_panel.txb_z0.text()), float(self.data_panel.txb_z1.text())

            if z0 <= z1:
                self.heat_map_plot.hist.setLevels(z0, z1)

        except ValueError:
            pass

    def btn_center_levels_clicked(self):
        z0, z1 = self.heat_map_plot.get_z_range()
        diff = z1 - z0
        self.heat_map_plot.hist.setLevels(-diff / 2, diff / 2)

    def btn_crop_matrix_clicked(self):
        if self.matrix is None:
            return

        w0, w1, t0, t1 = self.get_selected_range()

        self.matrix.crop_data(t0, t1, w0, w1)
        SVDWidget.instance.set_data(self.matrix)

        self.cb_SVD_filter_toggled()

        # self.plot_matrix(self.matrix, False)

    def btn_restore_matrix_clicked(self):
        if self.matrix is None:
            return
        self.matrix.restore_original_data()
        self.plot_matrix(self.matrix, False)

    def init_trace_and_spectrum(self):

        self.trace_plot_item = self.trace.plot([])
        self.spectrum_plot_item = self.spectrum.plot([])

    def init_fit_trace_sp(self):
        self.trace_plot_item_fit = self.trace.plot([])
        self.spectrum_plot_item_fit = self.spectrum.plot([])

    def update_trace_and_spectrum(self):
        if self.matrix is None:
            return

        time_pos = self.heat_map_hline.pos()[1]
        wl_pos = self.heat_map_vline.pos()[0]

        wavelengths = self.matrix.wavelengths
        times = self.matrix.times

        t_idx = find_nearest_idx(times, time_pos)
        wl_idx = find_nearest_idx(wavelengths, wl_pos)

        trace_y_data = self.matrix.D[:, wl_idx]

        pen = pg.mkPen(color=QColor('black'), width=1)

        if self.smooth_count == 0:
            spectrum_y_data = self.matrix.D[t_idx, :]
        else:
            avrg_slice_matrix = self.matrix.D[t_idx - self.smooth_count:t_idx + self.smooth_count, :]
            spectrum_y_data = np.average(avrg_slice_matrix, axis=0)

        if self.trace_plot_item is not None:
            self.trace_plot_item.setData(times, trace_y_data, pen=pen)
            self.spectrum_plot_item.setData(wavelengths, spectrum_y_data, pen=pen)

        if self.spectrum_plot_item_fit is not None and self.trace_plot_item_fit is not None and self.fit_matrix is not None:
            trace_y_data_fit = self.fit_matrix.Y[:, wl_idx]

            if self.smooth_count == 0:
                spectrum_y_data_fit = self.fit_matrix.Y[t_idx, :]
            else:
                avrg_slice_matrix_fit = self.fit_matrix.Y[t_idx - self.smooth_count:t_idx + self.smooth_count, :]
                spectrum_y_data_fit = np.average(avrg_slice_matrix_fit, axis=0)

            pen_fit = pg.mkPen(color=QColor('red'), width=1)
            self.trace_plot_item_fit.setData(times, trace_y_data_fit, pen=pen_fit)
            self.spectrum_plot_item_fit.setData(wavelengths, spectrum_y_data_fit, pen=pen_fit)

        if self.set_coordinate_func is not None:
            self.set_coordinate_func('w = {:.3g}, t = {:.3g}'.format(wavelengths[wl_idx], times[t_idx]))

        # self.spectrum.setTitle("Spectrum, t = {:.3g} us".format(time_pos))
        # self.trace.setTitle("Trace, \u03bb = {:.3g} nm".format(wl_pos))

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

        w0, w1, t0, t1 = self.get_selected_range()

        D_crop, times, wavelengths = crop_data(self.matrix.Y, self.matrix.times, self.matrix.wavelengths, t0,
                                                         t1, w0, w1)

        for i in range(self.n_spectra):
            sp = D_crop[int(i * D_crop.shape[0] / self.n_spectra)]
            color = pg.intColor(i, hues=self.n_spectra, values=1, maxHue=360, minHue=0)
            pen = pg.mkPen(color=color, width=1)
            self.spectra_plot.plot(wavelengths, sp, pen=pen)

    def plot_matrix(self, matrix, center_lines=True, keep_range=False, keep_fits=False):

        w_range, t_range = self.heat_map_plot.heat_map_plot.getViewBox().viewRange()
        z_range = self.heat_map_plot.get_z_range()

        self.spectrum.clearPlots()
        self.trace.clearPlots()
        # self.spectra_plot.clearPlots()

        self.matrix = matrix

        self.matrix_min = np.min(matrix.D)
        self.matrix_max = np.max(matrix.D)

        if self.matrix.original_data_matrix is not None:
            self.data_panel.lbl_matrix_size.setText(f'{matrix.original_data_matrix.shape[0] - 1} x {matrix.original_data_matrix.shape[1] - 1}')
        self.data_panel.lbl_cr_matrix_size.setText(f'{matrix.D.shape[0]} x {matrix.D.shape[1]}')

        self.heat_map_plot.set_matrix(matrix.D, matrix.times, matrix.wavelengths, gradient=HeatMapPlot.sym_grad,
                                      t_range=t_range if keep_range else None,
                                      w_range=w_range if keep_range else None,
                                      z_range=z_range if keep_range else None)

        self.spectrum.getViewBox().setLimits(xMin=matrix.wavelengths[0], xMax=matrix.wavelengths[-1],
                                             yMin=self.matrix_min, yMax=self.matrix_max)

        self.spectra_plot.getViewBox().setLimits(xMin=matrix.wavelengths[0], xMax=matrix.wavelengths[-1],
                                             yMin=self.matrix_min, yMax=self.matrix_max)

        self.trace.getViewBox().setLimits(xMin=matrix.times[0], xMax=matrix.times[-1],
                                          yMin=self.matrix_min, yMax=self.matrix_max)

        self.spectrum_vline.setBounds([matrix.wavelengths[0], matrix.wavelengths[-1]])
        self.spectra_vline.setBounds([matrix.wavelengths[0], matrix.wavelengths[-1]])
        self.trace_vline.setBounds([matrix.times[0], matrix.times[-1]])
        self.heat_map_vline.setBounds([matrix.wavelengths[0], matrix.wavelengths[-1]])
        self.heat_map_hline.setBounds([matrix.times[0], matrix.times[-1]])

        # autoscale heatmap
        # self.heat_map_plot.autoBtnClicked()

        # setupline in the middle of matrix
        if center_lines:
            self.heat_map_vline.setPos((matrix.wavelengths[-1] + matrix.wavelengths[0]) / 2)
            self.heat_map_hline.setPos((matrix.times[-1] + matrix.times[0]) / 2)

        # update their positions
        self.heat_map_hline.sigPositionChanged.emit(object)

        # self.hist.gradient.restoreState(self.sym_grad)

        self.init_trace_and_spectrum()

        if keep_fits:
            self.init_fit_trace_sp()

        # redraw trace and spectrum figures
        self.update_trace_and_spectrum()

        self.update_spectra()

        self.plot_chirp_points()

    def save_plot_to_clipboard_as_png(self, plot_item):
        self.img_exporter = ImageExporter(plot_item)
        self.img_exporter.export(copy=True)

    def save_plot_to_clipboard_as_svg(self, plot_item):
        self.svg_exporter = SVGExporter(plot_item)
        self.svg_exporter.export(copy=True)






# class SurfacePlot(object):
#
#     def __init__(self, parent=None):
#         self.traces = dict()
#         self.w = gl.GLViewWidget()
#         self.w.opts['distance'] = 40
#         self.w.setWindowTitle('pyqtgraph example: GLLinePlotItem')
#         self.w.setGeometry(0, 110, 1920, 1080)
#         self.w.show()
#
#         self.phase = 0
#         self.lines = 50
#         self.points = 1000
#         self.y = np.linspace(-10, 10, self.lines)
#         self.x = np.linspace(-10, 10, self.points)
#
#         for i, line in enumerate(self.y):
#             y = np.array([line] * self.points)
#             d = np.sqrt(self.x ** 2 + y ** 2)
#             sine = 10 * np.sin(d + self.phase)
#             pts = np.vstack([self.x, y, sine]).transpose()
#             self.traces[i] = gl.GLLinePlotItem(
#                 pos=pts,
#                 color=pg.glColor((i, self.lines * 1.3)),
#                 width=(i + 1) / 10,
#                 antialias=True
#             )
#             self.w.addItem(self.traces[i])
#
#     def start(self):
#         if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#             QtGui.QApplication.instance().exec_()
#
#     def set_plotdata(self, name, points, color, width):
#         self.traces[name].setData(pos=points, color=color, width=width)
#
#     def _update(self):
#         stime = time.time()
#         for i, line in enumerate(self.y):
#             y = np.array([line] * self.points)
#
#             amp = 10 / (i + 1)
#             phase = self.phase * (i + 1) - 10
#             freq = self.x * (i + 1) / 10
#
#             sine = amp * np.sin(freq - phase)
#             pts = np.vstack([self.x, y, sine]).transpose()
#
#             self.set_plotdata(
#                 name=i, points=pts,
#                 color=pg.glColor((i, self.lines * 1.3)),
#                 width=3
#             )
#             self.phase -= .00002
#
#         print('{:.0f} FPS'.format(1 / (time.time() - stime)))
#
#     def animation(self):
#         timer = QtCore.QTimer()
#         timer.timeout.connect(self.update)
#         timer.start(10)
#         self.start()
