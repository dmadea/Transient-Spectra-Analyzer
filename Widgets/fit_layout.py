import pyqtgraph as pg

from PyQt5.QtGui import *

from PyQt5.QtCore import *

from PyQt5.QtCore import Qt

# from pyqtgraph.graphicsItems.LegendItem import LegendItem

from pyqtgraphmodif.legend_item import LegendItem

import numpy as np

from misc import find_nearest_idx

class FitLayout(pg.GraphicsLayoutWidget):
    instance = None

    def __init__(self, set_coordinate_func=None, parent=None):
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOptions(antialias=True)

        super(FitLayout, self).__init__(parent)

        FitLayout.instance = self

        self.set_coordinate_func = set_coordinate_func

        # self.fit_matrix = None
        #
        # self.c_matrix = None
        # self.st_matrix = None
        # self.res_matrix = None
        # #
        # self.matrix_max = None
        # self.trace_plot_item = None
        # self.spectrum_plot_item = None
        #
        # self.trace_plot_item_fit = None
        # self.spectrum_plot_item_fit = None

        self.ST_plot = self.ci.addPlot(title="S<sup>T</sup> Matrix")
        #
        # self.spectrum_vline = pg.InfiniteLine(angle=90, movable=True)
        # self.spectrum.addItem(self.spectrum_vline, ignoreBounds=True)

        self.ST_plot.showAxis('top', show=True)
        self.ST_plot.showAxis('right', show=True)

        self.ST_plot.setLabel('left', text='\u0394A')
        self.ST_plot.setLabel('bottom', text='Wavelength (nm)')

        # self.trace = self.ci.addPlot(title="Trace")
        self.C_plot = self.ci.addPlot(title="C Matrix")
        #
        # self.trace_vline = pg.InfiniteLine(angle=90, movable=True)
        # self.trace.addItem(self.trace_vline, ignoreBounds=True)

        self.C_plot.showAxis('top', show=True)
        self.C_plot.showAxis('right', show=True)

        self.C_plot.setLabel('left', text='\u0394A')
        self.C_plot.setLabel('bottom', text='Time (us)')

        self.ST_plot.showGrid(x=True, y=True, alpha=0.1)
        self.C_plot.showGrid(x=True, y=True, alpha=0.1)

        self.heat_map_plot = HeatMapPlot()
        self.ci.addItem(self.heat_map_plot)

        self.C_legend = None
        self.ST_legend = None

        self.add_legend()

        #
        # def update_v_lines():
        #     time_pos = self.heat_map_vline.pos()
        #     wl_pos = self.heat_map_hline.pos()
        #
        #     self.spectrum_vline.setPos(wl_pos[1])
        #     self.trace_vline.setPos(time_pos[0])
        #
        # def update_heat_lines():
        #     time_pos = self.trace_vline.pos()
        #     wl_pos = self.spectrum_vline.pos()
        #
        #     self.heat_map_vline.setPos(time_pos[0])
        #     self.heat_map_hline.setPos(wl_pos[0])

        # self.heat_map_vline.sigPositionChanged.connect(update_v_lines)
        # self.heat_map_hline.sigPositionChanged.connect(update_v_lines)
        # self.spectrum_vline.sigPositionChanged.connect(update_heat_lines)
        # self.trace_vline.sigPositionChanged.connect(update_heat_lines)
        #
        # self.heat_map_vline.sigPositionChanged.connect(self.update_trace_and_spectrum)
        # self.heat_map_hline.sigPositionChanged.connect(self.update_trace_and_spectrum)
        # self.spectrum_vline.sigPositionChanged.connect(self.update_trace_and_spectrum)
        # self.trace_vline.sigPositionChanged.connect(self.update_trace_and_spectrum)

        #
        # self.heat_map_vline.sigPositionChangeFinished.connect(self.update_trace_and_spectrum)
        # self.heat_map_hline.sigPositionChangeFinished.connect(self.update_trace_and_spectrum)
        # self.spectrum_vline.sigPositionChangeFinished.connect(self.update_trace_and_spectrum)
        # self.trace_vline.sigPositionChangeFinished.connect(self.update_trace_and_spectrum)

        #
        # self.update_settings()

        # self.legend = None
        # self.add_legend()

        # proxy = pg.SignalProxy(self.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_moved)

    def add_legend(self, size=None, spacing=5, offset=(-30, 30)):

        try:
            # self.grpView.plotItem.legend.scene().removeItem(self.grpView.plotItem.legend)
            self.C_legend.scene().removeItem(self.C_legend)
            self.ST_legend.scene().removeItem(self.ST_legend)
        except:
            pass

        self.C_legend = LegendItem(size, spacing, offset)
        self.C_legend.setParentItem(self.C_plot.vb)
        self.C_plot.legend = self.C_legend

        self.ST_legend = LegendItem(size, spacing, offset)
        self.ST_legend.setParentItem(self.ST_plot.vb)
        self.ST_plot.legend = self.ST_legend

        # return self.legend

    @staticmethod
    def int_default_color(counter):
        colors = [
            (255, 0, 0, 255),  # red
            (0, 255, 0, 255),  # green
            (0, 0, 255, 255),  # blue
            (0, 0, 0, 255),  # black
            (255, 255, 0, 255),  # yellow
            (255, 0, 255, 255),  # magenta
            (0, 255, 255, 255),  # cyan
            (155, 155, 155, 255),  # gray
            (155, 0, 0, 255),  # dark red
            (0, 155, 0, 255),  # dark green
            (0, 0, 155, 255),  # dark blue
            (155, 155, 0, 255),  # dark yellow
            (155, 0, 155, 255),  # dark magenta
            (0, 155, 155, 255)  # dark cyan
        ]

        return QColor(*colors[counter % len(colors)])

    # def init_trace_and_spectrum(self):
    #
    #     self.trace_plot_item = self.C_plot.plot([1, 2], [1, 2])
    #     self.spectrum_plot_item = self.ST_plot.plot([1, 2], [1, 2])
    #
    # def init_fit_trace_sp(self):
    #
    #     self.trace_plot_item_fit = self.C_plot.plot([1, 2, 3], [1, 2, 3])
    #     self.spectrum_plot_item_fit = self.ST_plot.plot([1, 2, 3], [1, 2, 3])

    # def update_trace_and_spectrum(self):
    #     if self.matrix is None:
    #         return
    #
    #     time_pos = self.heat_map_vline.pos()[0]
    #     wl_pos = self.heat_map_hline.pos()[1]
    #
    #     wavelengths = self.matrix.wavelengths
    #     times = self.matrix.times
    #
    #     t_idx = Spectrum.find_nearest_idx(times, time_pos)
    #     wl_idx = Spectrum.find_nearest_idx(wavelengths, wl_pos)
    #
    #     trace_y_data = self.matrix.Y[:, wl_idx]
    #
    #     pen = pg.mkPen(color=QColor('black'), width=1)
    #
    #     if self.smooth_count == 0:
    #         spectrum_y_data = self.matrix.Y[t_idx, :]
    #     else:
    #         avrg_slice_matrix = self.matrix.Y[t_idx - self.smooth_count:t_idx + self.smooth_count, :]
    #         spectrum_y_data = np.average(avrg_slice_matrix, axis=0)
    #
    #     if self.trace_plot_item is not None:
    #         self.trace_plot_item.setData(times, trace_y_data, pen=pen)
    #         self.spectrum_plot_item.setData(wavelengths, spectrum_y_data, pen=pen)
    #
    #     if self.spectrum_plot_item_fit is not None and self.trace_plot_item_fit is not None:
    #         trace_y_data_fit = self.fit_matrix.Y[:, wl_idx]
    #
    #         if self.smooth_count == 0:
    #             spectrum_y_data_fit = self.fit_matrix.Y[t_idx, :]
    #         else:
    #             avrg_slice_matrix_fit = self.fit_matrix.Y[t_idx - self.smooth_count:t_idx + self.smooth_count, :]
    #             spectrum_y_data_fit = np.average(avrg_slice_matrix_fit, axis=0)
    #
    #         pen_fit = pg.mkPen(color=QColor('red'), width=1)
    #         self.trace_plot_item_fit.setData(times, trace_y_data_fit, pen=pen_fit)
    #         self.spectrum_plot_item_fit.setData(wavelengths, spectrum_y_data_fit, pen=pen_fit)
    #
    #     if self.set_coordinate_func is not None:
    #         self.set_coordinate_func('w = {:.3g}, t = {:.3g}'.format(wavelengths[wl_idx], times[t_idx]))
    #
    #     # self.spectrum.setTitle("Spectrum, t = {:.3g} us".format(time_pos))
    #     # self.trace.setTitle("Trace, \u03bb = {:.3g} nm".format(wl_pos))
    #
    # def set_fit_matrix(self, fit_matrix):
    #     self.fit_matrix = fit_matrix
    #
    #     self.ST_plot.clearPlots()
    #     self.C_plot.clearPlots()
    #
    #     self.init_trace_and_spectrum()
    #     self.init_fit_trace_sp()
    #     self.update_trace_and_spectrum()
    #
    #     self.ST_plot.autoBtnClicked()
    #     self.C_plot.autoBtnClicked()

    # def plot_matrix(self, matrix, center_lines=True):
    #
    #     self.ST_plot.clearPlots()
    #     self.C_plot.clearPlots()
    #
    #     self.spectrum_plot_item_fit = None
    #     self.trace_plot_item_fit = None
    #
    #     self.matrix = matrix
    #
    #     self.matrix_min = np.min(matrix.Y)
    #     self.matrix_max = np.max(matrix.Y)
    #
    #     if self.heat_map_levels is None:
    #         self.hist.setLevels(self.matrix_min, self.matrix_max)
    #         self.heat_map_levels = (self.matrix_min, self.matrix_max)
    #     self.hist.setHistogramRange(self.matrix_min, self.matrix_max)
    #
    #     self.heat_map.set_heat_map(matrix, matrix.times[0], matrix.wavelengths[0],
    #                                matrix.times[-1], matrix.wavelengths[-1])
    #
    #     # autoscale heatmap
    #     self.heat_map_plot.autoBtnClicked()
    #
    #     # setupline in the middle of matrix
    #     if center_lines:
    #         self.heat_map_hline.setPos((matrix.wavelengths[-1] + matrix.wavelengths[0]) / 2)
    #         self.heat_map_vline.setPos((matrix.times[-1] + matrix.times[0]) / 2)
    #
    #     # update their positions
    #     self.heat_map_hline.sigPositionChanged.emit(object)
    #
    #     self.init_trace_and_spectrum()
    #
    #     # redraw trace and spectrum figures
    #     self.update_trace_and_spectrum()


class HeatMapPlot(pg.GraphicsLayout):
    positive_grad = {'ticks': [(0.0, (255, 255, 255, 255)), (1.0, (100, 0, 0, 255)), (0.33, (255, 200, 0, 255)),
                               (0.66, (255, 0, 0, 255))], 'mode': 'rgb'}

    dark = 150
    seismic = {'ticks': [(0.0, (0, 0, dark, 255)), (1.0, (dark, 0, 0, 255)), (0.25, (0, 0, 255, 255)),
                         (0.5, (255, 255, 255, 255)), (0.75, (255, 0, 0, 255))], 'mode': 'rgb'}

    # sym_grad = {'ticks': [(0.0, (75, 0, 130, 255)), (1.0, (dark, 0, 0, 255)), (0.333, (0, 0, 255, 255)),
    #                       (0.5, (255, 255, 255, 255)), (0.666, (255, 255, 0, 255)), (0.833, (255, 0, 0, 255))],
    #             'mode': 'rgb'}

    sym_grad = {'ticks': [(0.0, (75, 0, 130, 255)), (1.0, (dark, 0, 0, 255)), (0.333, (0, 0, 255, 255)),
                          (0.5, (255, 255, 255, 255)), (0.625, (255, 255, 0, 255)), (0.75, (255, 165, 0, 255)),
                          (0.875, (255, 0, 0, 255))],
                'mode': 'rgb'}

    def __init__(self, parent=None, border=None, title="Residuals (CS<sup>T</sup> - D)", xlabel='\u2190 Time (us)',
                 ylabel='Wavelength (nm) \u2192'):
        super(HeatMapPlot, self).__init__(parent, border)

        # self.heat_map_levels = None

        self.heat_map_plot = self.addPlot(title=title)
        self.heat_map_plot.getViewBox().invertY(True)

        # signals
        self.range_changed = self.heat_map_plot.getViewBox().sigRangeChanged
        self.Y_range_changed = self.heat_map_plot.getViewBox().sigYRangeChanged

        self.heat_map = Heatmap()
        self.heat_map_plot.addItem(self.heat_map)

        self.heat_map_plot.showAxis('top', show=True)
        self.heat_map_plot.showAxis('right', show=True)

        # self.heat_map_vline = pg.InfiniteLine(angle=90, movable=True)
        # self.heat_map_hline = pg.InfiniteLine(angle=0, movable=True)

        # self.heat_map_plot.addItem(self.heat_map_vline, ignoreBounds=True)
        # self.heat_map_plot.addItem(self.heat_map_hline, ignoreBounds=True)

        self.heat_map_plot.setLabel('left', text=xlabel)
        self.heat_map_plot.setLabel('bottom', text=ylabel)

        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.heat_map)
        # self.hist.gradient.loadPreset('bipolar')
        self.hist.gradient.restoreState(self.sym_grad)
        self.hist.axis.setLabel('\u0394A')

        self.levels_changed = self.hist.sigLevelsChanged

        self.addItem(self.hist)

    def get_z_range(self):
        return self.hist.getLevels()

    def set_matrix(self, matrix, times, wavelengths, gradient=None, t_range=None, w_range=None, z_range=None):
        """

        :param matrix:
        :param times:
        :param wavelengths:
        :param gradient:
        :param t_range:  tuple '[min, max]' or None
        :param w_range:  tuple '[min, max]' or None
        :param z_range:  tuple '[min, max]' or None
        :return:
        """
        extrem = np.max(matrix) if np.abs(np.max(matrix)) > 1e-10 else np.min(matrix)

        matrix_min = -np.abs(extrem)
        matrix_max = -matrix_min

        self.hist.setLevels(z_range[0] if z_range else matrix_min,
                            z_range[1] if z_range else matrix_max)
        if z_range is None:
            self.hist.setHistogramRange(matrix_min, matrix_max)

        t_diff = np.abs(times[1:] - times[:-1])

        # need to interpolate if spacing is not equal in the time values to show correctly
        if not all(np.isclose(t_diff, t_diff[0], atol=0)):
            min_diff = t_diff.min()

            t_points = (times[-1] - times[0]) / min_diff + 1
            new_t = np.linspace(times[0], times[-1], int(np.ceil(t_points)))

            func = interp2d(wavelengths, times, matrix, kind='linear', copy=False)
            # replace the matrix by interpolated one
            matrix = func(wavelengths, new_t)

        self.heat_map.set_heat_map(matrix.T, wavelengths[0], times[0], wavelengths[-1], times[-1])

        # autoscale heatmap
        # self.heat_map_plot.autoBtnClicked()

        self.hist.gradient.restoreState(self.sym_grad if gradient is None else gradient)
        self.heat_map_plot.getViewBox().setLimits(xMin=wavelengths[0], xMax=wavelengths[-1],
                                                  yMin=times[0], yMax=times[-1])

        self.set_xy_range(w_range[0] if w_range else wavelengths[0], w_range[1] if w_range else wavelengths[-1],
                          t_range[0] if t_range else times[0], t_range[1] if t_range else times[-1])

    def set_xy_range(self, x0, x1, y0, y1, padding=0):
        self.heat_map_plot.getViewBox().setRange(xRange=[x0, x1], yRange=[y0, y1], padding=padding)


from scipy.interpolate import interp2d


class Heatmap(pg.ImageItem):

    def __init__(self, image=None):
        self.image = image
        pg.ImageItem.__init__(self, self.image)

    def set_heat_map(self, matrix, x0, y0, x_max, y_max):
        self.image = matrix

        self.resetTransform()
        self.translate(x0, y0)
        self.scale((x_max - x0) / self.width(), (y_max - y0) / self.height())

        self.render()
        self.update()
