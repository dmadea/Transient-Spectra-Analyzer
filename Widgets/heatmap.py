import pyqtgraph as pg
import numpy as np
from scipy.interpolate import interp2d
from pyqtgraphmodif.StringAxis import StringAxis
from misc import find_nearest_idx


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

    def __init__(self, parent=None, border=None, title="Residuals (CS<sup>T</sup> - D)", xlabel='\u2190 Time (ps)',
                 ylabel='Wavelength (nm) \u2192'):
        super(HeatMapPlot, self).__init__(parent, border)

        # self.heat_map_levels = None
        self.times = None

        self.stringaxis_left = StringAxis(None, orientation='left')
        self.stringaxis_right = StringAxis(None, orientation='right')

        self.heat_map_plot = self.addPlot(title=title, axisItems={'left': self.stringaxis_left,
                                                                  'right': self.stringaxis_right})
        self.heat_map_plot.getViewBox().invertY(True)

        # signals
        self.range_changed = self.heat_map_plot.getViewBox().sigRangeChanged
        self.Y_range_changed = self.heat_map_plot.getViewBox().sigYRangeChanged

        self.heat_map = Heatmap()
        self.heat_map_plot.addItem(self.heat_map)

        self.heat_map_plot.showAxis('top', show=True)
        self.heat_map_plot.showAxis('right', show=True)

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

        # need to interpolate if spacing is not equal in the time values to show correctly --- old stuff, not necessary
        # if not all(np.isclose(t_diff, t_diff[0], atol=0)):
        #     min_diff = t_diff.min()
        #
        #     t_points = (times[-1] - times[0]) / min_diff + 1
        #     new_t = np.linspace(times[0], times[-1], int(np.ceil(t_points)))
        #
        #     func = interp2d(wavelengths, times, matrix, kind='linear', copy=False)
        #     # replace the matrix by interpolated one
        #     matrix = func(wavelengths, new_t)

        self.heat_map.set_heat_map(matrix.T, wavelengths[0], times[0], wavelengths[-1], times[-1])

        # autoscale heatmap
        # self.heat_map_plot.autoBtnClicked()

        self.hist.gradient.restoreState(self.sym_grad if gradient is None else gradient)
        self.heat_map_plot.getViewBox().setLimits(xMin=wavelengths[0], xMax=wavelengths[-1],
                                                  yMin=times[0], yMax=times[-1])

        self.set_xy_range(w_range[0] if w_range else wavelengths[0], w_range[1] if w_range else wavelengths[-1],
                          t_range[0] if t_range else times[0], t_range[1] if t_range else times[-1])

        t_diff = np.abs(times[1:] - times[:-1])
        if not np.allclose(t_diff, t_diff[0], atol=0):  # if loaded data are not spaced linearly,
            self.times = times  # set times
            self.stringaxis_left.transform = self.transform_t_pos
            self.stringaxis_right.transform = self.transform_t_pos

    def transform_t_pos(self, time):
        """works for scalar and arrays"""
        if self.times is None:
            return time

        t_min, t_max = self.times.min(), self.times.max()
        idx = (time - t_min) / (t_max - t_min) * (self.times.shape[0] - 1)  # map the linear scale to data time scale
        idx = np.round(idx, 0).astype(int)
        if isinstance(idx, np.ndarray):
            idx[idx >= self.times.shape[0]] = self.times.shape[0] - 1
        else:
            idx = self.times.shape[0] - 1 if idx >= self.times.shape[0] else idx
        return self.times[idx]

    def inv_transform_t_pos(self, time):
        """works for scalar and arrays"""
        if self.times is None:
            return time

        t_min, t_max = self.times.min(), self.times.max()
        t_idx = find_nearest_idx(self.times, time)
        inv_t = t_min + t_idx * (t_max - t_min) / (self.times.shape[0] - 1)  # inverse transform
        return inv_t

    def set_xy_range(self, x0, x1, y0, y1, padding=0):
        y0, y1 = self.inv_transform_t_pos(y0), self.inv_transform_t_pos(y1)
        self.heat_map_plot.getViewBox().setRange(xRange=[x0, x1], yRange=[y0, y1], padding=padding)


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



