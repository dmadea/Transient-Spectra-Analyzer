import pyqtgraph as pg
import numpy as np
# from scipy.interpolate import interp2d
from pyqtgraphmodif.StringAxis import StringAxis
from misc import find_nearest_idx
from PyQt6.QtCore import Qt

from functools import reduce
import operator

from settings import Settings

from typing import Callable


class HeatMapWidget(pg.GraphicsLayoutWidget):

    def __init__(self, parent=None, border=None):

        super(HeatMapWidget, self).__init__(parent, border)

        self.heatmaps = [Heatmap(self, title='')]
        # self.heatmaps = [Heatmap(self) for i in range(4)]
        self.addItem(self.heatmaps[0], 0, 0)
        self.heatmaps[0].connect_signals()

    def set_heatmaps(self, matrices, center_lines=True):

        # TODO delete previous heatmaps

        n = len(matrices)

        if n > 1:
            self.heatmaps += [Heatmap(self) for i in range(n - 1)]

            # rows = n // 2
            # half = list(range(rows))
            # row_idxs = reduce(operator.add, [[i] * rows for i in range(rows)])
            # cols_idxs = half * rows
            # idxs = list(zip(row_idxs, cols_idxs))

            idxs = list(zip([0] * n, list(range(n))))

            for h, (r, c) in zip(self.heatmaps[1:], idxs[1:]):
                self.addItem(h, r, c)
                h.connect_signals()

        for h, m in zip(self.heatmaps, matrices):
            h.set_matrix(m.D, m.times, m.wavelengths, center_lines=center_lines)
            h.heatmap_pi.setTitle(m.get_filename())

    def get_positions(self):
        y_positions = []
        x_positions = []

        for h in self.heatmaps:
            y_pos = h.hline.pos()[1]
            x_pos = h.vline.pos()[0]
            y_positions.append(h.transform_t_pos(y_pos))
            x_positions.append(h.transform_wl_pos(x_pos))

        return x_positions, y_positions

    def connect_v_lines_position_changed(self, fn: Callable):
        for h in self.heatmaps:
            h.vline.sigPositionChanged.connect(fn)

    def connect_h_lines_position_changed(self, fn: Callable):
        for h in self.heatmaps:
            h.hline.sigPositionChanged.connect(fn)

class Heatmap(pg.GraphicsLayout):

    dark = 150
    positive_grad = {'ticks': [(0.0, (255, 255, 255, 255)), (1.0, (100, 0, 0, 255)), (0.33, (255, 200, 0, 255)),
                               (0.66, (255, 0, 0, 255))], 'mode': 'rgb'}

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
                 ylabel='Wavelength (nm) \u2192', z_label='\u0394A'):
        super(Heatmap, self).__init__(None, border)
        self.parentWidget = parent
        self.arr_ax0 = None
        self.arr_ax1 = None

        self.image = pg.ImageItem()

        self.string_axis_left = StringAxis(None, orientation='left')
        self.string_axis_right = StringAxis(None, orientation='right')

        self.string_axis_bottom = StringAxis(None, orientation='bottom')
        self.string_axis_top = StringAxis(None, orientation='top')

        # heat map plot item
        self.heatmap_pi = self.addPlot(title=title, axisItems={'left': self.string_axis_left,
                                                                'right': self.string_axis_right,
                                                                'bottom': self.string_axis_bottom,
                                                                'top': self.string_axis_top})
        self.heatmap_pi.getViewBox().invertY(True)

        self.heatmap_pi.addItem(self.image)
        self.heatmap_pi.showAxis('top', show=True)
        self.heatmap_pi.showAxis('right', show=True)

        self.heatmap_pi.setLabel('left', text=xlabel)
        self.heatmap_pi.setLabel('bottom', text=ylabel)

        self.addItem(self.heatmap_pi, 0, 0)

        # self.probe_label = pg.LabelItem('no cursor data shown', justify='left')

        # self.plotItem.setDownsampling(ds=True, auto=True, mode='subsample')
        # self.plotItem.setClipToView(True)

        # self.addItem(self.plotItem, 0, 0)
        # self.addItem(self.probe_label, 1, 0)

        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.image)
        self.hist.gradient.restoreState(self.sym_grad)
        self.hist.axis.setLabel(z_label)

        self.addItem(self.hist, 0, 1)

        label_format = f'{{value:.{Settings.coordinates_sig_figures}g}}'

        self.vline = self.heatmap_pi.addLine(0, 0, 100, angle=90, movable=True, pen=pg.mkPen((0, 0, 0)), label=label_format,
                                             labelOpts=dict(rotateAxis=(1, 0), color=Settings.infinite_line_label_color))
        self.hline = self.heatmap_pi.addLine(0, 0, 100, angle=0, movable=True, pen=pg.mkPen((0, 0, 0)), label=label_format,
                                             labelOpts=dict(color=Settings.infinite_line_label_color))

        # signals
        self.range_changed = self.heatmap_pi.getViewBox().sigRangeChanged
        self.Y_range_changed = self.heatmap_pi.getViewBox().sigYRangeChanged
        self.levels_changed = self.hist.sigLevelsChanged

        self.proxy = None

    def connect_signals(self):
        self.proxy = pg.SignalProxy(self.heatmap_pi.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_moved)

    def get_z_range(self):
        return self.hist.getLevels()

    def set_matrix(self, matrix, arr_ax0, arr_ax1, gradient=None, t_range=None, w_range=None, z_range=None,
                   center_lines=True):
        """

        :param matrix:
        :param arr_ax0:
        :param arr_ax1:
        :param gradient:
        :param t_range:  tuple '[min, max]' or None
        :param w_range:  tuple '[min, max]' or None
        :param z_range:  tuple '[min, max]' or None
        :return:
        """
        extreme = np.max(matrix) if np.abs(np.max(matrix)) > 1e-10 else np.min(matrix)

        matrix_min = -np.abs(extreme)
        matrix_max = -matrix_min

        x0, y0, x_max, y_max = arr_ax1[0], arr_ax0[0], arr_ax1[-1], arr_ax0[-1]
        self.image.setImage(matrix.T, rect=(x0, y0, x_max - x0, y_max - y0))   # [x, y, w, h]
        # self.image.resetTransform()
        # self.image.translate(x0, y0)
        # self.image.scale((x_max - x0) / self.image.width(), (y_max - y0) / self.image.height())
        self.image.render()
        # self.image.update()

        self.hist.setLevels(z_range[0] if z_range else matrix_min,
                            z_range[1] if z_range else matrix_max)
        if z_range is None:
            self.hist.setHistogramRange(matrix_min, matrix_max)

        # need to interpolate if spacing is not equal in the time values to show correctly --- old stuff, not necessary
        # if not all(np.isclose(arr_ax0_diff, arr_ax0_diff[0], atol=0)):
        #     min_diff = arr_ax0_diff.min()
        #
        #     t_points = (times[-1] - times[0]) / min_diff + 1
        #     new_t = np.linspace(times[0], times[-1], int(np.ceil(t_points)))
        #
        #     func = interp2d(wavelengths, times, matrix, kind='linear', copy=False)
        #     # replace the matrix by interpolated one
        #     matrix = func(wavelengths, new_t)

        # autoscale heatmap
        # self.heat_map_plot.autoBtnClicked()

        self.hist.gradient.restoreState(self.sym_grad if gradient is None else gradient)
        self.heatmap_pi.getViewBox().setLimits(xMin=arr_ax1[0], xMax=arr_ax1[-1],
                                               yMin=arr_ax0[0], yMax=arr_ax0[-1])

        self.set_xy_range(w_range[0] if w_range else arr_ax1[0], w_range[1] if w_range else arr_ax1[-1],
                          t_range[0] if t_range else arr_ax0[0], t_range[1] if t_range else arr_ax0[-1])

        arr_ax0_diff = np.abs(arr_ax0[1:] - arr_ax0[:-1])
        arr_ax1_diff = np.abs(arr_ax1[1:] - arr_ax1[:-1])

        if not np.allclose(arr_ax0_diff, arr_ax0_diff[0], atol=0):  # if loaded data are not spaced linearly,
            self.arr_ax0 = arr_ax0  # set times
            self.string_axis_left.transform = self.transform_t_pos
            self.string_axis_right.transform = self.transform_t_pos

        if not np.allclose(arr_ax1_diff, arr_ax1_diff[0], atol=0):
            self.arr_ax1 = arr_ax1
            self.string_axis_bottom.transform = self.transform_wl_pos
            self.string_axis_top.transform = self.transform_wl_pos

        self.vline.setBounds((x0, x_max))
        self.hline.setBounds((y0, y_max))

        if center_lines:
            self.vline.setPos((x0 + x_max) / 2)
            self.hline.setPos((y0 + y_max) / 2)

    def transform_t_pos(self, value):
        return self.transform_value_pos(value, self.arr_ax0)

    def inv_transform_t_pos(self, value):
        return self.inv_transform_value_pos(value, self.arr_ax0)

    def transform_wl_pos(self, value):
        return self.transform_value_pos(value, self.arr_ax1)

    def inv_transform_wl_pos(self, value):
        return self.inv_transform_value_pos(value, self.arr_ax1)

    def transform_value_pos(self, value, arr=None):
        """works for scalar and arrays"""
        if arr is None:
            return value

        t_min, t_max = arr.min(), arr.max()
        idx = (value - t_min) / (t_max - t_min) * (arr.shape[0] - 1)  # map the linear scale to data time scale
        idx = np.round(idx, 0).astype(int)
        if isinstance(idx, np.ndarray):
            idx[idx >= arr.shape[0]] = arr.shape[0] - 1
        else:
            idx = arr.shape[0] - 1 if idx >= arr.shape[0] else idx
        return arr[idx]

    def inv_transform_value_pos(self, value, arr=None):
        """works for scalar and arrays"""
        if arr is None:
            return value

        t_min, t_max = arr.min(), arr.max()
        t_idx = find_nearest_idx(arr,  value)
        inv_t = t_min + t_idx * (t_max - t_min) / (arr.shape[0] - 1)  # inverse transform
        return inv_t

    def set_xy_range(self, x0, x1, y0, y1, padding=0):
        y0, y1 = self.inv_transform_t_pos(y0), self.inv_transform_t_pos(y1)
        x0, x1 = self.inv_transform_wl_pos(x0), self.inv_transform_wl_pos(x1)

        self.heatmap_pi.getViewBox().setRange(xRange=[x0, x1], yRange=[y0, y1], padding=padding)

    def mouse_moved(self, ev):
        pos = ev[0]
        in_scene = self.heatmap_pi.sceneBoundingRect().contains(pos)
        on_vline = self.vline.sceneBoundingRect().contains(pos)
        on_hline = self.hline.sceneBoundingRect().contains(pos) or \
                   self.hist.regions[0].lines[0].sceneBoundingRect().contains(pos) or \
                   self.hist.regions[0].lines[1].sceneBoundingRect().contains(pos)

        on_hist_lut = self.hist.vb.sceneBoundingRect().contains(pos)

        #     try:
        # if in_scene:
        #         mouse_point = self.plotItem.vb.mapSceneToView(pos)
        #         n = Settings.coordinates_sig_figures
        #         # double format with n being the number of significant figures of a number
        #         self.probe_label.setText(f"x={{:.{n}g}}, y={{:.{n}g}}".format(mouse_point.x(), mouse_point.y()))
        #     except:
        #         pass
        # else:
        #     self.probe_label.setText("<span style='color: #808080'>No data at cursor</span>")

        # set the corresponding cursor
        if in_scene:
            if on_vline:
                self.parentWidget.viewport().setCursor(Qt.CursorShape.SizeHorCursor)
            elif on_hline:
                self.parentWidget.viewport().setCursor(Qt.CursorShape.SizeVerCursor)
            elif on_hist_lut:
                self.parentWidget.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.parentWidget.viewport().setCursor(Qt.CursorShape.CrossCursor)



