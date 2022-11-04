import pyqtgraph as pg
import numpy as np
# from scipy.interpolate import interp2d
from pyqtgraphmodif.StringAxis import StringAxis
from misc import find_nearest_idx
from PyQt6.QtCore import Qt

from settings import Settings

from typing import Callable

from pyqtgraph.functions import mkBrush, mkColor

from pyqtgraphmodif.infinite_line_modif import InfiniteLine
from Widgets.genericlayoutwidget import GenericLayoutWidget
from Widgets.genericplotlayout import GenericPlotLayout


class HeatMapWidget(GenericLayoutWidget):

    def initialize(self):
        self.plots = [Heatmap(self, title='')]
        self.addItem(self.plots[0], 0, 0)
        self.plots[0].connect_signals()

    def get_labels(self):
        return [(h.heatmap_pi.getAxis('bottom').labelText,
                 h.heatmap_pi.getAxis('bottom').labelUnits,
                 h.heatmap_pi.getAxis('left').labelText,
                 h.heatmap_pi.getAxis('left').labelUnits,
                 h.hist.axis.labelText,
                 h.hist.axis.labelUnits) for h in self.plots]

    # def set_labels(self, index: int, x_label: str, y_label: str, z_label: str):
    #     self.plots[index].heatmap_pi.setLabel('bottom', x_label)
    #     self.plots[index].heatmap_pi.setLabel('left', y_label)
    #     self.plots[index].hist.axis.setLabel(z_label)

    def set_heatmaps(self, matrices, center_lines=True, keep_ranges=False):

        n = len(matrices)

        if keep_ranges:
            assert len(self.plots) == n

            x_ranges = []
            y_ranges = []
            z_ranges = []

            for h in self.plots:
                x_ranges.append(h.get_x_range())
                y_ranges.append(h.get_y_range())
                z_ranges.append(h.get_z_range())

        self.clear_plots()

        if n > 1:
            self.plots += [Heatmap(self) for i in range(n - 1)]


            idxs = self.get_mode_idxs(self.default_mode, n)

            for h, (r, c) in zip(self.plots[1:], idxs[1:]):
                self.addItem(h, r, c)
                h.connect_signals()

        for i, (h, m) in enumerate(zip(self.plots, matrices)):

            x_rng = x_ranges[i] if keep_ranges else None
            y_rng = y_ranges[i] if keep_ranges else None
            z_rng = y_ranges[i] if keep_ranges else None

            h.set_matrix(m.D, m.times, m.wavelengths, center_lines=center_lines, x_range=x_rng,
                         y_range=y_rng, z_range=z_rng)
            h.heatmap_pi.setTitle(m.get_filename(), size=Settings.plot_title_font_size)

    def get_positions(self):
        y_positions = []
        x_positions = []

        for h in self.plots:
            y_pos = h.get_ypos()
            x_pos = h.get_xpos()
            y_positions.append(h.transform_t_pos(y_pos))
            x_positions.append(h.transform_wl_pos(x_pos))

        return x_positions, y_positions

    def connect_v_lines_position_changed(self, fn: Callable):
        for h in self.plots:
            h.connect_v_line_position_changed(fn)

    def connect_h_lines_position_changed(self, fn: Callable):
        for h in self.plots:
            h.connect_h_line_position_changed(fn)

    def connect_ranges_changed(self, fn: Callable):
        for h in self.plots:
            h.connect_range_changed(fn)

    def clear_plots(self):
        for h in self.plots:
            h.disconnect_signals()
            h.clear()
        self.ci.clear()
        self.plots = []
        self.initialize()

    def autoscale(self):
        for h in self.plots:
            h.heatmap_pi.autoBtnClicked()

    def set_lines_color(self):
        for h in self.plots:
            h.vline.label.setColor(Settings.infinite_line_label_color)
            h.hline.label.setColor(Settings.infinite_line_label_color)

            h.vline.label.color = mkColor(Settings.infinite_line_label_color)
            h.hline.label.color = mkColor(Settings.infinite_line_label_color)


def inv_transform_value_pos(value, arr=None):
    """works for scalar and arrays"""
    if arr is None:
        return value

    t_min, t_max = arr[0], arr[-1]
    t_idx = find_nearest_idx(arr, value)
    inv_t = t_min + t_idx * (t_max - t_min) / (arr.shape[0] - 1)  # inverse transform
    return inv_t


def transform_value_pos(value, arr=None):
    """works for scalar and arrays"""
    if arr is None:
        return value

    t_min, t_max = arr[0], arr[-1]
    idx = (value - t_min) / (t_max - t_min) * (arr.shape[0] - 1)  # map the linear scale to data time scale
    idx = np.round(idx, 0).astype(int)
    if isinstance(idx, np.ndarray):
        idx[idx >= arr.shape[0]] = arr.shape[0] - 1
    else:
        idx = arr.shape[0] - 1 if idx >= arr.shape[0] else idx
    return arr[idx]


class Heatmap(GenericPlotLayout):

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

    def __init__(self, parent=None, border=None, title="Residuals", xlabel='Time',
                 ylabel='Wavelength \u2192', z_label='\u0394A', keep_levels_centered=True):
        super(Heatmap, self).__init__(None, border)
        self.keep_levels_centered = keep_levels_centered
        self.parentWidget = parent
        self.arr_ax0 = None
        self.arr_ax1 = None

        self.x_arr_nonlinear = False
        self.y_arr_nonlinear = False

        self.levels_changing = False

        self.image = pg.ImageItem()

        self.string_axis_left = StringAxis(orientation='left')
        self.string_axis_right = StringAxis(orientation='right')

        self.string_axis_bottom = StringAxis(orientation='bottom')
        self.string_axis_top = StringAxis(orientation='top')

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

        self.probe_label = pg.LabelItem("<span style='color: #808080'>No data at cursor</span>", justify='left')
        self.addItem(self.probe_label, 1, 0)

        label_format = f'{{value:.{Settings.coordinates_sig_figures}g}}'

        self.vline = InfiniteLine(self.arr_ax1, angle=90, movable=True, label=label_format,
                                  labelOpts=dict(rotateAxis=(1, 0), position=Settings.heatmap_line_label_position))
        self.vline.setZValue(10000)
        self.heatmap_pi.addItem(self.vline)

        self.hline = InfiniteLine(self.arr_ax0, angle=0, movable=True, label=label_format,
                                  labelOpts=dict(position=Settings.heatmap_line_label_position))
        self.hline.setZValue(10000)
        self.heatmap_pi.addItem(self.hline)

        self.vline.sigPositionChanged.connect(self.vline_moved)
        self.hline.sigPositionChanged.connect(self.hline_moved)

        # self.vline = self.heatmap_pi.addLine(0, 0, 10000, angle=90, movable=True, pen=pg.mkPen((0, 0, 0)), label=label_format,
        #                                      labelOpts=dict(rotateAxis=(1, 0), position=Settings.heatmap_line_label_position,
        #                                                     color=mkColor(Settings.infinite_line_label_color)))
        # self.hline = self.heatmap_pi.addLine(0, 0, 10000, angle=0, movable=True, pen=pg.mkPen((0, 0, 0)), label=label_format,
        #                                      labelOpts=dict(position=Settings.heatmap_line_label_position,
        #                                                     color=mkColor(Settings.infinite_line_label_color)))

        brush = mkBrush(color=(255, 255, 255, Settings.infinite_line_label_brush_alpha))

        self.vline.label.fill = brush
        self.hline.label.fill = brush

        self.vline.label.setColor(Settings.infinite_line_label_color)
        self.hline.label.setColor(Settings.infinite_line_label_color)

        self.layout.setSpacing(0)
        self.setContentsMargins(0, 0, 0, 0)


    def vline_moved(self, line):
        if self.arr_ax1 is None:
            return

        xpos = self.transform_wl_pos(self.get_xpos())
        self.set_xpos(inv_transform_value_pos(xpos, self.arr_ax1))

    def hline_moved(self, line):
        if self.arr_ax0 is None:
            return

        ypos = self.transform_t_pos(self.get_ypos())
        self.set_ypos(inv_transform_value_pos(ypos, self.arr_ax0))

    def connect_signals(self):
        self.heatmap_pi.scene().sigMouseMoved.connect(self.mouse_moved)
        self.hist.sigLevelsChanged.connect(self.levels_changed)
        self.heatmap_pi.scene().sigMouseClicked.connect(self.mouse_clicked)

    def disconnect_signals(self):
        self.heatmap_pi.scene().sigMouseMoved.disconnect(self.mouse_moved)
        self.heatmap_pi.scene().sigMouseClicked.disconnect(self.mouse_clicked)
        self.hist.sigLevelsChanged.disconnect(self.levels_changed)

    def levels_changed(self, lut_item):
        if self.levels_changing:
            return

        self.levels_changing = True

        if self.keep_levels_centered:
            z0, z1 = self.hist.getLevels()
            self.hist.setLevels(-np.abs(z1), z1)  # only take care of the second level

        self.levels_changing = False

    def connect_range_changed(self, fn: Callable):
        self.heatmap_pi.getViewBox().sigRangeChanged.connect(lambda *args: fn(self, *args))

    def connect_v_line_position_changed(self, fn: Callable):
        self.vline.sigPositionChanged.connect(lambda: fn(self))

    def connect_h_line_position_changed(self, fn: Callable):
        self.hline.sigPositionChanged.connect(lambda: fn(self))

    def set_x_range(self, x0, x1):
        self.heatmap_pi.getViewBox().setXRange(x0, x1, padding=0)

    def set_y_range(self, y0, y1):
        self.heatmap_pi.getViewBox().setYRange(y0, y1, padding=0)

    def get_x_range(self):
        return self.heatmap_pi.getViewBox().getXRange()

    def get_y_range(self):
        return self.heatmap_pi.getViewBox().getYRange()

    def get_z_range(self):
        return self.hist.getLevels()

    def set_matrix(self, matrix, arr_ax0, arr_ax1, gradient=None, y_range=None, x_range=None, z_range=None,
                   center_lines=True):
        """

        :param matrix:
        :param arr_ax0:
        :param arr_ax1:
        :param gradient:
        :param y_range:  tuple '[min, max]' or None
        :param x_range:  tuple '[min, max]' or None
        :param z_range:  tuple '[min, max]' or None
        :return:
        """
        extreme = np.max(matrix) if np.abs(np.max(matrix)) > 1e-10 else np.min(matrix)

        matrix_min = -np.abs(extreme)
        matrix_max = -matrix_min

        x0, y0, x_max, y_max = arr_ax1[0], arr_ax0[0], arr_ax1[-1], arr_ax0[-1]
        self.image.setImage(matrix.T, rect=(x0, y0, x_max - x0, y_max - y0))   # [x, y, w, h]
        self.image.render()

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

        self.set_xy_range(x_range[0] if x_range else arr_ax1[0], x_range[1] if x_range else arr_ax1[-1],
                          y_range[0] if y_range else arr_ax0[0], y_range[1] if y_range else arr_ax0[-1])

        arr_ax0_diff = np.abs(arr_ax0[1:] - arr_ax0[:-1])
        arr_ax1_diff = np.abs(arr_ax1[1:] - arr_ax1[:-1])

        self.arr_ax0 = arr_ax0  # set times
        self.arr_ax1 = arr_ax1
        if not np.allclose(arr_ax0_diff, arr_ax0_diff[0], atol=0):  # if loaded data are not spaced linearly,
            self.string_axis_left.transform = self.transform_t_pos
            self.string_axis_right.transform = self.transform_t_pos
            self.y_arr_nonlinear = True

        if not np.allclose(arr_ax1_diff, arr_ax1_diff[0], atol=0):
            self.string_axis_bottom.transform = self.transform_wl_pos
            self.string_axis_top.transform = self.transform_wl_pos
            self.x_arr_nonlinear = True

        self.vline.setBounds((x0, x_max))
        self.hline.setBounds((y0, y_max))

        if center_lines:
            self.vline.setPos((x0 + x_max) / 2)
            self.hline.setPos((y0 + y_max) / 2)

    def get_xpos(self):
        return self.vline.pos()[0]

    def set_xpos(self, value):
        self.vline.setPos(value)

    def get_ypos(self):
        return self.hline.pos()[1]

    def set_ypos(self, value):
        self.hline.setPos(value)

    def transform_t_pos(self, value):
        return transform_value_pos(value, self.arr_ax0 if self.y_arr_nonlinear else None)

    def inv_transform_t_pos(self, value):
        return inv_transform_value_pos(value, self.arr_ax0 if self.y_arr_nonlinear else None)

    def transform_wl_pos(self, value):
        return transform_value_pos(value, self.arr_ax1 if self.x_arr_nonlinear else None)

    def inv_transform_wl_pos(self, value):
        return inv_transform_value_pos(value, self.arr_ax1 if self.x_arr_nonlinear else None)

    def set_xy_range(self, x0, x1, y0, y1, padding=0):
        y0, y1 = self.inv_transform_t_pos(y0), self.inv_transform_t_pos(y1)
        x0, x1 = self.inv_transform_wl_pos(x0), self.inv_transform_wl_pos(x1)

        self.heatmap_pi.getViewBox().setRange(xRange=[x0, x1], yRange=[y0, y1], padding=padding)

    def mouse_moved(self, pos):
        in_scene = self.heatmap_pi.sceneBoundingRect().contains(pos)
        on_vline = self.vline.sceneBoundingRect().contains(pos)
        on_hline = self.hline.sceneBoundingRect().contains(pos) or \
                   self.hist.regions[0].lines[0].sceneBoundingRect().contains(pos) or \
                   self.hist.regions[0].lines[1].sceneBoundingRect().contains(pos)

        on_hist_lut = self.hist.vb.sceneBoundingRect().contains(pos)

        # set the corresponding cursor
        if in_scene:
            mouse_point = self.heatmap_pi.vb.mapSceneToView(pos)
            n = Settings.coordinates_sig_figures
            # double format with n being the number of significant figures of a number
            self.probe_label.setText(f"x={{:.{n}g}}, y={{:.{n}g}}".format(mouse_point.x(), mouse_point.y()))

            if on_vline:
                self.parentWidget.viewport().setCursor(Qt.CursorShape.SizeHorCursor)
            elif on_hline:
                self.parentWidget.viewport().setCursor(Qt.CursorShape.SizeVerCursor)
            elif on_hist_lut:
                self.parentWidget.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.parentWidget.viewport().setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.probe_label.setText("<span style='color: #808080'>No data at cursor</span>")

    def mouse_clicked(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton and ev.modifiers() == Qt.KeyboardModifier.ControlModifier:
            pos = self.heatmap_pi.getViewBox().mapToView(ev.pos())

            self.set_xpos(pos.x())
            self.set_ypos(pos.y())



