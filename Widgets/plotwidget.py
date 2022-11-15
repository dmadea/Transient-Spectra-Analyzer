import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt

from settings import Settings
from typing import Callable
from misc import find_nearest_idx

from pyqtgraph.functions import mkBrush, mkColor, mkPen

from pyqtgraphmodif.infinite_line_modif import InfiniteLine
from Widgets.genericlayoutwidget import GenericLayoutWidget
from pyqtgraphmodif.StringAxis import StringAxis
from Widgets.genericplotlayout import GenericPlotLayout
from pyqtgraph import TextItem

from scipy.signal import find_peaks


class PlotWidget(GenericLayoutWidget):

    def initialize(self):
        self.plots = [PlotLayout(self)]
        self.addItem(self.plots[0], 0, 0)
        self.plots[0].connect_signals()

    def set_data(self, matrices, axis=0, title_prefix='Spectrum [', title_postfix=']'):

        self.clear_plots()

        n = len(matrices)

        if n > 1:
            self.plots += [PlotLayout(self) for i in range(n - 1)]

            idxs = self.get_mode_idxs(self.default_mode, n)

            for plot, (r, c) in zip(self.plots[1:], idxs[1:]):
                self.addItem(plot, r, c)
                plot.connect_signals()

        for plot, m in zip(self.plots, matrices):
            plot.arr_ax0 = m.times if axis == 0 else m.wavelengths
            min_val, max_val = plot.arr_ax0[0], plot.arr_ax0[-1]
            plot.set_limits(min_val, max_val,  m.D.min(), m.D.max())
            plot.title = f"{title_prefix}{m.get_filename()}{title_postfix}"

    def get_positions(self):
        return [plot.vline.pos()[0] for plot in self.plots]

    def connect_v_lines_position_changed(self, fn: Callable):
        for plot in self.plots:
            plot.connect_v_line_position_changed(fn)

    def connect_ranges_changed(self, fn: Callable):
        for plot in self.plots:
            plot.connect_range_changed(fn)

    # def get_plotted_data(self):
    #     return [p.plotted_data for p in self.plots]

    def clear_plots(self):
        for plot in self.plots:
            plot.disconnect_signals()
            plot.clear()

        self.ci.clear()
        self.plots = []
        self.initialize()

    def set_lines_color(self):
        for plot in self.plots:
            plot.vline.label.setColor(mkColor(Settings.infinite_line_label_color))

    def show_peaks(self, show=True):
        for plot in self.plots:
            plot.show_peaks = show
            x, y = plot.plotted_data.getData()
            plot.set_main_data(x, y)


class PlotLayout(GenericPlotLayout):

    def __init__(self, parent=None, border=None, ylabel='\u0394A', xlabel='Wavelength (nm)'):
        super(PlotLayout, self).__init__(None, border)
        self.parentWidget = parent

        self.arr_ax0 = None
        self.peak_labels = []
        self.show_peaks = False
        self.title = ''

        self.string_axis_left = StringAxis(orientation='left', keep_constant_space=True)
        self.string_axis_right = StringAxis(orientation='right', keep_constant_space=True)

        self.string_axis_bottom = StringAxis(orientation='bottom')
        self.string_axis_top = StringAxis(orientation='top')

        # heat map plot item
        self.plot_item = self.addPlot(axisItems={'left': self.string_axis_left,
                                      'right': self.string_axis_right,
                                      'bottom': self.string_axis_bottom,
                                      'top': self.string_axis_top})

        self.plotted_data = self.plot_item.plot([])
        self.fit_data = self.plot_item.plot([])
        self.factored_data = self.plot_item.plot([])
        self.probe_label = pg.LabelItem("<span style='color: #808080'>No data at cursor</span>", justify='left')
        self.addItem(self.probe_label, 1, 0)

        label_format = f'{{value:.{Settings.coordinates_sig_figures}g}}'

        self.vline = InfiniteLine(self.arr_ax0, angle=90, movable=True, label=label_format,
                                  labelOpts=dict(rotateAxis=(1, 0), position=Settings.heatmap_line_label_position))
        self.vline.setZValue(10000)
        self.plot_item.addItem(self.vline)

        # self.vline = self.plot_item.addLine(0, 0, 10000, angle=90, movable=True, pen=pg.mkPen((0, 0, 0)), label=label_format,
        #                                     labelOpts=dict(rotateAxis=(1, 0),
        #                                                    position=Settings.heatmap_line_label_position,
        #                                                    color=Settings.infinite_line_label_color))

        self.vline.sigPositionChanged.connect(self.vline_moved)

        brush = mkBrush(color=(255, 255, 255, Settings.infinite_line_label_brush_alpha))

        self.plot_item.addItem(self.vline, ignoreBounds=True)

        self.plot_item.showAxis('top', show=False)
        self.plot_item.showAxis('right', show=False)

        self.plot_item.setLabel('left', text=ylabel)
        self.plot_item.setLabel('bottom', text=xlabel)
        self.plot_item.showGrid(x=True, y=True, alpha=0.1)

        self.vline.label.fill = brush
        self.vline.label.setColor(Settings.infinite_line_label_color)

        self.layout.setSpacing(0)
        self.setContentsMargins(0, 0, 0, 0)

    def set_main_data(self, x: np.ndarray, y: np.ndarray, pen=None, add2title=''):
        pen = pen if pen is not None else mkPen(color=(0, 0, 0), width=1)

        self.plotted_data.setData(x, y, pen=pen)
        self.plot_item.setTitle(f'{self.title} at {add2title}', size=Settings.plot_title_font_size)

        if len(self.peak_labels) > 0:
            for label in self.peak_labels:
                self.plot_item.removeItem(label)
            self.peak_labels.clear()

        if self.show_peaks:
            x0, x1 = self.plot_item.getViewBox().state['viewRange'][0]
            self.plot_peaks(x0, x1)

    def set_factored_data(self, x: np.ndarray, y: np.ndarray, pen=None):
        pen = pen if pen is not None else mkPen(color=(0, 255, 0), width=1)
        self.factored_data.setData(x, y, pen=pen)

    def set_fit_data(self, x: np.ndarray, y: np.ndarray, pen=None):
        pen = pen if pen is not None else mkPen(color=(255, 0, 0), width=1)
        self.fit_data.setData(x, y, pen=pen)

    def vline_moved(self, line):
        if self.arr_ax0 is None:
            return

        x = self.get_xpos()
        new_x = self.arr_ax0[find_nearest_idx(self.arr_ax0, x)]

        self.set_xpos(new_x)

    def set_limits(self, xmin, xmax, ymin, ymax):
        ydiff = ymax - ymin
        self.plot_item.getViewBox().setLimits(xMin=xmin, xMax=xmax, yMin=ymin - 0.05 * ydiff, yMax=ymax + 0.05 * ydiff)
        self.vline.setBounds((xmin, xmax))

    def get_xpos(self):
        return self.vline.pos()[0]

    def set_xpos(self, value):
        self.vline.setPos(value)

    def set_x_range(self, x0, x1):
        self.plot_item.getViewBox().setXRange(x0, x1, padding=0)

    def connect_signals(self):
        self.plot_item.scene().sigMouseMoved.connect(self.mouse_moved)
        self.plot_item.scene().sigMouseClicked.connect(self.mouse_clicked)
        self.plot_item.getViewBox().sigXRangeChanged.connect(self.x_range_changed)

    def disconnect_signals(self):
        self.plot_item.scene().sigMouseMoved.disconnect(self.mouse_moved)
        self.plot_item.scene().sigMouseClicked.disconnect(self.mouse_clicked)
        self.plot_item.getViewBox().sigXRangeChanged.disconnect(self.x_range_changed)

    def connect_range_changed(self, fn: Callable):
        self.plot_item.getViewBox().sigRangeChanged.connect(lambda *args: fn(self, *args))

    def connect_v_line_position_changed(self, fn: Callable):
        self.vline.sigPositionChanged.connect(lambda: fn(self))

    def plot_peaks(self, x0, x1):
        x, y = self.plotted_data.getData()

        idx0, idx1 = find_nearest_idx(x, (x0, x1))
        x = x[idx0:idx1 + 1]
        y = y[idx0:idx1 + 1]

        distance = x.shape[0] / Settings.peak_density
        distance = 1 if distance < 1 else distance

        peaks = find_peaks(y, distance=distance)[0]

        label_format = f'{{value:.{Settings.coordinates_sig_figures}g}}'
        for xval, yval in zip(x[peaks], y[peaks]):
            ti = TextItem(label_format.format(value=xval), color=Settings.peak_color, anchor=(0.5, 1))
            self.peak_labels.append(ti)
            self.plot_item.addItem(ti)
            ti.setPos(xval, yval)

    def x_range_changed(self, sender_vb, ranges):
        x0, x1 = ranges

        if self.show_peaks:
            if len(self.peak_labels) > 0:
                for label in self.peak_labels:
                    self.plot_item.removeItem(label)
                self.peak_labels.clear()
            self.plot_peaks(x0, x1)

    def mouse_moved(self, pos):
        in_scene = self.plot_item.sceneBoundingRect().contains(pos)
        on_vline = self.vline.sceneBoundingRect().contains(pos)

        # set the corresponding cursor
        if in_scene:
            mouse_point = self.plot_item.vb.mapSceneToView(pos)
            n = Settings.coordinates_sig_figures
            # double format with n being the number of significant figures of a number
            self.probe_label.setText(f"x={{:.{n}g}}, y={{:.{n}g}}".format(mouse_point.x(), mouse_point.y()))

            if on_vline:
                self.parentWidget.viewport().setCursor(Qt.CursorShape.SizeHorCursor)
            else:
                self.parentWidget.viewport().setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.probe_label.setText("<span style='color: #808080'>No data at cursor</span>")

    def mouse_clicked(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton and ev.modifiers() == Qt.KeyboardModifier.ControlModifier:
            pos = self.plot_item.getViewBox().mapToView(ev.pos())

            self.set_xpos(pos.x())


