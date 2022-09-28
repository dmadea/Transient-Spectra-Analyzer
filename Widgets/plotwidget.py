import pyqtgraph as pg
from PyQt6.QtCore import Qt

from settings import Settings
from typing import Callable


class PlotWidget(pg.GraphicsLayoutWidget):

    def __init__(self, parent=None, border=None):

        super(PlotWidget, self).__init__(parent, border)

        self.plots = [PlotLayout(self)]
        self.addItem(self.plots[0], 0, 0)
        self.plots[0].connect_signals()
        # self.plots = [PlotLayout(self) for i in range(2)]

        # for i, plot in enumerate(self.plots):
        #     self.addItem(plot, 0, i)
        #     plot.connect_signals()

    def set_data(self, matrices, axis=0, title_prefix='Spectrum [', title_postfix=']'):

        # TODO delete previous data

        n = len(matrices)

        if n > 1:
            self.plots += [PlotLayout(self) for i in range(n - 1)]

            idxs = list(zip([0] * n, list(range(n))))

            for h, (r, c) in zip(self.plots[1:], idxs[1:]):
                self.addItem(h, r, c)
                h.connect_signals()

        for h, m in zip(self.plots, matrices):
            min_val, max_val = (m.times[0], m.times[-1]) if axis == 0 else (m.wavelengths[0], m.wavelengths[-1])
            h.set_limits(min_val, max_val, m.D.min(), m.D.max())
            h.plot_item.setTitle(f"{title_prefix}{m.get_filename()}{title_postfix}")

    def get_positions(self):
        return [plot.vline.pos()[0] for plot in self.plots]

    def connect_v_lines_position_changed(self, fn: Callable):
        for plot in self.plots:
            plot.vline.sigPositionChanged.connect(fn)

    def get_plots(self):
        return [p.plotted_data for p in self.plots]


class PlotLayout(pg.GraphicsLayout):

    def __init__(self, parent=None, border=None, ylabel='\u0394A', xlabel='Wavelength (nm)'):
        super(PlotLayout, self).__init__(None, border)
        self.parentWidget = parent

        self.plot_item = self.addPlot()
        self.plotted_data = self.plot_item.plot([])

        label_format = f'{{value:.{Settings.coordinates_sig_figures}g}}'

        self.vline = self.plot_item.addLine(angle=90, movable=True, pen=pg.mkPen((0, 0, 0)), label=label_format,
                                            labelOpts=dict(rotateAxis=(1, 0), color=Settings.infinite_line_label_color))
        self.plot_item.addItem(self.vline, ignoreBounds=True)

        self.plot_item.showAxis('top', show=True)
        self.plot_item.showAxis('right', show=True)

        self.plot_item.setLabel('left', text=ylabel)
        self.plot_item.setLabel('bottom', text=xlabel)
        self.plot_item.showGrid(x=True, y=True, alpha=0.1)

        self.proxy = None

        # signals
        # self.range_changed = self.heatmap_pi.getViewBox().sigRangeChanged

    def set_limits(self, xmin, xmax, ymin, ymax):
        self.plot_item.getViewBox().setLimits(xMin=xmin, xMax=xmax, yMin=ymin, yMax=ymax)
        self.vline.setBounds((xmin, xmax))

    def connect_signals(self):
        self.proxy = pg.SignalProxy(self.plot_item.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_moved)

    def mouse_moved(self, ev):
        pos = ev[0]
        in_scene = self.plot_item.sceneBoundingRect().contains(pos)
        on_vline = self.vline.sceneBoundingRect().contains(pos)

        # set the corresponding cursor
        if in_scene:
            if on_vline:
                self.parentWidget.viewport().setCursor(Qt.CursorShape.SizeHorCursor)
            else:
                self.parentWidget.viewport().setCursor(Qt.CursorShape.CrossCursor)


