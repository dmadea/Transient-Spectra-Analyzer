import pyqtgraph as pg

from pyqtgraphmodif.LegendItemModif import LegendItemModif
# from pyqtgraph.graphicsItems.LegendItem import LegendItem
from Widgets.heatmap import Heatmap


class FitLayout(pg.GraphicsLayoutWidget):
    """Layout of fit widget, only C, ST and residual matrices are displayed here."""
    instance = None

    def __init__(self, set_coordinate_func=None, parent=None):
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOptions(antialias=True)

        super(FitLayout, self).__init__(parent)

        FitLayout.instance = self

        self.set_coordinate_func = set_coordinate_func

        self.ST_plot = self.ci.addPlot(title="S<sup>T</sup> Matrix")

        self.ST_plot.showAxis('top', show=True)
        self.ST_plot.showAxis('right', show=True)

        self.ST_plot.setLabel('left', text='\u0394A')
        self.ST_plot.setLabel('bottom', text='Wavelength (nm)')

        self.C_plot = self.ci.addPlot(title="C Matrix")

        self.C_plot.showAxis('top', show=True)
        self.C_plot.showAxis('right', show=True)

        self.C_plot.setLabel('left', text='\u0394A')
        self.C_plot.setLabel('bottom', text='Time (ps)')

        self.ST_plot.showGrid(x=True, y=True, alpha=0.1)
        self.C_plot.showGrid(x=True, y=True, alpha=0.1)

        self.heat_map_plot = Heatmap()
        self.addItem(self.heat_map_plot)

        self.C_legend = None
        self.ST_legend = None

        self.add_legend()

    def add_legend(self, size=None, spacing=0, offset=(-30, 30)):

        try:
            # self.grpView.plotItem.legend.scene().removeItem(self.grpView.plotItem.legend)
            self.C_legend.scene().removeItem(self.C_legend)
            self.ST_legend.scene().removeItem(self.ST_legend)
        except:
            pass

        self.C_legend = LegendItemModif(size, verSpacing=spacing, offset=offset)
        self.C_legend.setParentItem(self.C_plot.vb)
        self.C_plot.legend = self.C_legend

        self.ST_legend = LegendItemModif(size, verSpacing=spacing, offset=offset)
        self.ST_legend.setParentItem(self.ST_plot.vb)
        self.ST_plot.legend = self.ST_legend

        # return self.legend