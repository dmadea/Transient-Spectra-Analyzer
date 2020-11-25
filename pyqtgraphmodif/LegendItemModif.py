
from pyqtgraph.graphicsItems.LegendItem import LegendItem, ItemSample
from pyqtgraphmodif.LabelItemModif import LabelItemModif

from PyQt5 import QtGui, QtCore
from pyqtgraph import functions as fn
from pyqtgraph.graphicsItems.ScatterPlotItem import ScatterPlotItem, drawSymbol
from pyqtgraph.graphicsItems.PlotDataItem import PlotDataItem
from pyqtgraph.graphicsItems.BarGraphItem import BarGraphItem


class LegendItemModif(LegendItem):

    def __init__(self, size=None, offset=None, horSpacing=25, verSpacing=0,
                 pen=None, brush=None, labelTextColor=None, frame=True,
                 labelTextSize='9pt', rowCount=1, colCount=1,  **kwargs):

        super(LegendItemModif, self).__init__(size, offset, horSpacing, verSpacing,
                 pen, brush, labelTextColor, frame,
                 labelTextSize, rowCount, colCount, **kwargs)

        self.verSpacing = verSpacing  # verSpacing parameter used as a setup for setMinimumHeight in LabelItemModif

    def addItem(self, item, name):
        """
        Add a new entry to the legend.

        ==============  ========================================================
        **Arguments:**
        item            A :class:`~pyqtgraph.PlotDataItem` from which the line
                        and point style of the item will be determined or an
                        instance of ItemSample (or a subclass), allowing the
                        item display to be customized.
        title           The title to display for this item. Simple HTML allowed.
        ==============  ========================================================
        """
        # USED LabelItemModif insted of LabelItem
        label = LabelItemModif(name, color=self.opts['labelTextColor'],
                          justify='left', size=self.opts['labelTextSize'], verspacing=self.verSpacing)
        if isinstance(item, ItemSampleModif):  # Changed from ItemSample to ItemSampleModif
            sample = item
        else:
            sample = ItemSampleModif(item)  # Changed from ItemSample to ItemSampleModif
        self.items.append((sample, label))
        self._addItemToLayout(sample, label)
        self.updateSize()


class ItemSampleModif(ItemSample):

    def paint(self, p, *args):
        opts = self.item.opts

        if opts.get('antialias'):
            p.setRenderHint(p.Antialiasing)

        if not isinstance(self.item, ScatterPlotItem):
            p.setPen(fn.mkPen(opts['pen']))
            # p.drawLine(0, 11, 20, 11)
            p.drawLine(0, 15, 20, 15)  # CHANGED THIS LINE

            if (opts.get('fillLevel', None) is not None and
                    opts.get('fillBrush', None) is not None):
                p.setBrush(fn.mkBrush(opts['fillBrush']))
                p.setPen(fn.mkPen(opts['fillBrush']))
                p.drawPolygon(QtGui.QPolygonF(
                    [QtCore.QPointF(2, 18), QtCore.QPointF(18, 2),
                     QtCore.QPointF(18, 18)]))

        symbol = opts.get('symbol', None)
        if symbol is not None:
            if isinstance(self.item, PlotDataItem):
                opts = self.item.scatter.opts
            p.translate(10, 10)
            drawSymbol(p, symbol, opts['size'], fn.mkPen(opts['pen']),
                       fn.mkBrush(opts['brush']))

        if isinstance(self.item, BarGraphItem):
            p.setBrush(fn.mkBrush(opts['brush']))
            p.drawRect(QtCore.QRectF(2, 2, 18, 18))
