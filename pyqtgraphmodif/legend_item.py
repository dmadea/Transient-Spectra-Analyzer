
from pyqtgraph.graphicsItems.GraphicsWidget import GraphicsWidget
# from pyqtgraph.graphicsItems.LabelItem import LabelItem
from pyqtgraphmodif.label_item import LblItem
from PyQt5 import QtGui, QtCore
from pyqtgraph import functions as fn
from pyqtgraph.Point import Point
from pyqtgraph.graphicsItems.ScatterPlotItem import ScatterPlotItem, drawSymbol
from pyqtgraph.graphicsItems.PlotDataItem import PlotDataItem
from pyqtgraph.graphicsItems.GraphicsWidgetAnchor import GraphicsWidgetAnchor


class LegendItem(GraphicsWidget, GraphicsWidgetAnchor):
    """
    Displays a legend used for describing the contents of a pyqtgraph-modif.
    LegendItems are most commonly created by calling PlotItem.addLegend().

    Note that this item should not be added directly to a PlotItem. Instead,
    Make it a direct descendant of the PlotItem::

        legend.setParentItem(plotItem)

    """
    # added spacing argument
    def __init__(self, size=None, spacing=5, offset=None):
        """
        ==============  ===============================================================
        **Arguments:**
        size            Specifies the fixed size (width, height) of the legend. If
                        this argument is omitted, the legend will autimatically resize
                        to fit its contents.
        offset          Specifies the offset position relative to the legend's parent.
                        Positive values offset from the left or top; negative values
                        offset from the right or bottom. If offset is None, the
                        legend must be anchored manually by calling anchor() or
                        positioned by calling setPos().
        ==============  ===============================================================

        """

        GraphicsWidget.__init__(self)
        GraphicsWidgetAnchor.__init__(self)
        self.setFlag(self.ItemIgnoresTransformations)
        self.layout = QtGui.QGraphicsGridLayout()
        self.setLayout(self.layout)
        self.items = []
        self.size = size
        self.offset = offset
        if size is not None:
            self.setGeometry(QtCore.QRectF(0, 0, self.size[0], self.size[1]))


        self.spacing = spacing

    def setParentItem(self, p):
        ret = GraphicsWidget.setParentItem(self, p)
        if self.offset is not None:
            offset = Point(self.offset)
            anchorx = 1 if offset[0] <= 0 else 0
            anchory = 1 if offset[1] <= 0 else 0
            anchor = (anchorx, anchory)
            self.anchor(itemPos=anchor, parentPos=anchor, offset=offset)
        return ret

    def addItem(self, item, name):
        """
        Add a new entry to the legend.

        ==============  ========================================================
        **Arguments:**
        item            A PlotDataItem from which the line and point style
                        of the item will be determined or an instance of
                        ItemSample (or a subclass), allowing the item display
                        to be customized.
        title           The title to display for this item. Simple HTML allowed.
        ==============  ========================================================
        """
        # label = LabelItem(name) # original code
        # added left justify
        label = LblItem(name, spacing=self.spacing, justify='left')

        if isinstance(item, ItemSample):
            sample = item
        else:
            sample = ItemSample(item)
        row = self.layout.rowCount()
        self.items.append((sample, label))
        self.layout.addItem(sample, row, 0)
        self.layout.addItem(label, row, 1)
        self.updateSize()

    def removeLastItem(self):
        sample, label = self.items[-1]
        self.items.remove((sample, label))  # remove from itemlist
        self.layout.removeItem(sample)  # remove from layout
        sample.close()  # remove from drawing
        self.layout.removeItem(label)
        label.close()
        self.updateSize()  # redraq box

    def removeItem(self, name):
        """
        Removes one item from the legend.

        ==============  ========================================================
        **Arguments:**
        title           The title displayed for this item.
        ==============  ========================================================
        """
        # Thanks, Ulrich!
        # cycle for a match
        for sample, label in self.items:
            if label.text == name:  # hit
                self.items.remove((sample, label))  # remove from itemlist
                self.layout.removeItem(sample)  # remove from layout
                sample.close()  # remove from drawing
                self.layout.removeItem(label)
                label.close()
                self.updateSize()  # redraq box

    def updateSize(self):
        if self.size is not None:
            return

        height = 0
        width = 0
        # print("-------")
        for sample, label in self.items:
            height += max(sample.height(), label.height()) + 3
            width = max(width, sample.width() + label.width())
            # print(width, height)
        # print width, height
        self.setGeometry(0, 0, width + 25, height)

    def boundingRect(self):
        return QtCore.QRectF(0, 0, self.width(), self.height())

    def paint(self, p, *args):
        # this lines were commented out, removal of the awful gray rectangle over the legend, also removal of white border
        p.setPen(fn.mkPen(255, 255, 255, 0))
        # p.setBrush(fn.mkBrush(100,100,100,50))
        p.drawRect(self.boundingRect())

    def hoverEvent(self, ev):
        ev.acceptDrags(QtCore.Qt.LeftButton)

    def mouseDragEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            dpos = ev.pos() - ev.lastPos()
            self.autoAnchor(self.pos() + dpos)


class ItemSample(GraphicsWidget):
    """ Class responsible for drawing a single item in a LegendItem (sans label).

    This may be subclassed to draw custom graphics in a Legend.
    """

    ## Todo: make this more generic; let each item decide how it should be represented.
    def __init__(self, item):
        GraphicsWidget.__init__(self)
        self.item = item

        # added size hint, sample dimensions changed from 20, 20 to 22, 20
        self._sizeHint = {
            QtCore.Qt.MinimumSize: (22, 0),
            QtCore.Qt.PreferredSize: (22, 0),
            QtCore.Qt.MaximumSize: (-1, -1),  # bounds.width()*2, bounds.height()*2),
            QtCore.Qt.MinimumDescent: (0, 0)  ##?? what is this?
        }

    def boundingRect(self):
        return QtCore.QRectF(0, 0, 22, 20)
        # return QtCore.QRectF(0, 0, 100, 100)

    def paint(self, p, *args):
        # p.setRenderHint(p.Antialiasing)  # only if the data is antialiased.
        opts = self.item.opts

        if opts.get('fillLevel', None) is not None and opts.get('fillBrush', None) is not None:
            p.setBrush(fn.mkBrush(opts['fillBrush']))
            p.setPen(fn.mkPen(None))
            p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(2, 18), QtCore.QPointF(18, 2), QtCore.QPointF(18, 18)]))

        if not isinstance(self.item, ScatterPlotItem):
            p.setPen(fn.mkPen(opts['pen']))
            # p.drawLine(2, 18, 18, 2)
            # changed to straight line, instead of 45° line
            p.drawLine(0, 12, 22, 12)

        symbol = opts.get('symbol', None)
        if symbol is not None:
            if isinstance(self.item, PlotDataItem):
                opts = self.item.scatter.opts

            pen = fn.mkPen(opts['pen'])
            brush = fn.mkBrush(opts['brush'])
            size = opts['size']

            p.translate(10, 10)
            path = drawSymbol(p, symbol, size, pen, brush)

    def sizeHint(self, hint, constraint):
        if hint not in self._sizeHint:
            return QtCore.QSizeF(0, 0)
        return QtCore.QSizeF(*self._sizeHint[hint])

