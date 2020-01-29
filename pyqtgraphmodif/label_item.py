from PyQt5 import QtCore

from pyqtgraph.graphicsItems.LabelItem import LabelItem

__all__ = ['LabelItem']

class LblItem(LabelItem):
    """
    GraphicsWidget displaying text.
    Used mainly as axis labels, titles, etc.

    Note: To display text inside a scaled view (ViewBox, PlotWidget, etc) use TextItem
    """

    def __init__(self, text=' ', parent=None, angle=0, spacing=5, **args):
        self.spacing = spacing
        super(LblItem, self).__init__(text, parent, angle, **args)


    def updateMin(self):
        bounds = self.itemRect()
        self.setMinimumWidth(bounds.width())
        # changed
        self.setMinimumHeight(self.spacing)

        self._sizeHint = {
            QtCore.Qt.MinimumSize: (bounds.width(), bounds.height()),
            QtCore.Qt.PreferredSize: (bounds.width(), bounds.height()),
            QtCore.Qt.MaximumSize: (-1, -1),  # bounds.width()*2, bounds.height()*2),
            QtCore.Qt.MinimumDescent: (0, 0)  ##?? what is this?
        }

        self.updateGeometry()
    #
    # def sizeHint(self, hint, constraint):
    #     if hint not in self._sizeHint:
    #         return QtCore.QSizeF(0, 0)
    #     return QtCore.QSizeF(*self._sizeHint[hint])
    # #
    # def itemRect(self):
    #     return self.item.mapRectToParent(self.item.boundingRect())

    # def paint(self, p, *args):
    #     p.setPen(fn.mkPen('r'))
    #     p.drawRect(self.rect())
    #     p.setPen(fn.mkPen('g'))
    #     p.drawRect(self.itemRect())
    #
