from PyQt6 import QtCore

from pyqtgraph.graphicsItems.LabelItem import LabelItem

__all__ = ['LabelItem']


class LabelItemModif(LabelItem):
    """
    GraphicsWidget displaying text.
    Used mainly as axis labels, titles, etc.

    Note: To display text inside a scaled view (ViewBox, PlotWidget, etc) use TextItem
    """

    def __init__(self, text=' ', parent=None, angle=0, verspacing=5, **args):
        self.verspacing = verspacing  # added vertical spacing parameter
        super(LabelItemModif, self).__init__(text, parent, angle, **args)

    def updateMin(self):
        bounds = self.itemRect()
        self.setMinimumWidth(bounds.width())

        # changed here
        # self.setMinimumHeight(bounds.height())  # commented
        self.setMinimumHeight(self.verspacing)  # added

        self._sizeHint = {
            QtCore.Qt.MinimumSize: (bounds.width(), bounds.height()),
            QtCore.Qt.PreferredSize: (bounds.width(), bounds.height()),
            QtCore.Qt.MaximumSize: (-1, -1),  # bounds.width()*2, bounds.height()*2),
            QtCore.Qt.MinimumDescent: (0, 0)  ##?? what is this?
        }

        self.updateGeometry()
