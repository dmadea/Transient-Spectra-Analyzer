
from pyqtgraph.graphicsItems.InfiniteLine import InfiniteLine as pgInfiniteLine
from pyqtgraph.functions import mkPen
import numpy as np
from PyQt6.QtCore import Qt


class InfiniteLine(pgInfiniteLine):

    def __init__(self, datapoints: np.ndarray | None, pos=None, angle=90, pen=None, movable=False, bounds=None, hoverPen=None, label=None,
                 labelOpts=None, span=(0, 1), markers=None, name=None, dragPen=None):

        self.datapoints = datapoints

        color = (0, 150, 0)
        self.hoverPen = mkPen(color=color, width=2) if hoverPen is None else hoverPen
        self.dragPen = mkPen(color=color, width=2, style=Qt.PenStyle.DashLine) if dragPen is None else dragPen
        self.pen = mkPen(color='k', width=1) if pen is None else pen

        super().__init__(pos, angle, self.pen, movable, bounds, self.hoverPen, label, labelOpts, span, markers, name)

    #     self.sigPositionChangeFinished.connect(self.update_position)
    #
    # def update_position(self, line):
    #     if self.datapoints is None:
    #         return


    def mouseDragEvent(self, ev):
        super(InfiniteLine, self).mouseDragEvent(ev)
        if self.moving:
            self.currentPen = self.dragPen
        else:
            self.currentPen = self.pen



