import pyqtgraph as pg
from pyqtgraphmodif.dock_modif import DockDisplayMode
import numpy as np


class GenericLayoutWidget(pg.GraphicsLayoutWidget):

    def __init__(self, parent=None, border=None, default_mode=DockDisplayMode.Row):

        super(GenericLayoutWidget, self).__init__(parent, border)

        self.parent = parent

        self.default_mode = default_mode
        self.plots = []
        self.initialize()

        self.ci.layout.setSpacing(0)
        self.ci.setContentsMargins(0, 0, 0, 0)

    def initialize(self):
        pass

    def get_labels(self):
        raise NotImplemented()

    def show_peaks(self):
        raise NotImplemented()

    def set_labels(self, index: int, x_label: str, x_label_unit: str, y_label: str, y_label_unit: str, z_label: str,
                   z_label_unit: str,):
        self.parent.set_axis_label(index, x_label, x_label_unit, y_label, y_label_unit, z_label, z_label_unit)

    def set_display_mode(self, mode: DockDisplayMode):
        for plot in self.plots:
            self.removeItem(plot)

        idxs = self.get_mode_idxs(mode, len(self.plots))

        for plot, (r, c) in zip(self.plots, idxs):
            self.addItem(plot, r, c)

    def get_mode_idxs(self, mode: DockDisplayMode, n: int):
        if mode == DockDisplayMode.Row:  # row-wise
            idxs = list(zip(list(range(n)), [0] * n))
        elif mode == DockDisplayMode.Column:  # column-wise
            idxs = list(zip([0] * n, list(range(n))))
        else:  # added column wise
            sqrt = n ** 0.5
            n_rows = int(sqrt)
            n_cols = int(np.ceil(sqrt))

            idxs = []

            if n_cols * n_rows < n:
                n_cols += 1

            for i in range(n_rows):
                for j in range(n_cols):
                    if len(idxs) < n:
                        idxs.append((i, j))

        return idxs