import pyqtgraph as pg
import numpy as np

## help from https://stackoverflow.com/questions/31775468/show-string-values-on-x-axis-in-pyqtgraph
class StringAxis(pg.AxisItem):
    """AxisItem subclass for correctly labeling ticks on heatmap, this is necessary for nonlinearly-spaced data."""
    def __init__(self, transform=None, *args, **kwargs):
        pg.AxisItem.__init__(self, *args, **kwargs)
        self.transform = transform  # transform function

    def tickStrings(self, values, scale, spacing):
        if self.transform is None:
            return super(StringAxis, self).tickStrings(values, scale, spacing)

        vs = np.asarray(values) * scale
        tr_values = self.transform(vs)
        strings = [f'{v:.3g}' for v in tr_values]

        return strings
