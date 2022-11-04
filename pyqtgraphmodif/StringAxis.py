import pyqtgraph as pg
import numpy as np
from settings import Settings


## help from https://stackoverflow.com/questions/31775468/show-string-values-on-x-axis-in-pyqtgraph
class StringAxis(pg.AxisItem):
    """AxisItem subclass for correctly labeling ticks on heatmap, this is necessary for nonlinearly-spaced data."""
    def __init__(self, orientation='left', pen=None, textPen=None, tickPen = None, linkView=None, parent=None, maxTickLength=-5, showValues=True,
                 text='', units='', unitPrefix='', transform=None, keep_constant_space=False, **args):

        super(StringAxis, self).__init__(orientation, pen, textPen, tickPen, linkView, parent, maxTickLength, showValues, text, units, unitPrefix, **args)
        self.transform = transform  # transform function
        self.keep_constant_space = keep_constant_space

        if self.keep_constant_space:
            self.style['autoExpandTextSpace'] = False
            self.style['tickTextWidth'] = 35

        # self.direction = ">" if orientation == 'left' else "<"
        # self.digits = "7"

    def tickStrings(self, values, scale, spacing):
        if self.transform is None:
            strings = super(StringAxis, self).tickStrings(values, scale, spacing)
            return strings
            # label_format = f'{{s:{self.direction}{self.digits}}}'
            # strings = [label_format.format(s=s) for s in strings] if self.keep_constant_space else strings
            # return strings

        vs = np.asarray(values)
        tr_values = self.transform(vs) * scale
        # label_format = f'{{value:{self.direction}{self.digits}.{Settings.coordinates_sig_figures}g}}' if self.keep_constant_space else f'{{value:.{Settings.coordinates_sig_figures}g}}'
        label_format = f'{{value:.{Settings.coordinates_sig_figures}g}}'
        strings = [label_format.format(value=v) for v in tr_values]

        return strings
