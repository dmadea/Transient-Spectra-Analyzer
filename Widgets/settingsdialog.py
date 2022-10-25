
from PyQt6.QtWidgets import QDialog, QGridLayout
from pyqtgraph import mkPen
from pyqtgraph.parametertree import Parameter, ParameterTree


class SettingsDialog(QDialog):

    def __init__(self, options: list, title='Matrix options'):
        super(SettingsDialog, self).__init__()

        self.setWindowTitle(title)

        # children = [
        #     dict(name='matrix_1', title='Matrix # Options', type='group', children=[
        #         dict(name='x_label', type='str', value="Wavelength / nm"),
        #         dict(name='y_label', type='str', value="Time / min"),
        #         dict(name='z_label', type='str', value="\u0394A"),
        #         # dict(name='y_label', type='string', limits=[0, None], value=500),
        #     ]),
        #     dict(name='useOpenGL', type='bool', value=True,
        #          readonly=True),
        #     dict(name='pen', type='pen', value=mkPen('red')),
        #     dict(name='antialias', type='bool', value=True),
        #     dict(name='connect', type='list', limits=['all', 'pairs', 'finite', 'array'], value='all'),
        #     dict(name='fill', type='bool', value=False),
        #     dict(name='skipFiniteCheck', type='bool', value=False),
        #     dict(name='plotMethod', title='Plot Method', type='list', limits=['pyqtgraph', 'drawPolyline'])
        # ]

        self.params = Parameter.create(name='Parameters', type='group', children=options)
        self.pt = ParameterTree(showHeader=False)
        self.pt.setParameters(self.params)

        self.main_layout = QGridLayout(self)
        self.main_layout.addWidget(self.pt)

        self.setLayout(self.main_layout)



