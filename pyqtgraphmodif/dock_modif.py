from pyqtgraph.dockarea import DockLabel as _DockLabel
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMenu
from PyQt6.QtGui import QAction, QCursor

from Widgets.settingsdialog import SettingsDialog


class DockDisplayMode:
    Row = 0
    Column = 1
    Matrix = 2


class DockLabel(_DockLabel):

    def __init__(self, text, widget, closable=False, fontSize="12px", display_mode=DockDisplayMode.Row,
                 show_setting_option=False):
        super(DockLabel, self).__init__(text, closable, fontSize)

        self.widget = widget
        self.show_setting_option = show_setting_option
        self.mode = display_mode
        self.sett_dialog = None

    def update_widget_options(self, parent, child, *args):
        if self.widget is None:
            return

        # print(parent, child)

        index = int(parent.opts['name'])

        self.widget.set_labels(index, parent['x_label'], parent['y_label'], parent['z_label'])

    def open_options(self):
        if self.widget is None:
            return

        mat_opt_template = [dict(name='x_label', type='str', value="Wavelength / nm"),
                            dict(name='y_label', type='str', value="Time / min"),
                            dict(name='z_label', type='str', value="\u0394A")]

        opts = []

        n = len(self.widget.plots)

        labels = self.widget.get_labels()

        for i in range(n):
            mat_opt = mat_opt_template.copy()
            mat_opt[0]['value'], mat_opt[1]['value'], mat_opt[2]['value'] = labels[i]
            opts.append(dict(name=f'{i}', title=f'Matrix {i} Options', type='group', children=mat_opt))

        self.sett_dialog = SettingsDialog(opts)

        for i in range(n):
            self.sett_dialog.params.child(f'{i}').sigTreeStateChanged.connect(self.update_widget_options)

        self.sett_dialog.show()

        # opts = [
        #     dict(name='matrix_1', title='Matrix # Options', type='group', children=[
        #         dict(name='x_label', type='str', value="Wavelength / nm"),
        #         dict(name='y_label', type='str', value="Time / min"),
        #         dict(name='z_label', type='str', value="\u0394A"),
        #         # dict(name='y_label', type='string', limits=[0, None], value=500),
        #     ]),
        #     dict(name='useOpenGL', type='bool', value=True,
        #          readonly=True),
        #     dict(name='plotMethod', title='Plot Method', type='list', limits=['pyqtgraph', 'drawPolyline'])
        # ]

    def create_menu(self):
        menu = QMenu()

        display_menu = QMenu("Display Mode")
        # group = QActionGroup(self)

        row = QAction("Row-wise")
        column = QAction("Column-wise")
        matrix = QAction("Matrix")

        display_menu.addAction(row)
        display_menu.addAction(column)
        display_menu.addAction(matrix)

        row.triggered.connect(lambda: self.set_mode(DockDisplayMode.Row))
        column.triggered.connect(lambda: self.set_mode(DockDisplayMode.Column))
        matrix.triggered.connect(lambda: self.set_mode(DockDisplayMode.Matrix))

        row.setCheckable(True)
        column.setCheckable(True)
        matrix.setCheckable(True)

        row.setChecked(self.mode == DockDisplayMode.Row)
        column.setChecked(self.mode == DockDisplayMode.Column)
        matrix.setChecked(self.mode == DockDisplayMode.Matrix)

        # row.setActionGroup(group)
        # column.setActionGroup(group)
        # matrix.setActionGroup(group)

        menu.addMenu(display_menu)

        # add settings
        if self.show_setting_option:
            sett_act = QAction("Settings")
            sett_act.triggered.connect(lambda: self.open_options())
            menu.addAction(sett_act)

        cursor = QCursor()
        menu.exec_(cursor.pos())

    def set_mode(self, mode):
        self.mode = mode
        if self.widget is not None:
            self.widget.set_display_mode(mode)

    def mousePressEvent(self, ev):
        super(DockLabel, self).mousePressEvent(ev)

        if ev.buttons() == Qt.MouseButton.RightButton:
            self.create_menu()
