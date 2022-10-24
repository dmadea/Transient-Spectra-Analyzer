

from pyqtgraph.dockarea import DockLabel as _DockLabel
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMenu
from PyQt6.QtGui import QAction, QCursor


class DockDisplayMode:
    Row = 0
    Column = 1
    Matrix = 2


class DockLabel(_DockLabel):

    def __init__(self, text, widget, closable=False, fontSize="12px", display_mode=DockDisplayMode.Row):
        super(DockLabel, self).__init__(text, closable, fontSize)

        self.widget = widget
        self.mode = display_mode

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
