# Modified QLineEdit with added lost_focus signal

from PyQt6.QtWidgets import QLineEdit
from PyQt6.QtCore import pyqtSignal


class MyLineEdit(QLineEdit):
    focus_lost = pyqtSignal()

    def focusOutEvent(self, e):
        self.focus_lost.emit()
        super(MyLineEdit, self).focusOutEvent(e)