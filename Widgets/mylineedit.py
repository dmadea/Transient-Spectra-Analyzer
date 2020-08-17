# Modified QLineEdit with added lost_focus signal

from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtCore import pyqtSignal


class MyLineEdit(QLineEdit):
    focus_lost = pyqtSignal()

    def focusOutEvent(self, e):
        self.focus_lost.emit()
        super(MyLineEdit, self).focusOutEvent(e)