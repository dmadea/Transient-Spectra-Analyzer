from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
import sys


class GenericInputDialog(QtWidgets.QDialog):

    # static variables
    is_opened = False
    _instance = None

    def __init__(self, widget_list=None, label_text='Some descriptive text...', title='GenericInputDialog',
                 set_result=None, parent=None):
        super(GenericInputDialog, self).__init__(parent, Qt.WindowStaysOnTopHint | Qt.MSWindowsFixedSizeDialogHint)

        # # disable resizing of the window,
        # # help from https://stackoverflow.com/questions/16673074/in-qt-c-how-can-i-fully-disable-resizing-a-window-including-the-resize-icon-w
        # self.setWindowFlags(Qt.Dialog | Qt.MSWindowsFixedSizeDialogHint)

        GenericInputDialog.is_opened = True
        GenericInputDialog._instance = self

        self.setWindowTitle(title)
        self.accepted = False

        self.set_result = set_result  # function

        # self.resize(300, 150)

        self.button_box = QtWidgets.QDialogButtonBox(self)
        self.button_box.setOrientation(QtCore.Qt.Horizontal)
        self.button_box.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self.VLayout = QVBoxLayout()
        # self.VLayout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel(label_text)

        self.VLayout.addWidget(self.label)

        self.grid_layout = QtWidgets.QGridLayout()
        self.grid_layout.setSpacing(7)
        self.grid_layout.setContentsMargins(15, 0, 0, 0)

        if widget_list is None:
            widget_list = [("par1", QLineEdit("some  text")),
                           ("par2", QLineEdit("another  text")),
                           ("par3", QCheckBox("...."))]

        # self.label_list = []
        self.widget_list = []

        i = 0
        for label, widget in widget_list:
            if isinstance(label, str):
                # self.label_list.append(QLabel(label))
                label = QLabel(label)
            if widget is not None:
                self.widget_list.append(widget)
                self.grid_layout.addWidget(widget, i, 1)

            if label is not None:
                self.grid_layout.addWidget(label, i, 0)
            i += 1

        self.VLayout.addItem(self.grid_layout)
        self.VLayout.addWidget(self.button_box)

        self.setLayout(self.VLayout)

        self.widget_list[0].setFocus()
        if isinstance(self.widget_list[0], QLineEdit):
            self.widget_list[0].selectAll()

        # self.show()
        # self.exec()
        # sys.exit(self.exec_())


    @classmethod
    def if_opened_activate(cls):
        """Returns True if dialog is already opened and has been activated, otherwise returns False."""
        if cls.is_opened:
            cls._instance.activateWindow()
            cls._instance.setFocus()
            return True
        else:
            return False

    # # virtual method
    # def set_result(self):
    #     pass

    def accept(self):
        self.set_result()
        self.accepted = True
        GenericInputDialog.is_opened = False
        GenericInputDialog._instance = None
        super(GenericInputDialog, self).accept()

    def reject(self):
        GenericInputDialog.is_opened = False
        GenericInputDialog._instance = None
        super(GenericInputDialog, self).reject()



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = GenericInputDialog()
    Dialog.show()
    sys.exit(app.exec_())