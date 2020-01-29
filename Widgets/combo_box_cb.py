from PyQt5.QtWidgets import QComboBox, QApplication
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtCore import Qt

import fitting.constraints
import sys, inspect


class ComboBoxCB(QComboBox):

    def __init__(self, parent=None):
        super(ComboBoxCB, self).__init__(parent)

        self.setEditable(True)

        self.items = []  # list of checkboxes
        self._constraints = []

        classes = inspect.getmembers(sys.modules[fitting.constraints.__name__], inspect.isclass)
        tuples = filter(lambda tup: issubclass(tup[1], fitting.constraints.Constraint)
                                    and tup[1].name is not 'name', classes)
        self._av_constraints = list(map(lambda tup: tup[1], tuples))

        self.n = len(self._av_constraints)
        self.model = QStandardItemModel(self.n, 1)

        for i in range(self.n):
            item = QStandardItem(self._av_constraints[i].name)

            item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setData(Qt.Unchecked, Qt.CheckStateRole)

            self.items.append(item)

            self.model.setItem(i, 0, item)

        self.setModel(self.model)
        self.model.dataChanged.connect(self.data_changed)
        self.set_text("")

    @property
    def constraints(self):

        self._constraints = []

        for i in range(self.n):
            if self.items[i].checkState() == Qt.Checked:
                self._constraints.append(self._av_constraints[i]())

        return self._constraints

    def data_changed(self, index1, index2):
        text = ','.join(
            self._av_constraints[i].abbr for i in range(self.n) if self.items[i].checkState() == Qt.Checked)
        self.set_text(text)

    def set_text(self, text):
        l_edit = self.lineEdit()
        l_edit.setText(text)
        l_edit.setReadOnly(True)


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    cb = ComboBoxCB()
    cb.show()
    sys.exit(app.exec_())
