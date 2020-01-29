from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5.QtCore import Qt
from dialogs.gui_fit import Ui_Dialog

from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QMessageBox, QLineEdit, QCheckBox, QLabel
import numpy as np

# from spectrum import Spectrum
# from scipy.optimize import curve_fit

# import pyqtgraph as pg

from logger import Logger

from user_namespace import global_fit

import fitmodels

import sys, inspect


# from settings import Settings


class FitDialog(QtWidgets.QDialog, Ui_Dialog):
    # static variables
    is_opened = False
    _instance = None

    max_params = 10
    max_species = 10

    def __init__(self, LFP_matrix, parent=None):
        super(FitDialog, self).__init__(parent, QtCore.Qt.WindowStaysOnTopHint)  # window stays on top
        self.setupUi(self)

        # self.plot_widget = plot_widget
        self.LFP_matrix = LFP_matrix

        # self.fitted_params = None
        # self.covariance_matrix = None
        # self.errors = None

        self.current_model = None

        self.setWindowTitle("Global Fit")

        # get all models from fitmodels, get classes that inherits from Model base class and sort them by name
        # and number of species
        classes = inspect.getmembers(sys.modules[fitmodels.__name__], inspect.isclass)
        tuples = filter(lambda tup: issubclass(tup[1], fitmodels._Model) and tup[1] is not fitmodels._Model, classes)
        self.models = sorted(list(map(lambda tup: tup[1], tuples)), key=lambda cls: (cls.name, cls.n_full))

        # fill the combo box items with model names
        self.cbModel.addItems(map(lambda m: m.name, self.models))

        self.methods = [
            {'name': 'Levenberg–Marquardt', 'abbr': 'leastsq'},
            {'name': 'Nelder-Mead, Simplex method (no error)', 'abbr': 'nelder'},
            {'name': 'L-BFGS-B (no error)', 'abbr': 'lbfgsb'},
            {'name': 'Powell (no error)', 'abbr': 'powell'}
        ]

        # fill the combo box items with method names
        self.cbMethod.addItems(map(lambda m: m['name'], self.methods))

        self.cbDisplaResult.setChecked(True)
        # self.cbUpdateInitValues.setCheckState(Qt.Checked)

        self.btnFit.clicked.connect(self.fit)
        self.btnCorrelation.clicked.connect(self.correlation_analysis)
        self.btnShowLastResult.clicked.connect(self.show_last_result)

        self.btnOK.clicked.connect(self.accept)
        self.btnCancel.clicked.connect(self.reject)
        self.cbModel.currentIndexChanged.connect(self.model_changed)

        self.species_list = []
        self.visible_list = []
        self.spectra_list = []

        for i in range(self.max_species):
            self.species_list.append(QLabel())
            self.visible_list.append(QCheckBox())
            self.spectra_list.append(QLabel(''))

            self.species_list[i].setAlignment(Qt.AlignLeft)

            self.gridLayout_2.addWidget(self.species_list[i], i + 1, 0, 1, 1)
            self.gridLayout_2.addWidget(self.visible_list[i], i + 1, 1, 1, 1)
            self.gridLayout_2.addWidget(self.spectra_list[i], i + 1, 2, 1, 1)

        self.params_list = []
        self.lower_bound_list = []
        self.value_list = []
        self.upper_bound_list = []
        self.fixed_list = []
        self.error_list = []

        for i in range(self.max_params):
            self.params_list.append(QLabel())
            self.lower_bound_list.append(QLineEdit())
            self.value_list.append(QLineEdit())
            self.upper_bound_list.append(QLineEdit())
            self.fixed_list.append(QCheckBox())
            self.error_list.append(QLineEdit())

            self.gridLayout.addWidget(self.params_list[i], i + 1, 0, 1, 1)
            self.gridLayout.addWidget(self.lower_bound_list[i], i + 1, 1, 1, 1)
            self.gridLayout.addWidget(self.value_list[i], i + 1, 2, 1, 1)
            self.gridLayout.addWidget(self.upper_bound_list[i], i + 1, 3, 1, 1)
            self.gridLayout.addWidget(self.fixed_list[i], i + 1, 4, 1, 1)
            self.gridLayout.addWidget(self.error_list[i], i + 1, 5, 1, 1)

            self.fixed_list[i].stateChanged.connect(self.fixed_checked_changed)

        self.model_changed()

        self.accepted = False
        FitDialog.is_opened = True
        FitDialog._instance = self

        self.show()

        self.exec()

    @staticmethod
    def get_instance():
        return FitDialog._instance

    @staticmethod
    def check_state(checked):
        return Qt.Checked if checked else 0

    def correlation_analysis(self):
        pass

    def show_last_result(self):
        try:
            self.LFP_matrix.plot_figures_one()
        except:
            pass

    def model_changed(self):
        index = self.cbModel.currentIndex()
        self.current_model = self.models[index]()
        params_count = self.current_model.params.__len__()
        species_count = self.current_model.n_full

        for i in range(self.max_species):
            visible = species_count > i

            self.species_list[i].setVisible(visible)
            self.visible_list[i].setVisible(visible)
            self.spectra_list[i].setVisible(visible)

        for i in range(species_count):
            self.species_list[i].setText(self.current_model.species_names[i])
            self.visible_list[i].setChecked(self.current_model.visible[i])

        for i in range(self.max_params):
            visible = params_count > i

            self.params_list[i].setVisible(visible)
            self.lower_bound_list[i].setVisible(visible)
            self.value_list[i].setVisible(visible)
            self.upper_bound_list[i].setVisible(visible)
            self.fixed_list[i].setVisible(visible)
            self.error_list[i].setVisible(visible)

        for i, p in enumerate(self.current_model.params.values()):
            self.params_list[i].setText(p.name)
            self.lower_bound_list[i].setText(str(p.min))
            self.upper_bound_list[i].setText(str(p.max))
            self.value_list[i].setText(str(p.value))
            self.fixed_list[i].setChecked(not p.vary)

    def fit(self):
        try:
            self._fit()
        except Exception as e:
            Logger.status_message(e.__str__())
            QMessageBox.warning(self, 'fitting Error', e.__str__(), QMessageBox.Ok)

    def _fit(self):

        # check whether at least one visible checkbox is checked, if not, check first one
        checksum = 0
        for i in range(self.current_model.n_full):
            checksum += int(self.visible_list[i].isChecked())
        if checksum == 0:
            self.visible_list[0].setChecked(True)

        # get params from field to current model
        for i in range(self.current_model.params.__len__()):
            param = self.params_list[i].text()
            self.current_model.params[param].value = float(self.value_list[i].text())
            self.current_model.params[param].min = float(self.lower_bound_list[i].text())
            self.current_model.params[param].max = float(self.upper_bound_list[i].text())
            self.current_model.params[param].vary = not self.fixed_list[i].isChecked()

        visible_species = []
        for i in range(self.current_model.n_full):
            visible_species.append(self.visible_list[i].isChecked())

        # visible is a property and has to be assigned this way (not element-wise)
        self.current_model.visible = visible_species
        # get method (mimizing algortihm) for this fit
        method = self.methods[self.cbMethod.currentIndex()]['abbr']

        global_fit(self.current_model, verbose=self.cbDisplaResult.isChecked(), method=method)

        # update current model with fitted parameters
        self.current_model = self.LFP_matrix.model

        # update fields with new values (parameter values and errors)
        for i in range(self.current_model.params.__len__()):
            param = self.params_list[i].text()

            self.value_list[i].setText("{:.4g}".format(self.current_model.params[param].value))
            error = self.current_model.params[param].stderr
            self.error_list[i].setText("{:.4g}".format(error) if error is not None else '')

    def fixed_checked_changed(self, value):
        checkbox = self.sender()

        i = 0
        for i, ch in enumerate(self.fixed_list):
            if ch == checkbox:
                break

        self.lower_bound_list[i].setEnabled(True if value == Qt.Unchecked else False)
        self.upper_bound_list[i].setEnabled(True if value == Qt.Unchecked else False)

    def accept(self):

        self.accepted = True
        FitDialog.is_opened = False
        FitDialog._instance = None
        super(FitDialog, self).accept()

    def reject(self):
        FitDialog.is_opened = False
        FitDialog._instance = None
        super(FitDialog, self).reject()


if __name__ == "__main__":
    import sys


    def my_exception_hook(exctype, value, traceback):
        # Print the error and traceback
        print(exctype, value, traceback)
        # Call the normal Exception hook after
        sys._excepthook(exctype, value, traceback)
        sys.exit(1)


    from PyQt5.QtWidgets import QApplication

    sys._excepthook = sys.excepthook

    # Set the exception hook to our wrapping function
    sys.excepthook = my_exception_hook

    app = QtWidgets.QApplication(sys.argv)
    Dialog = FitDialog(None, None)
    # Dialog.show()
    sys.exit(app.exec_())
