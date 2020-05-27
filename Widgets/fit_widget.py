from .Fit_gui import Ui_Form
from .fit_layout import FitLayout
from .fit_layout import HeatMapPlot

from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5.QtCore import Qt
from dialogs.gui_fit import Ui_Dialog
from PyQt5.QtWidgets import *

from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QMessageBox, QLineEdit, QCheckBox, QLabel, QWidget
import numpy as np

from logger import Logger

from user_namespace import global_fit

from plotwidget import PlotWidget
from LFP_matrix import LFP_matrix

import fitmodels
from fitting.fitter import Fitter
from .combo_box_cb import ComboBoxCB
from fitting.constraints import *
import pyqtgraph as pg

import sys, inspect

from fitting.fitresult import FitResult

from gui_console import Console
from fitting.constraints import ConstraintClosure


class FitWidget(QWidget, Ui_Form):
    max_params = 50
    max_species = 10

    instance = None

    def __init__(self, matrix=None, parent=None):
        super(FitWidget, self).__init__(parent)
        self.setupUi(self)

        FitWidget.instance = self

        # reference to matrix in parent - instance of LFP_matrix
        self.matrix = matrix
        # visible ST and C matrices - used as initial estimates for fitter
        self._ST = None
        self._C = None
        self.fit_result = None
        self.fitter = Fitter()

        self._au = None  # augmented matrix

        self.C_matrix_constraints = []

        Console.push_variables({'f': self.fitter})
        Console.push_variables({'fw': self})

        self.fit_plot_layout = FitLayout(None, self)
        self.main_layout.addWidget(self.fit_plot_layout)

        self.current_model = None

        # get all models from fitmodels, get classes that inherits from Model base class and sort them by name
        # and number of species
        classes = inspect.getmembers(sys.modules[fitmodels.__name__], inspect.isclass)
        tuples = filter(lambda tup: issubclass(tup[1], (fitmodels._Model)) and tup[1] is not (fitmodels._Model and fitmodels._Photokinetic_Model), classes)
        self.models = sorted(list(map(lambda tup: tup[1], tuples)), key=lambda cls: (cls.name, cls.n))

        # fill the combo box items with model names
        self.cbModel.addItems(map(lambda m: m.name, self.models))

        self.methods = [
            {'name': 'Levenberg–Marquardt', 'abbr': 'leastsq'},
            {'name': 'Trust-Region Reflective method', 'abbr': 'least_squares'},
            {'name': 'Nelder-Mead, Simplex method', 'abbr': 'nelder'},
            {'name': 'L-BFGS-B', 'abbr': 'lbfgsb'},
            {'name': 'Powell', 'abbr': 'powell'}
        ]

        self.regressors = [
            {'name': 'OLS (Ordinary Least Squares)', 'abbr': 'ols'},
            {'name': 'NNLS (Non-Negative Least Squares)', 'abbr': 'nnls'},
        ]

        # fill the combo box items with method names
        self.cbMethod.addItems(map(lambda m: m['name'], self.methods))
        self.cbMethod.setCurrentIndex(1)
        self.cbRegressorC.addItems(map(lambda m: m['name'], self.regressors))
        self.cbRegressorS.addItems(map(lambda m: m['name'], self.regressors))

        self.btnFit.clicked.connect(self.fit)
        self.cbModel.currentIndexChanged.connect(self.model_changed)
        self.sbN.valueChanged.connect(self.sbN_value_changed)
        self.sbN.setMaximum(self.max_species)
        self.btnC_calc.clicked.connect(self.calc_C)
        self.btnST_calc.clicked.connect(self.calc_ST)
        self.btnOneIt.clicked.connect(self.one_iter_clicked)
        self.btnSimulateModel.clicked.connect(self.simulate_model_clicked)

        # soft modeling

        self.species_label_list = []
        self.S_constrain_list = []
        self.ST_fix_list = []
        self.C_constrain_list = []
        self.C_conc_profile_list = []
        self.C_fix_list = []

        for i in range(self.max_species):
            self.species_label_list.append(QLabel(f'{i + 1}.'))
            self.S_constrain_list.append(ComboBoxCB())
            self.ST_fix_list.append(QCheckBox())
            self.C_constrain_list.append(ComboBoxCB())
            self.C_conc_profile_list.append(QComboBox())
            self.C_fix_list.append(QCheckBox())

            self.S_constrain_list[i].setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            self.C_constrain_list[i].setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            self.C_conc_profile_list[i].setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

            self.S_constrain_list[i].setMinimumSize(1, 1)
            self.C_constrain_list[i].setMinimumSize(1, 1)
            self.C_conc_profile_list[i].setMinimumSize(1, 1)

            # self.species_list[i].setAlignment(Qt.AlignLeft)
            # Qt.AlignCenter

            self.glSC_setup.addWidget(self.species_label_list[i], i + 1, 0, Qt.AlignLeft, 1)
            self.glSC_setup.addWidget(self.S_constrain_list[i], i + 1, 1, Qt.AlignLeft, 1)
            self.glSC_setup.addWidget(self.ST_fix_list[i], i + 1, 2, Qt.AlignLeft, 1)
            self.glSC_setup.addWidget(self.C_constrain_list[i], i + 1, 4, Qt.AlignLeft, 1)
            self.glSC_setup.addWidget(self.C_conc_profile_list[i], i + 1, 5, Qt.AlignLeft, 1)
            self.glSC_setup.addWidget(self.C_fix_list[i], i + 1, 6, Qt.AlignLeft, 1)

        # kinetic hard modeling

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
            #
            # self.lower_bound_list[i].setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            # self.value_list[i].setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            # self.upper_bound_list[i].setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            # self.error_list[i].setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            #
            # self.lower_bound_list[i].setMinimumSize(1, 1)
            # self.value_list[i].setMinimumSize(1, 1)
            # self.upper_bound_list[i].setMinimumSize(1, 1)
            # self.error_list[i].setMinimumSize(1, 1)

            self.glKinetics.addWidget(self.params_list[i], i + 1, 0, 1, 1)
            self.glKinetics.addWidget(self.lower_bound_list[i], i + 1, 1, 1, 1)
            self.glKinetics.addWidget(self.value_list[i], i + 1, 2, 1, 1)
            self.glKinetics.addWidget(self.upper_bound_list[i], i + 1, 3, 1, 1)
            self.glKinetics.addWidget(self.fixed_list[i], i + 1, 4, 1, 1)
            self.glKinetics.addWidget(self.error_list[i], i + 1, 5, 1, 1)

            self.fixed_list[i].stateChanged.connect(self.fixed_checked_changed)

        self.model_changed()
        self.sbN_value_changed()

    def set_ST_full(self, data):
        if self.matrix is None:
            return

        self._ST = data
        self.plot_opt_matrices()

    def set_ST(self, component, data):
        if self.matrix is None:
            return

        self._ST[component] = np.asarray(data)

        self.plot_opt_matrices()

    def ssq(self):
        return ((self.matrix.Y - self._C @ self._ST) ** 2).sum()

    def lof(self):
        return np.sqrt(self.ssq() / (self.matrix.Y ** 2).sum()) * 100

    def R2(self):
        return (1 - self.ssq() / (self.matrix.Y ** 2).sum()) * 100

    def print_stats(self):
        print(f"Lack of Fit:    {self.lof():.05g}")
        print(f"R squared:      {self.R2():.05g}")

    def set_C(self, component, data):
        if self.matrix is None:
            return

        self._C[:, component] = np.asarray(data)

        self.plot_opt_matrices()

    def set_C_noise(self, components=None, noise_amp=0.1):
        n = self._C.shape[1]

        # all
        if not components:
            self._C = np.random.random((self._C.shape[0], n)) * noise_amp
        else:
            for idx in components:
                self._C[:, idx] = np.random.random((self._C.shape[0])) * noise_amp

        self.plot_opt_matrices()

    def set_ST_noise(self, components=None, noise_amp=0.1):
        n = self._ST.shape[0]

        # all
        if not components:
            self._ST = np.random.random((n, self._ST.shape[1])) * noise_amp
        else:
            for idx in components:
                self._ST[idx] = np.random.random((self._ST.shape[1])) * noise_amp

        self.plot_opt_matrices()

    def init_matrices(self):
        if self.matrix is None:
            return

        species_count = int(self.sbN.value())

        # t = self.matrix.times.shape[0] if self._au is None else self._au.aug_mat.shape[0]
        # w = self.matrix.wavelengths.shape[0] if self._au is None else self._au.aug_mat.shape[1]
        t, w = self.matrix.times.shape[0], self.matrix.wavelengths.shape[0]

        # try:
        # if ST and C are not initialized or if number of dimensions changed, initialize with zeros
        if self._ST is None or self._ST.shape[1] != w or self._C.shape[0] != t:
            self._ST = np.zeros((species_count, w))
            self._C = np.zeros((t, species_count))
            return

        # crop matrices or v/h stack
        if species_count < self._ST.shape[0]:
            self._ST = self._ST[:species_count]
            self._C = self._C[:, :species_count]
        else:
            d = species_count - self._ST.shape[0]
            self._ST = np.vstack((self._ST, np.zeros((d, w))))
            self._C = np.hstack((self._C, np.zeros((t, d))))

        # except Exception as e:
        #     Logger.console_message(e.__str__())

    def sbN_value_changed(self):
        species_count = int(self.sbN.value())

        for i in range(self.max_species):
            visible = species_count > i

            self.species_label_list[i].setVisible(visible)
            self.S_constrain_list[i].setVisible(visible)
            self.ST_fix_list[i].setVisible(visible)
            self.C_constrain_list[i].setVisible(visible)
            self.C_conc_profile_list[i].setVisible(visible)
            self.C_fix_list[i].setVisible(visible)

        self.init_matrices()
        self.current_model.update_n(species_count)
        self.update_model_par_count()

    @staticmethod
    def check_state(checked):
        return Qt.Checked if checked else 0

    def model_changed(self):
        index = self.cbModel.currentIndex()
        self.current_model = self.models[index]()

        self.current_model.update_n(int(self.sbN.value()))
        self.update_model_par_count()

        if self.matrix is None:
            return

        # set spectra matrix to model as a parameter
        wls = self._au[0, 0].wavelengths if self._au else self.matrix.wavelengths
        setattr(self.current_model, 'wavelengths', wls)
        setattr(self.current_model, 'ST', self._ST)
        if self._au:
            setattr(self.current_model, 'aug_matrix', self._au)
            self.current_model.update_n()

    def update_model_par_count(self):
        params_count = self.current_model.params.__len__()

        species_count = self.current_model.n
        cb_fill_list = ['Unknown (MCR-ALS)'] + [self.current_model.species_names[i] for i in range(species_count)]

        for i in range(self.max_species):
            cur_idx = self.C_conc_profile_list[i].currentIndex()
            self.C_conc_profile_list[i].clear()
            self.C_conc_profile_list[i].addItems(cb_fill_list)
            self.C_conc_profile_list[i].setCurrentIndex(cur_idx if cur_idx >= 0 else 0)

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

    def simulate_model_clicked(self):
        if self.current_model is None:
            return

        self.update_params()
        self.fitter_update_options()

        if self.current_model.method == 'RFA':
            T = self.current_model.get_T()
            self._ST = T.dot(self.current_model.VT)

            self.current_model.ST = self._ST
            self._C = self.current_model.calc_C(C_out=self._C)
        else:
            self._C = self.current_model.calc_C(C_out=self._C)

            # self.fitter_update_options()
            self.fitter.calc_ST()
            # self.fitter.ST_opt[0] *= 26139.01 / self.fitter.ST_opt[0].max()
            self._ST = self.fitter.ST_opt

        self.plot_opt_matrices()


        #
        # t_diff = np.abs(self.matrix.times[1:] - self.matrix.times[:-1])
        #
        # min_diff = t_diff.min()
        #
        # t_points = (self.matrix.times[-1] - self.matrix.times[0]) / min_diff + 1
        # new_t = np.linspace(self.matrix.times[0], self.matrix.times[-1], np.ceil(t_points))
        #
        # self.current_model.init_times(new_t)

        # n = C.shape[1]
        #
        # self.fit_plot_layout.C_plot.clear()
        # self.fit_plot_layout.ST_plot.clear()
        #
        # self.fit_plot_layout.add_legend(spacing=13)
        #
        # for i in range(n):
        #     pen = pg.mkPen(color=FitLayout.int_default_color(i), width=1.5, style=Qt.DashLine)
        #     self.fit_plot_layout.C_plot.plot(new_t, C[:, i], pen=pen,
        #                                      name=self.current_model.species_names[i])

        # self.matrix.times_fine = new_t
        # self.matrix.C_fine = C

    def calc_T(self):
        if hasattr(self.current_model, 'T'):
            # D = U S V^T
            # C = US T^-1, S^T = T V^T
            V = self.current_model.VT.T
            T = self._ST @ V

            self.current_model.update_T(T)
            self.update_model_par_count()

    def calc_ST(self):
        if self.matrix is None:
            return

        try:
            self.fitter_update_options()
            self.fitter.calc_ST()
            self._ST = self.fitter.ST_opt
            self.plot_opt_matrices()

            self.calc_T()

        except Exception as e:
            Logger.console_message(e.__str__())

    def calc_C(self):
        if self.matrix is None:
            return

        try:
            self.fitter_update_options()
            self.fitter.calc_C()
            self._C = self.fitter.C_opt
            self.plot_opt_matrices()

            self.calc_T()

        except Exception as e:
            Logger.console_message(e.__str__())

    def one_iter_clicked(self):
        self._fit(max_iter=1)

    def fit(self):
        # try:
        self._fit()
        # except Exception as e:
        #     Logger.status_message(e.__str__())
        #     QMessageBox.warning(self, 'fitting Error', e.__str__(), QMessageBox.Ok)

    def _fit(self, max_iter=None, st_constraints=None, c_constraints=None):

        self.fitter_update_options(max_iter=max_iter, st_constraints=st_constraints, c_constraints=c_constraints)

        n = int(self.sbN.value())  # number of species

        # pure H-fit
        # if self.current_model.connectivity.count(0) == 0:
        #     if not self.fitter.st_fix:
        #         self.fitter.var_pro(C_est=self._C, c_fix=self.fitter.c_fix)
        #     else:
        #         self.fitter.H_fit(self.current_model, self._ST, self._C, self.fitter.st_fix, self.fitter.c_fix)
        #
        #     self.current_model = self.fitter.c_model
        # elif self.current_model.connectivity.count(0) == n:  # pure MCR fit
        #     self.fitter.HS_MCR_fit(c_model=None)
        # else:  # mix of two, HS-fit
        if self.current_model.connectivity.count(0) == 0:
            # self.fitter.HS_MCR_fit(c_model=self.current_model)
            # self.fitter.var_pro()
            if self.current_model.method is 'RFA':
                self.fitter.obj_func_fit()
            else:
                self.fitter.HS_MCR_fit(c_model=self.current_model)
            self.current_model = self.fitter.c_model
        elif self.current_model.connectivity.count(0) == n:  # pure MCR fit
            self.fitter.HS_MCR_fit(c_model=None)
        else:  # mix of two, HS-fit
            self.fitter.HS_MCR_fit(c_model=self.current_model)
            self.current_model = self.fitter.c_model

        self.update_fields_H_fit()

        self._C = self.fitter.C_opt
        self._ST = self.fitter.ST_opt

        self.plot_opt_matrices()

    def plot_opt_matrices(self):
        if self.matrix is None:
            return

        n = int(self.sbN.value())

        D_fit = np.dot(self._C, self._ST)
        R = D_fit - self.matrix.Y

        # self.matrix.E = R
        self.matrix.C_fit = self._C
        self.matrix.ST_fit = self._ST

        if self._au:
            self._au.ST_aug = self._ST
            self._au.C_aug = self._C

        self.fit_plot_layout.C_plot.clear()
        self.fit_plot_layout.ST_plot.clear()

        self.fit_plot_layout.add_legend(spacing=13)

        for i in range(n):
            pen_fit = pg.mkPen(color=FitLayout.int_default_color(i), width=1)
            self.fit_plot_layout.C_plot.plot(self.matrix.times, self._C[:, i], pen=pen_fit,
                                             name=f"Component {i + 1}")
            self.fit_plot_layout.ST_plot.plot(self.matrix.wavelengths, self._ST[i], pen=pen_fit,
                                              name=f"Component {i + 1}")

        self.fit_plot_layout.heat_map_plot.set_matrix(R, self.matrix.times, self.matrix.wavelengths,
                                                      gradient=HeatMapPlot.seismic)

        mat = LFP_matrix.from_value_matrix(D_fit, self.matrix.times, self.matrix.wavelengths)
        PlotWidget.instance.set_fit_matrix(mat)

    # def set_Closure_constraint(self, set=True, value=1, hard_closure=True):
    #     if set:
    #         self.C_matrix_constraints = [ConstraintClosure(hard_closure=hard_closure, value=value)]
    #     else:
    #         self.C_matrix_constraints = []

    def fitter_update_options(self, max_iter=None, st_constraints=None, c_constraints=None):
        if self.matrix is None:
            return

        # t = self._au[2, 0].times if self._au else self.matrix.times

        self.current_model.init_times(self.matrix.times)
        self.update_params()

        # set spectra matrix to model as a parameter
        wls = self._au[0, 0].wavelengths if self._au else self.matrix.wavelengths
        setattr(self.current_model, 'wavelengths', wls)
        setattr(self.current_model, 'ST', self._ST)
        if self._au:
            setattr(self.current_model, 'aug_matrix', self._au)
        else:
            setattr(self.current_model, 'matrix', self.matrix)

        # get the conectivity with kinetic model
        n = int(self.sbN.value())  # number of species
        max_iter = int(self.sb_max_iter.value()) if max_iter is None else max_iter
        # first index - 0, no kinetic constraints, others - kinetic species
        connectivity = [self.C_conc_profile_list[i].currentIndex() for i in range(n)]

        # update connectivity
        self.current_model.connectivity = connectivity
        method = self.methods[self.cbMethod.currentIndex()]['abbr']

        # constraints
        _st_constraints = [self.S_constrain_list[i].constraints for i in range(n)]
        _c_constraints = [self.C_constrain_list[i].constraints for i in range(n)]

        c_constraints = c_constraints if c_constraints is not None else _c_constraints
        st_constraints = st_constraints if st_constraints is not None else _st_constraints

        # indexes of fixed spectra and concentrations
        c_fix = []
        st_fix = []

        for i in range(n):
            if self.C_fix_list[i].isChecked(): c_fix.append(i)
            if self.ST_fix_list[i].isChecked(): st_fix.append(i)

        c_fix = c_fix if len(c_fix) > 0 else None
        st_fix = st_fix if len(st_fix) > 0 else None

        C_matrix_constraints = None
        if self.cbClosureConstraintC.isChecked():
            C_matrix_constraints = [ConstraintClosure(hard_closure=True, value=float(self.txbClosureConstrValue.text()))]

        self.fitter.update_options(D=self.matrix.Y, times=self.matrix.times, wls=self.matrix.wavelengths,
                                   c_model=self.current_model, n=n, max_iter=max_iter,
                                   C_est=self._C, ST_est=self._ST,
                                   c_constraints=c_constraints, st_constraints=st_constraints,
                                   c_fix=c_fix, st_fix=st_fix, fit_alg=method,
                                   C_matrix_constraints=C_matrix_constraints,
                                   C_regressor=self.regressors[self.cbRegressorC.currentIndex()]['abbr'],
                                   S_regressor=self.regressors[self.cbRegressorS.currentIndex()]['abbr'],
                                   au=self._au)

    def update_params(self):

        if self.current_model is None:
            return

        # get params from field to current model
        for i in range(self.current_model.params.__len__()):
            param = self.params_list[i].text()

            val = float(self.value_list[i].text())
            min, max = float(self.lower_bound_list[i].text()), float(self.upper_bound_list[i].text())
            val = val if min <= val <= max else (min if np.abs(min - val) < np.abs(max - val) else max)

            self.current_model.params[param].min = min
            self.current_model.params[param].max = max
            self.current_model.params[param].value = val
            self.current_model.params[param].vary = not self.fixed_list[i].isChecked()

    def update_fields_H_fit(self):

        values_errors = np.zeros((self.current_model.params.__len__(), 2), dtype=np.float32)

        for i in range(self.current_model.params.__len__()):
            param = self.params_list[i].text()
            self.value_list[i].setText("{:.4g}".format(self.current_model.params[param].value))
            error = self.current_model.params[param].stderr
            self.error_list[i].setText("{:.4g}".format(error) if error is not None else '')

            values_errors[i, 0] = self.current_model.params[param].value
            values_errors[i, 1] = error if error is not None else 0

        if self.fitter.minimizer is not None:
            self.fit_result = FitResult(self.fitter.last_result, self.fitter.minimizer, values_errors,
                                        self.current_model)
            Console.push_variables({'fit': self.fit_result})

    def fixed_checked_changed(self, value):
        checkbox = self.sender()

        i = 0
        for i, ch in enumerate(self.fixed_list):
            if ch == checkbox:
                break

        self.lower_bound_list[i].setEnabled(True if value == Qt.Unchecked else False)
        self.upper_bound_list[i].setEnabled(True if value == Qt.Unchecked else False)


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
    Dialog = FitWidget(None, None)
    # Dialog.show()
    sys.exit(app.exec_())
