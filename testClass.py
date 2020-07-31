import sys, os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *

from PyQt5.QtGui import QColor

import pyqtgraph as pg


from settings import Settings
from logger import Logger

from dialogs.settingsdialog import SettingsDialog
from plotwidget import PlotWidget


# import code

from user_namespace import UserNamespace
from Widgets.fit_widget import FitWidget
from Widgets.svd_widget import SVDWidget

from menubar import MenuBar
from gui_console import Console

from dialogs.fitdialog import FitDialog
from pyqtgraph.dockarea import *
import Widgets.Fit_gui

from scipy.linalg import svd, svdvals, diagsvd
import numpy as np


import lfp_parser
from LFP_matrix import LFP_matrix


class fMain(QMainWindow):
    # resized = QtCore.pyqtSignal()
    #
    # mooving = False  # boolean value indicates whenever we are selecting the zooming region
    # # x0, y0, x1, y1 = 0, 0, 0, 0

    def __init__(self, parent=None):
        super(fMain, self).__init__(parent)

        self.setWindowTitle("Transient Spectra Analyzer")

        self.resize(1800, 1000)

        self.console = Console(self)

        self.matrix = None  # object of LFP matrix


        # self.dockTreeWidget = QtWidgets.QDockWidget(self)
        # self.dockTreeWidget.setTitleBarWidget(QWidget())
        # self.dockTreeWidget.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        # self.dockTreeWidget.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea)

        # self.treeWidget = TreeWidget(self.dockTreeWidget)
        # self.dockTreeWidget.setWidget(self.treeWidget)

        # self.addDockWidget(Qt.LeftDockWidgetArea, self.dockTreeWidget)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.console)
        # fixing the resize bug https://stackoverflow.com/questions/48119969/qdockwidget-splitter-jumps-when-qmainwindow-resized
        # self.resizeDocks([self.dockTreeWidget], [270], Qt.Horizontal)
        self.resizeDocks([self.console], [200], Qt.Vertical)
        # self.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)

        self.coor_label = QLabel()
        self.grpView = PlotWidget(set_coordinate_func=self.coor_label.setText, parent=self)

        self.fit_widget = FitWidget(None, self)

        self.tabWidget = QtWidgets.QTabWidget(self)
        # self.tData = QtWidgets.QWidget()
        # self.tData.setObjectName("tData")

        self.tabWidget.addTab(self.grpView, "Data")
        #
        # self.area = DockArea()
        #
        # d1 = Dock("Dock1", size=(1, 1))  ## give this dock the minimum possible size
        # d2 = Dock("Dock2 - Console", size=(1, 1), closable=True)
        # # d3 = Dock("Dock3", size=(500, 400))
        # # d4 = Dock("Dock4 (tabbed) - Plot", size=(500, 200))
        # # d5 = Dock("Dock5 - Image", size=(500, 200))
        # # d6 = Dock("Dock6 (tabbed) - Plot", size=(500, 200))
        # self.area.addDock(d1,      'left')
        # self.area.addDock(d2, 'right')
        #
        # w1 = pg.PlotWidget(title="Plot inside dock with no title bar")
        # w1.plot(np.random.normal(size=100))
        # d1.addWidget(w1)
        #
        # w2 = pg.PlotWidget(title="Dock 4 plot")
        # w2.plot([])
        # d2.addWidget(w2)
        #
        #
        # self.tabWidget.addTab(self.area, "Data")

        self.tSVD = SVDWidget(self)
        self.tabWidget.addTab(self.tSVD, "SVD")

        # self.l = QVBoxLayout()
        #
        # model = QStandardItemModel(3, 1)
        #
        # for i in range(3):
        #     item = QStandardItem(f"Item {i}")
        #
        #     item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        #     item.setData(Qt.Unchecked, Qt.CheckStateRole)
        #
        #     model.setItem(i, 0, item)
        #
        #     # model.dataChanged.connect()
        #
        # combo = QComboBox()
        # combo.setModel(model)
        # combo.setCurrentText('asdda')
        #
        # self.l.addWidget(combo)
        # self.tSVD.setLayout(self.l)


        # self.tFit = QtWidgets.QWidget()
        self.tabWidget.addTab(self.fit_widget, "Fit")

        self.setCentralWidget(self.tabWidget)

        self.createStatusBar()
        self.logger = Logger(self.console.show_message, self.statusBar().showMessage)

        # self.treeWidget.itemCheckStateChanged.connect(self.item_checked_changed)
        # self.treeWidget.itemEdited.connect(self.item_edited)
        # self.treeWidget.itemsDeleted.connect(self.items_deleted)
        # self.treeWidget.redraw_spectra.connect(self.redraw_all_spectra)

        self.user_namespace = UserNamespace(self)

        self.setMenuBar(MenuBar(self))

        Settings.load()

        self.update_recent_files()

        Console.push_variables({'main_widget': self})
        # Console.push_variables({'plot_widget': self.grpView})

        Console.execute_command("from LFP_matrix import LFP_matrix\nimport fitmodels as m\n"
                                "import matplotlib.pyplot as plt\nfrom Widgets.fit_widget import FitWidget\n"
                                "import augmentedmatrix")

        Console.push_variables({'pw': self.grpView})



    def createStatusBar(self):
        statusBar = QStatusBar()
        self.setStatusBar(statusBar)
        statusBar.showMessage("Ready", 3000)
        console_button = QPushButton("Console")
        console_button.setFlat(True)
        console_button.setCheckable(True)
        console_button.toggled.connect(self.console.setVisible)
        statusBar.addPermanentWidget(self.coor_label)
        statusBar.addPermanentWidget(console_button)


    def perform_SVD(self):

        Console.show_message("Performing singular value decomposition")

        self.matrix.SVD()

        Console.push_variables({'U': self.matrix.U, 'S': self.matrix.S, 'V_T': self.matrix.V_T})


    def update_recent_files(self):
        num = min(len(Settings.recent_project_filepaths), len(self.menuBar().recent_file_actions))

        for i in range(num):
            filepath = Settings.recent_project_filepaths[i]
            tail = os.path.split(filepath)[1]
            text = os.path.splitext(tail)[0]
            self.menuBar().recent_file_actions[i].setText(text)
            self.menuBar().recent_file_actions[i].setData(filepath)
            self.menuBar().recent_file_actions[i].setVisible(True)

    def add_recent_file(self, filepath):
        # if there is the same filepath in the list, remove this entry
        if filepath in Settings.recent_project_filepaths:
            Settings.recent_project_filepaths.remove(filepath)

        Settings.recent_project_filepaths.insert(0, filepath)
        while len(Settings.recent_project_filepaths) > len(self.menuBar().recent_file_actions):
            # remove last one
            Settings.recent_project_filepaths = Settings.recent_project_filepaths[:-1]
        Settings.save()
        self.update_recent_files()

    def test(self):

        # try:
        self.grpView.save_plot_to_clipboard_as_png()
        # except Exception as ex:
        #     print(ex.__str__(), ex)

    def actioncopy_to_svg_clicked(self):

        self.grpView.save_plot_to_clipboard_as_svg()

    #
    # def export_selected_spectra_as(self):
    #
    #     if ExportSpectraAsDialog.is_opened:
    #         ExportSpectraAsDialog.get_instance().activateWindow()
    #         ExportSpectraAsDialog.get_instance().setFocus()
    #         return
    #
    #     if len(self.treeWidget.selectedIndexes()) == 0:
    #         return
    #
    #     dialog = ExportSpectraAsDialog(Settings.export_spectra_as_dialog_path, Settings.export_spectra_as_dialog_ext)
    #
    #     if not dialog.accepted:
    #         return
    #
    #     path, ext = dialog.result
    #
    #
    #
    #     sp_list = self.treeWidget.get_hierarchic_list(
    #         self.treeWidget.myModel.iterate_selected_items(skip_groups=True,
    #                                                        skip_childs_in_selected_groups=False))
    #
    #     try:
    #         if ext == '.csv':
    #             Spectrum.list_to_string(sp_list, include_group_name=Settings.files_exp_include_group_name,
    #                                     include_header=Settings.files_exp_include_header,
    #                                     delimiter=Settings.csv_exp_delimiter,
    #                                     decimal_sep=Settings.csv_exp_decimal_sep, save_to_file=True,
    #                                     dir_path=path, extension=ext)
    #         else:
    #             Spectrum.list_to_string(sp_list, include_group_name=Settings.files_exp_include_group_name,
    #                                     include_header=Settings.files_exp_include_header,
    #                                     delimiter=Settings.general_exp_delimiter,
    #                                     decimal_sep=Settings.general_exp_decimal_sep, save_to_file=True,
    #                                     dir_path=path, extension=ext)
    #     except Exception as ex:
    #         QMessageBox.warning(self, 'Error', ex.__str__(), QMessageBox.Ok)
    #
    #     Settings.export_spectra_as_dialog_path = path
    #     Settings.export_spectra_as_dialog_ext = ext

    def open_settings(self):

        if SettingsDialog.is_opened:
            SettingsDialog.get_instance().activateWindow()
            SettingsDialog.get_instance().setFocus()
            return

        sett_dialog = SettingsDialog()

        if not sett_dialog.accepted:
            return

        self.grpView.update_settings()

        self.redraw_all_spectra()

    # def open_project(self, filepath=None, open_dialog=True):
    #
    #     if open_dialog:
    #         # filter = "Data Files (*.txt, *.csv, *.dx)|*.txt;*.csv;*.dx|All Files (*.*)|*.*"
    #         filter = "Project files (*.smpj);;All Files (*.*)"
    #         initial_filter = "Project files (*.smpj)"
    #
    #         filepaths = QFileDialog.getOpenFileName(caption="Open project",
    #                                                 directory=Settings.open_project_dialog_path,
    #                                                 filter=filter,
    #                                                 initialFilter=initial_filter)
    #         if filepaths[0] == '':
    #             return
    #
    #         Settings.open_project_dialog_path = os.path.split(filepaths[0])[0]
    #         filepath = filepaths[0]
    #
    #     try:
    #         project = Project.deserialize(filepath)
    #     except:
    #         return
    #
    #     if self.treeWidget.top_level_items_count() != 0:
    #         reply = QMessageBox.question(self, 'Open project', "Do you want to merge the project with current project? "
    #                                                            "By clicking No, current project will be deleted and replaced.",
    #                                      QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
    #         if reply == QMessageBox.Yes:
    #             pass
    #         elif reply == QMessageBox.No:
    #             # delete all spectra and import new
    #             self.treeWidget.clear()
    #             project.settings.set_settings()
    #         else:
    #             return
    #
    #     self.treeWidget.import_spectra(project.spectra_list)
    #     self.add_recent_file(filepath)
    #
    #     # # load file specific user settings
    #     # project.settings.set_settings()
    #     #
    #     # # delete all spectra and import new
    #     # self.treeWidget.clear()
    #     # self.treeWidget.import_spectra(project.spectra_list)

    # def save_project(self):
    #
    #
    #     # filter = "Data Files (*.txt, *.csv, *.dx)|*.txt;*.csv;*.dx|All Files (*.*)|*.*"
    #     filter = "Project files (*.smpj)"
    #
    #     filepath = QFileDialog.getSaveFileName(caption="Save project",
    #                                            directory=Settings.save_project_dialog_path,
    #                                            filter=filter, initialFilter=filter)
    #
    #     if filepath[0] == '':
    #         return
    #
    #     Logger.message("Saving project to {}".format(filepath[0]))
    #
    #     Settings.save_project_dialog_path = os.path.split(filepath[0])[0]
    #
    #     sp_list = self.treeWidget.get_hierarchic_list(
    #         self.treeWidget.myModel.iterate_items(ItemIterator.NoChildren))
    #
    #     project = Project(sp_list)
    #     project.serialize(filepath[0])
    #
    #     self.add_recent_file(filepath[0])
        Logger.message("Done")

    def open_fit_dialog(self):

        if self.matrix is not None:
            d = FitDialog(self.matrix)



    def file_menu_import_files(self):

        # filter = "Data Files (*.txt, *.csv, *.dx)|*.txt;*.csv;*.dx|All Files (*.*)|*.*"
        filter = "Data Files (*.txt, *.csv, *.dx);;All Files (*.*)"
        initial_filter = "All Files (*.*)"

        filename = QFileDialog.getOpenFileName(caption="Import files",
                                                 directory=Settings.import_files_dialog_path,
                                                 filter=filter, initialFilter=initial_filter)

        if len(filename[0]) == 0:
            return

        Settings.import_files_dialog_path = os.path.split(filename[0])[0]

        Settings.save()

        path_to_file = filename[0]

        self.matrix = lfp_parser.parse_file(path_to_file)

        self.grpView.plot_matrix(self.matrix)
        self.tSVD.set_data(self.matrix)

        tail = os.path.split(path_to_file)[1]
        name_of_file = os.path.splitext(tail)[0]  # without extension

        self.setWindowTitle(name_of_file + ' - Transient Spectra Analyzer')

        Console.push_variables({'matrix': self.matrix})
        self.fit_widget.matrix = self.matrix
        self.fit_widget.init_matrices()


def my_exception_hook(exctype, value, traceback):
    # Print the error and traceback
    print(exctype, value, traceback)
    # Call the normal Exception hook after
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)


if __name__ == "__main__":
    # Back up the reference to the exceptionhook
    sys._excepthook = sys.excepthook

    # Set the exception hook to our wrapping function
    sys.excepthook = my_exception_hook

    app = QtWidgets.QApplication(sys.argv)
    # app.setStyle('Windows')
    form = fMain()
    form.show()

    # form.interact()

    sys.exit(app.exec_())

# print(QtWidgets.QStyleFactory.keys())
