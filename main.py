import sys, os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *

from PyQt5.QtGui import QColor
import pyqtgraph as pg

from settings import Settings

from logger import Logger, Transcript

# from dialogs.settingsdialog import SettingsDialog
from plotwidget import PlotWidget

# import code

from user_namespace import UserNamespace
from Widgets.fit_widget import FitWidget
from Widgets.svd_widget import SVDWidget

from menubar import MenuBar
from gui_console import Console

# from dialogs.fitdialog import FitDialog
from pyqtgraph.dockarea import *
import Widgets.Fit_gui

import lfp_parser
from LFP_matrix import LFP_matrix
import sys


class fMain(QMainWindow):

    def __init__(self, parent=None):
        super(fMain, self).__init__(parent)

        self.setWindowTitle("Transient Spectra Analyzer")

        self.resize(1800, 1000)

        self.console = Console(self)
        self.matrix = None  # object of LFP matrix

        self.addDockWidget(Qt.RightDockWidgetArea, self.console)
        # fixing the resize bug https://stackoverflow.com/questions/48119969/qdockwidget-splitter-jumps-when-qmainwindow-resized
        # self.resizeDocks([self.dockTreeWidget], [270], Qt.Horizontal)
        self.resizeDocks([self.console], [200], Qt.Horizontal)
        # self.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)

        self.coor_label = QLabel()  # coordinates

        self.plot_widget = PlotWidget(set_coordinate_func=self.coor_label.setText, parent=self)
        self.SVD_widget = SVDWidget(self)
        self.fit_widget = FitWidget(None, self)

        self.tabWidget = QtWidgets.QTabWidget(self)
        self.tabWidget.addTab(self.plot_widget, "Data")
        self.tabWidget.addTab(self.SVD_widget, "SVD + EFA")
        self.tabWidget.addTab(self.fit_widget, "Fit")

        self.tabWidget.currentChanged.connect(self.tabChanged)

        self.setCentralWidget(self.tabWidget)

        self.createStatusBar()
        self.logger = Logger(self.console.show_message, self.statusBar().showMessage)
        sys.stdout = Transcript()

        self.user_namespace = UserNamespace(self)

        self.setMenuBar(MenuBar(self))
        Settings.load()

        self.update_recent_files()

        Console.push_variables({'main_widget': self})

        Console.execute_command("from LFP_matrix import LFP_matrix\nimport fitmodels as m\n"
                                "import matplotlib.pyplot as plt\nfrom Widgets.fit_widget import FitWidget\n"
                                "import augmentedmatrix")

        Console.push_variables({'pw': self.plot_widget})
        Console.push_variables({'fw': self.fit_widget})
        Console.push_variables({'sw': self.SVD_widget})

    def tabChanged(self):
        if self.tabWidget.currentIndex() == 1 and self.SVD_widget.data_panel.cb_SVD.isChecked():
            self.SVD_widget.SVD_from_selection_toggled()  # recalculate SVD from selection

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

    def update_recent_files(self):
        num = min(len(Settings.recent_project_filepaths), len(self.menuBar().recent_file_actions))

        for i in range(num):
            filepath = Settings.recent_project_filepaths[i]
            head, tail = os.path.split(filepath)
            text = os.path.split(head)[1] + '\\' + tail
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
        self.plot_widget.save_plot_to_clipboard_as_png()
        # except Exception as ex:
        #     print(ex.__str__(), ex)

    def actioncopy_to_svg_clicked(self):

        self.plot_widget.save_plot_to_clipboard_as_svg()

    def open_file(self, filepath=False):
        ## strange bug, qt passes False as an argument when called from menubar

        # filter = "Data Files (*.txt, *.csv, *.dx)|*.txt;*.csv;*.dx|All Files (*.*)|*.*"
        filter = "Data Files (*.txt, *.csv, *.a*);;All Files (*.*)"
        initial_filter = "All Files (*.*)"

        if filepath is False:
            filepath = QFileDialog.getOpenFileName(caption="Import files",
                                                   directory=Settings.import_files_dialog_path,
                                                   filter=filter, initialFilter=initial_filter)[0]

            if len(filepath) == 0:
                return

        Settings.import_files_dialog_path = os.path.split(filepath)[0]

        matrix = lfp_parser.parse_file(filepath)

        self.setup_matrix(matrix)
        self.add_recent_file(filepath)

    def setup_matrix(self, matrix, *args, **kwargs):

        if matrix is None or not isinstance(matrix, (LFP_matrix, str)):
            raise ValueError(f"matrix cannot be None or have to be type of {type(LFP_matrix)}")

        if isinstance(matrix, str):
            matrix = lfp_parser.parse_file(matrix)

        self.matrix = matrix

        self.plot_widget.plot_matrix(self.matrix, **kwargs)
        self.SVD_widget.set_data(self.matrix)

        Console.push_variables({'matrix': self.matrix})
        self.fit_widget.matrix = self.matrix
        self.fit_widget.init_matrices()

        tail = os.path.split(matrix.filename)[1]
        name_of_file = os.path.splitext(tail)[0]  # without extension

        self.setWindowTitle(name_of_file + ' - Transient Spectra Analyzer')


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

