import os

import numpy as np
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt

from settings import Settings

from logger import Logger, Transcript

# from dialogs.settingsdialog import SettingsDialog
from Widgets.maindisplaydockarea import MainDisplayDockArea

# import code

from user_namespace import UserNamespace
from Widgets.fitwidget import FitWidget
from Widgets.svddockarea import SVDDockArea

from menubar import MenuBar
from gui_console import Console

# from dialogs.fitdialog import FitDialog

import lfp_parser
from LFP_matrix import LFP_matrix
import sys


class fMain(QMainWindow):

    def __init__(self, parent=None):
        super(fMain, self).__init__(parent)

        self.setWindowTitle("Transient Spectra Analyzer")

        self.resize(1800, 1000)

        self.console = Console(self)
        # self.matrix = None  # object of LFP matrix
        self.matrices = []  # loaded matrices of type LFP matrix

        self.addDockWidget(Qt.RightDockWidgetArea, self.console)
        # fixing the resize bug https://stackoverflow.com/questions/48119969/qdockwidget-splitter-jumps-when-qmainwindow-resized
        # self.resizeDocks([self.dockTreeWidget], [270], Qt.Horizontal)
        self.resizeDocks([self.console], [200], Qt.Horizontal)
        # self.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)

        self.main_display_widget = MainDisplayDockArea(parent=self)
        self.SVD_widget = SVDDockArea(self)
        self.fit_widget = FitWidget(None, self)

        self.tabWidget = QtWidgets.QTabWidget(self)
        self.tabWidget.addTab(self.main_display_widget, "Data")
        self.tabWidget.addTab(self.SVD_widget, "SVD + EFA")
        self.tabWidget.addTab(self.fit_widget, "Fit")

        self.tabWidget.currentChanged.connect(self.tabChanged)

        self.setCentralWidget(self.tabWidget)

        self.createStatusBar()
        self.logger = Logger(self.console.show_message, self.statusBar().showMessage)
        sys.stdout = Transcript()

        self.user_namespace = UserNamespace(self)

        self.menu_bar = MenuBar(self, common_dimension_triggered=self.main_display_widget.set_common_dim)
        self.setMenuBar(self.menu_bar)
        Settings.load()

        self.update_recent_files()

        Console.execute_command("from LFP_matrix import LFP_matrix\nimport fitmodels as m\n"
                                "import matplotlib.pyplot as plt\n"
                                "import augmentedmatrix")

        Console.push_variables({'main_widget': self})
        Console.push_variables({'pw': self.main_display_widget})
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
        # statusBar.addPermanentWidget(self.coor_label)
        statusBar.addPermanentWidget(console_button)

    def update_recent_files(self):
        num = min(len(Settings.recent_project_filepaths), len(self.menu_bar.recent_file_actions))

        for i in range(num):
            filepath = Settings.recent_project_filepaths[i]
            head, tail = os.path.split(filepath)
            text = os.path.split(head)[1] + '\\' + tail
            self.menu_bar.recent_file_actions[i].setText(text)
            self.menu_bar.recent_file_actions[i].setData(filepath)
            self.menu_bar.recent_file_actions[i].setVisible(True)

    def add_recent_file(self, filepath):
        # if there is the same filepath in the list, remove this entry
        if filepath in Settings.recent_project_filepaths:
            Settings.recent_project_filepaths.remove(filepath)

        Settings.recent_project_filepaths.insert(0, filepath)
        while len(Settings.recent_project_filepaths) > len(self.menu_bar.recent_file_actions):
            # remove last one
            Settings.recent_project_filepaths = Settings.recent_project_filepaths[:-1]
        Settings.save()
        self.update_recent_files()

    def test(self):

        # try:
        self.main_display_widget.save_plot_to_clipboard_as_png()
        # except Exception as ex:
        #     print(ex.__str__(), ex)

    def actioncopy_to_svg_clicked(self):

        self.main_display_widget.save_plot_to_clipboard_as_svg()

    def open_file(self, filepaths=False):
        ## strange bug, qt passes False as an argument when called from menubar

        # filter = "Data Files (*.txt, *.csv, *.dx)|*.txt;*.csv;*.dx|All Files (*.*)|*.*"
        filter = "Data Files (*.txt, *.csv, *.a*);;All Files (*.*)"
        initial_filter = "All Files (*.*)"

        if filepaths is False:
            filepaths = QFileDialog.getOpenFileNames(caption="Import files",
                                                     directory=Settings.import_files_dialog_path,
                                                     filter=filter, initialFilter=initial_filter)[0]

            if len(filepaths) == 0:
                return

        Settings.import_files_dialog_path = os.path.split(filepaths[0])[0]

        self.matrices = [lfp_parser.parse_file(filepath) for filepath in filepaths]

        self.plot_matrices(self.matrices)
        self.add_recent_file(filepaths[0])

    def plot_matrices(self, matrices=None):

        # if matrix is None or not isinstance(matrix, (LFP_matrix, str)):
        #     raise ValueError(f"matrix cannot be None or have to be type of {type(LFP_matrix)}")
        mats_to_plot = self.matrices if matrices is None else matrices

        self.main_display_widget.plot_matrices(mats_to_plot)
        self.menu_bar.set_common_dimension(self.main_display_widget.same_dimension)

        self.SVD_widget.set_data(mats_to_plot[0])

        Console.push_variables({'matrices': mats_to_plot})
        self.fit_widget.matrix = mats_to_plot[0]
        self.fit_widget.init_matrices()

        self.setWindowTitle(mats_to_plot[0].get_filename() + ' - Transient Spectra Analyzer')


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

