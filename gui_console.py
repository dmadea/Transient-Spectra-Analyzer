from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5.QtWidgets import QDockWidget, QWidget

from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager

# great help from here https://stackoverflow.com/questions/11513132/embedding-ipython-qt-console-in-a-pyqt-application/12375397#12375397
# just copied :)
class ConsoleWidget(RichJupyterWidget):

    # https://github.com/ipython/ipykernel/issues/370
    # !!! required package ipykernel ver. 4.10, higher versions does not work, resp. after first unhandled exception occurs
    # other commands will not execute, but only error will be written

    def __init__(self, customBanner=None, *args, **kwargs):
        super(ConsoleWidget, self).__init__(*args, **kwargs)

        if customBanner is not None:
            self.banner = customBanner

        self.font_size = 6
        self.kernel_manager = kernel_manager = QtInProcessKernelManager()
        kernel_manager.start_kernel(show_banner=False)
        kernel_manager.kernel.gui = 'qt'
        self.kernel_client = kernel_client = self._kernel_manager.client()
        kernel_client.start_channels()

        def stop():
            kernel_client.stop_channels()
            kernel_manager.shutdown_kernel()
            guisupport.get_app_qt().exit()

        self.exit_requested.connect(stop)

    def set_focus(self):

        # TODO--- does not work...
        self._control.setFocus()

    def push_vars(self, variableDict):
        """
        Given a dictionary containing name / value pairs, push those variables
        to the Jupyter console widget
        """
        self.kernel_manager.kernel.shell.push(variableDict)

    def clear(self):
        """
        Clears the terminal
        """
        self._control.clear()

        # self.kernel_manager

    def print_text(self, text):
        """
        Prints some plain text to the console
        """
        # self._append_plain_text(text)
        self.append_stream(text)

    def print_html(self, text):
        self._append_html(text, True)

    def execute_command(self, command, hidden=True):
        """
        Execute a command in the frame of the console widget
        """
        self._execute(command, hidden)


class Console(QDockWidget):
    """
    Console window used for advanced commands, debugging,
    logging, and profiling.
    """

    _instance = None

    def __init__(self, parentWindow):
        QDockWidget.__init__(self, parentWindow)
        self.setTitleBarWidget(QWidget())
        self.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea)
        self.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.setVisible(False)

        Console._instance = self

        banner = """Simple Spectra Manipulator console based on IPython. Numpy package was imported as np. Three variables are setup:
    pw - Plot Widget
    sw - SVD widget - factor anaylsis
    fw - Fit widget
    f - fitter

Enjoy.
        
"""

        # Add console window
        self.console_widget = ConsoleWidget(banner)
        self.setWidget(self.console_widget)



        startup_comands = """
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from user_namespace import *
#from IPython.display import display, Math, Latex
from augmentedmatrix import AugmentedMatrix

_cdict = {'red':   ((0.0, 0.0, 0.0),
                   (2/5, 0.0, 0.0),
                   (1/2, 1.0, 1.0),
                   (3/5, 1.0, 1.0),
                   (4/5, 1.0, 1.0),
                   (1.0, 0.3, 0.3)),

         'green': ((0.0, 0, 0),
                   (2/5, 0.0, 0.0),
                   (1/2, 1.0, 1.0),
                   (3/5, 1.0, 1.0),
                   (4/5, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.3, 0.3),
                   (2/5, 1.0, 1.0),
                   (1/2, 1.0, 1.0),
                   (3/5, 0.0, 0.0),
                   (4/5, 0.0, 0.0),
                   (1.0, 0.0, 0.0))
        }

_ = matplotlib.colors.LinearSegmentedColormap('div', _cdict)
matplotlib.cm.register_cmap('div', _)
np.seterr(divide='ignore')

%matplotlib  # setup default backend

"""

        # execute first commands
        self.console_widget.execute_command(startup_comands)

    def setVisible(self, visible):
        """
        Override to set proper focus.
        """
        QDockWidget.setVisible(self, visible)
        if visible:
            self.console_widget.setFocus()

    @staticmethod
    def show_message(message):
        if Console._instance is not None:
            Console._instance.console_widget.print_text(message)

    def print_html(self, text):
        self.console_widget.print_html('\n' + text)

    @staticmethod
    def execute_command(cmd, hidden=True):
        if Console._instance is not None:
            Console._instance.console_widget.execute_command(cmd, hidden)

    @staticmethod
    def push_variables(variable_dict):
        if Console._instance is not None:
            Console._instance.console_widget.push_vars(variable_dict)





