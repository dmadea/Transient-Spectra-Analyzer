from PyQt5 import QtCore


class Task(QtCore.QThread):  # https://doc.qt.io/qt-5/qthread.html
    # messageAdded = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super(Task, self).__init__(parent)
        self.started.connect(self.preRun)
        self.finished.connect(self.postRun)
        self.isInterruptionRequested()

    def preRun(self):
        """
        The code in this method is runs before run in GUI thread.
        """
        pass

    def run(self):
        """
        The code in this method is run in another thread.
        """
        pass

    def postRun(self):
        """
        The code in this method is run in GUI thread.
        """
        pass

    # def postTerminated(self):
    #     """
    #     The code in this method is run in GUI thread.
    #     """
    #     pass
