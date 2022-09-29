#!python3.10


from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QCoreApplication
from main import fMain

import sys, os
if sys.executable.endswith("pythonw.exe"):
  sys.stdout = open(os.devnull, "w");
  sys.stderr = open(os.path.join(os.getenv("TEMP"), "stderr-"+os.path.basename(sys.argv[0])), "w")


application = QApplication(sys.argv)
QCoreApplication.setOrganizationName("Simple Spectra Manipulator")
QCoreApplication.setOrganizationDomain("")
QCoreApplication.setApplicationName("Simple Spectra Manipulator")
window = fMain()
window.show()
application.lastWindowClosed.connect(application.quit)
application.exec_()
