# from PyQt5.QtCore import *
# from PyQt5.QtWidgets import *
#
# from PyQt5.QtGui import *
#
# from matplotlib.figure import Figure
# from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
#
#
# class MathTextLabel(QWidget):
#
#     def __init__(self, mathText, parent=None, **kwargs):
#         QWidget.__init__(self, parent, **kwargs)
#
#         l = QVBoxLayout(self)
#         l.setContentsMargins(0, 0, 0, 0)
#
#         r, g, b, a = self.palette().base().color().getRgbF()
#
#         self._figure = Figure(edgecolor=(r, g, b), facecolor=(r, g, b))
#         self._canvas = FigureCanvas(self._figure)
#         l.addWidget(self._canvas)
#
#         self._figure.clear()
#         text = self._figure.suptitle(
#             mathText,
#             x=0.0,
#             y=1.0,
#             horizontalalignment='left',
#             verticalalignment='top',
#             size=qApp.font().pointSize() * 2)
#         self._canvas.draw()
#
#         (x0, y0), (x1, y1) = text.get_window_extent().get_points()
#         w = x1 - x0
#         h = y1 - y0
#
#         self._figure.set_size_inches(w / 80, h / 80)
#         self.setFixedSize(w, h)
#
#
# if __name__ == '__main__':
#     from sys import argv, exit
#
#
#     class Widget(QWidget):
#         def __init__(self, parent=None, **kwargs):
#             QWidget.__init__(self, parent, **kwargs)
#
#             l = QVBoxLayout(self)
#             l.addWidget(QLabel("<h1>Discrete Fourier Transform</h1>"))
#
#             mathText = r'$X_k = \sum_{n=0}^{N-1} x_n . e^{\frac{-i2\pi kn}{N}}$'
#             l.addWidget(MathTextLabel(mathText, self), alignment=Qt.AlignHCenter)
#
#
#     a = QApplication(argv)
#     w = Widget()
#     w.show()
#     w.raise_()
#     exit(a.exec_())
import numpy as np


a = [1, 1, 1, 2, 3, 4, 4, 5, 6, 6, 6, 90]

last = a[0]
i0, l = 0, 0
processed = True

while processed:
    processed = False
    for i in range(len(a)):
        if a[i] == last:
            l += 1
        else:
            if l > 1:
                processed = True
                del a[i0+1: i0 + l]
                i0, l = 0, 0
                break
            i0 = i
            l = 1
            last = a[i]
    if l > 1:
        processed = True
        del a[i0 + 1: i0 + l]
        i0, l = 0, 0


print(a)





#
# from pyqtgraph.dockarea import *
# from pyqtgraph import LayoutWidget
#
# import pyqtgraph.examples
# pyqtgraph.examples.run()
# #
# # from PyQt5 import QtCore, QtGui
#
# import pyqtgraph.opengl as gl
# import pyqtgraph as pg
# import numpy as np
# import sys
# import time
#
#
# class Visualizer(object):
#     def __init__(self):
#         self.traces = dict()
#         self.app = QtGui.QApplication(sys.argv)
#         self.w = gl.GLViewWidget()
#         self.w.opts['distance'] = 40
#         self.w.setWindowTitle('pyqtgraph example: GLLinePlotItem')
#         self.w.setGeometry(0, 110, 1920, 1080)
#         self.w.show()
#
#         self.phase = 0
#         self.lines = 50
#         self.points = 1000
#         self.y = np.linspace(-10, 10, self.lines)
#         self.x = np.linspace(-10, 10, self.points)
#
#         for i, line in enumerate(self.y):
#             y = np.array([line] * self.points)
#             d = np.sqrt(self.x ** 2 + y ** 2)
#             sine = 10 * np.sin(d + self.phase)
#             pts = np.vstack([self.x, y, sine]).transpose()
#             self.traces[i] = gl.GLLinePlotItem(
#                 pos=pts,
#                 color=pg.glColor((i, self.lines * 1.3)),
#                 width=(i + 1) / 10,
#                 antialias=True
#             )
#             self.w.addItem(self.traces[i])
#
#     def start(self):
#         if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#             QtGui.QApplication.instance().exec_()
#
#     def set_plotdata(self, name, points, color, width):
#         self.traces[name].setData(pos=points, color=color, width=width)
#
#     def update(self):
#         stime = time.time()
#         for i, line in enumerate(self.y):
#             y = np.array([line] * self.points)
#
#             amp = 10 / (i + 1)
#             phase = self.phase * (i + 1) - 10
#             freq = self.x * (i + 1) / 10
#
#             sine = amp * np.sin(freq - phase)
#             pts = np.vstack([self.x, y, sine]).transpose()
#
#             self.set_plotdata(
#                 name=i, points=pts,
#                 color=pg.glColor((i, self.lines * 1.3)),
#                 width=3
#             )
#             self.phase -= .00002
#
#         print('{:.0f} FPS'.format(1 / (time.time() - stime)))
#
#     def animation(self):
#         timer = QtCore.QTimer()
#         timer.timeout.connect(self.update)
#         timer.start(10)
#         self.start()
#
#
# # # Start event loop.
# # if __name__ == '__main__':
# #     v = Visualizer()
# #     v.animation()
# #
# from sympy import *
# # x, y = Symbol('x'), Function('y')
# # sol = dsolve(y(x).diff(x) - 1/y(x), y(x))
# #
# # print(sol)
#
#
#
#
# t, k1, k2 = symbols('t, k1, k2')
#
# A, B = Function('A'), Function('B')
#
# eq1 = Eq(A(t).diff(t), -k1*A(t))
# eq2 = Eq(B(t).diff(t), k1*A(t) -k2*B(t))
#
# print(eq1, eq2)
#
# sol1 = dsolve(eq1, A(t), ics={A(0): 2})
# sol2 = dsolve(Eq(B(t).diff(t), k1*2*exp(-k1*t) - k2*B(t)), B(t), ics={B(0): 0})
#
# print(sol2)
#
#
#
#
#
#
#

