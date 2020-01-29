
import numpy as np
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication
import pyqtgraph.functions as fn
from pyqtgraph.exporters import ImageExporter
from pyqtgraph.Qt import USE_PYSIDE



# to fix th bug in pyqtgraph
class ImgExporter(ImageExporter):


    def __init__(self, item):
        super(ImgExporter, self).__init__(item)

    def export(self, fileName=None, toBytes=False, copy=True):

        if fileName is None and not toBytes and not copy:
            if USE_PYSIDE:
                filter = ["*." + str(f) for f in QtGui.QImageWriter.supportedImageFormats()]
            else:
                filter = ["*." + bytes(f).decode('utf-8') for f in QtGui.QImageWriter.supportedImageFormats()]
            preferred = ['*.png', '*.tif', '*.jpg']
            for p in preferred[::-1]:
                if p in filter:
                    filter.remove(p)
                    filter.insert(0, p)
            self.fileSaveDialog(filter=filter)
            return

        targetRect = QtCore.QRect(0, 0, self.params['width'], self.params['height'])
        sourceRect = self.getSourceRect()

        # self.png = QtGui.QImage(targetRect.size(), QtGui.QImage.Format_ARGB32)
        # self.png.fill(pyqtgraph.mkColor(self.params['background']))
        w, h = self.params['width'], self.params['height']
        if w == 0 or h == 0:
            raise Exception("Cannot export image with size=0 (requested export size is %dx%d)" % (w, h))
        # -------changed here |||||||||||||| int was necessary
        bg = np.empty((int(self.params['width']), int(self.params['height']), 4), dtype=np.ubyte)
        color = self.params['background']
        bg[:, :, 0] = color.blue()
        bg[:, :, 1] = color.green()
        bg[:, :, 2] = color.red()
        bg[:, :, 3] = color.alpha()
        self.png = fn.makeQImage(bg, alpha=True)

        ## set resolution of image:
        origTargetRect = self.getTargetRect()
        resolutionScale = targetRect.width() / origTargetRect.width()
        # self.png.setDotsPerMeterX(self.png.dotsPerMeterX() * resolutionScale)
        # self.png.setDotsPerMeterY(self.png.dotsPerMeterY() * resolutionScale)

        painter = QtGui.QPainter(self.png)
        # dtr = painter.deviceTransform()
        try:
            self.setExportMode(True, {'antialias': self.params['antialias'], 'background': self.params['background'],
                                      'painter': painter, 'resolutionScale': resolutionScale})
            painter.setRenderHint(QtGui.QPainter.Antialiasing, self.params['antialias'])
            self.getScene().render(painter, QtCore.QRectF(targetRect), QtCore.QRectF(sourceRect))
        finally:
            self.setExportMode(False)
        painter.end()

        if copy:
            QApplication.clipboard().setImage(self.png)
        elif toBytes:
            return self.png
        else:
            self.png.save(fileName)




