from PyQt6.QtGui import *
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QSizePolicy, QLabel, QPushButton, QGridLayout, QVBoxLayout, QHBoxLayout, QCheckBox, QSpacerItem
from .mylineedit import MyLineEdit

from misc import setup_size_policy
import random


class DataPanel(QWidget):

    def _QLabel(self, text, **kwargs):
        name = "qlabel_" + "".join([random.choice('abcdefghijklmnopqrstuvwxyz') for i in range(10)])
        qlabel = QLabel(text, **kwargs)
        setattr(self, name, qlabel)

        return qlabel

    def __init__(self, parent=None):
        super(DataPanel, self).__init__(parent=parent)

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.grid = QGridLayout()
        self.main_layout.addLayout(self.grid)

        self.grid.addWidget(self._QLabel("Matrix size:"), 0, 0, Qt.AlignmentFlag.AlignLeft, 1)
        self.grid.addWidget(self._QLabel("Cropped matrix size:"), 1, 0, Qt.AlignmentFlag.AlignLeft, 1)

        self.lbl_matrix_size = QLabel("")
        self.lbl_cr_matrix_size = QLabel("")
        self.lbl_visible_area_msize = QLabel("")

        self.grid.addWidget(self.lbl_matrix_size, 0, 1, Qt.AlignmentFlag.AlignLeft, 1)
        self.grid.addWidget(self.lbl_cr_matrix_size, 1, 1, Qt.AlignmentFlag.AlignLeft, 1)

        self.grid.addWidget(self._QLabel("Visible area:"), 2, 0, Qt.AlignmentFlag.AlignLeft, 1)
        self.grid.addWidget(self.lbl_visible_area_msize, 2, 1, Qt.AlignmentFlag.AlignLeft, 1)

        self.txb_t0 = MyLineEdit()
        self.txb_t1 = MyLineEdit()
        self.txb_w0 = MyLineEdit()
        self.txb_w1 = MyLineEdit()

        self.grid.addWidget(self._QLabel("t0"), 3, 0, Qt.AlignmentFlag.AlignLeft, 1)
        self.grid.addWidget(self._QLabel("t1"), 3, 1, Qt.AlignmentFlag.AlignLeft, 1)

        self.grid.addWidget(self.txb_t0, 4, 0, Qt.AlignmentFlag.AlignLeft, 1)
        self.grid.addWidget(self.txb_t1, 4, 1, Qt.AlignmentFlag.AlignLeft, 1)

        self.grid.addWidget(self._QLabel("w0"), 5, 0, Qt.AlignmentFlag.AlignLeft, 1)
        self.grid.addWidget(self._QLabel("w1"), 5, 1, Qt.AlignmentFlag.AlignLeft, 1)

        self.grid.addWidget(self.txb_w0, 6, 0, Qt.AlignmentFlag.AlignLeft, 1)
        self.grid.addWidget(self.txb_w1, 6, 1, Qt.AlignmentFlag.AlignLeft, 1)

        # self.btn_crop_matrix = QPushButton("Crop to visible area")
        # self.btn_restore_matrix = QPushButton("Restore original matrix")

        hlayout_crop = QHBoxLayout()
        # hlayout_crop.addWidget(self.btn_crop_matrix)
        # hlayout_crop.addWidget(self.btn_restore_matrix)
        self.main_layout.addLayout(hlayout_crop)

        # self.btn_crop_matrix.clicked.connect(self.btn_crop_matrix_clicked)
        # self.btn_restore_matrix.clicked.connect(self.btn_restore_matrix_clicked)

        self.txb_n_spectra = MyLineEdit()
        self.btn_redraw_spectra = QPushButton("Redraw")

        hlayout = QHBoxLayout()
        hlayout.addWidget(self._QLabel("Number of spectra shown:"))
        hlayout.addWidget(self.txb_n_spectra)
        hlayout.addWidget(self.btn_redraw_spectra)
        self.main_layout.addLayout(hlayout)

        self.cb_SVD_filter = QCheckBox("SVD filter:")
        self.txb_SVD_filter = MyLineEdit()

        hlayout2 = QHBoxLayout()
        hlayout2.addWidget(self.cb_SVD_filter)
        hlayout2.addWidget(self.txb_SVD_filter)
        self.main_layout.addLayout(hlayout2)

        self.cb_ICA_filter = QCheckBox("ICA subtract filter:")
        self.txb_ICA_filter = MyLineEdit()

        hlayout3 = QHBoxLayout()
        hlayout3.addWidget(self.cb_ICA_filter)
        hlayout3.addWidget(self.txb_ICA_filter)
        self.main_layout.addLayout(hlayout3)

        self.grid2 = QGridLayout()
        self.main_layout.addLayout(self.grid2)

        self.grid2.addWidget(self._QLabel("Heat map levels:"), 0, 0, Qt.AlignmentFlag.AlignLeft, 1)

        self.txb_z0 = MyLineEdit()
        self.txb_z1 = MyLineEdit()

        self.grid2.addWidget(self._QLabel("z0"), 1, 0, Qt.AlignmentFlag.AlignLeft, 1)
        self.grid2.addWidget(self._QLabel("z1"), 1, 1, Qt.AlignmentFlag.AlignLeft, 1)

        self.grid2.addWidget(self.txb_z0, 2, 0, Qt.AlignmentFlag.AlignLeft, 1)
        self.grid2.addWidget(self.txb_z1, 2, 1, Qt.AlignmentFlag.AlignLeft, 1)

        # self.btn_center_levels = QPushButton("Center levels")
        # self.main_layout.addWidget(self.btn_center_levels)

        self.cb_show_chirp_points = QCheckBox("Show chirp points")
        self.btn_fit_chirp_params = QPushButton("Fit chirp params")

        self.main_layout.addWidget(self._QLabel("For femto fitting:"))

        hlayout2 = QHBoxLayout()
        hlayout2.addWidget(self.cb_show_chirp_points)
        hlayout2.addWidget(self.btn_fit_chirp_params)
        self.main_layout.addLayout(hlayout2)

        spacerItem = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.main_layout.addItem(spacerItem)

        setup_size_policy(self)

        # self.main_layout.addStretch(1)

    def btn_crop_matrix_clicked(self):
        pass

    def btn_restore_matrix_clicked(self):
        pass
