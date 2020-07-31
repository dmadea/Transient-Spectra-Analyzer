from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QLineEdit, QLabel, QPushButton, QGridLayout, QVBoxLayout, QHBoxLayout, QCheckBox
from .mylineedit import MyLineEdit


class DataPanel(QWidget):

    def __init__(self, parent=None):
        super(DataPanel, self).__init__(parent=parent)

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.grid = QGridLayout()
        self.main_layout.addLayout(self.grid)

        self.grid.addWidget(QLabel("Matrix size:"), 0, 0, Qt.AlignLeft, 1)
        self.grid.addWidget(QLabel("Cropped\nmatrix size:"), 1, 0, Qt.AlignLeft, 1)

        self.lbl_matrix_size = QLabel("")
        self.lbl_cr_matrix_size = QLabel("")

        self.grid.addWidget(self.lbl_matrix_size, 0, 1, Qt.AlignLeft, 1)
        self.grid.addWidget(self.lbl_cr_matrix_size, 1, 1, Qt.AlignLeft, 1)

        self.grid.addWidget(QLabel("Visible area:"), 2, 0, Qt.AlignLeft, 1)

        self.txb_t0 = MyLineEdit()
        self.txb_t1 = MyLineEdit()
        self.txb_w0 = MyLineEdit()
        self.txb_w1 = MyLineEdit()

        self.grid.addWidget(QLabel("t0"), 3, 0, Qt.AlignLeft, 1)
        self.grid.addWidget(QLabel("t1"), 3, 1, Qt.AlignLeft, 1)

        self.grid.addWidget(self.txb_t0, 4, 0, Qt.AlignLeft, 1)
        self.grid.addWidget(self.txb_t1, 4, 1, Qt.AlignLeft, 1)

        self.grid.addWidget(QLabel("w0"), 5, 0, Qt.AlignLeft, 1)
        self.grid.addWidget(QLabel("w1"), 5, 1, Qt.AlignLeft, 1)

        self.grid.addWidget(self.txb_w0, 6, 0, Qt.AlignLeft, 1)
        self.grid.addWidget(self.txb_w1, 6, 1, Qt.AlignLeft, 1)

        self.btn_crop_matrix = QPushButton("Crop to visible area")
        self.btn_restore_matrix = QPushButton("Restore original matrix")

        self.btn_crop_matrix.clicked.connect(self.btn_crop_matrix_clicked)
        self.btn_restore_matrix.clicked.connect(self.btn_restore_matrix_clicked)

        self.main_layout.addWidget(self.btn_crop_matrix)
        self.main_layout.addWidget(self.btn_restore_matrix)

        self.txb_n_spectra = MyLineEdit()

        hlayout = QHBoxLayout()
        hlayout.addWidget(QLabel("Number of\nspectra shown:"))
        hlayout.addWidget(self.txb_n_spectra)
        self.main_layout.addLayout(hlayout)

        self.btn_redraw_spectra = QPushButton("Redraw spectra to selection")
        self.main_layout.addWidget(self.btn_redraw_spectra)

        self.cb_SVD_filter = QCheckBox("SVD filter:")
        self.txb_SVD_filter = MyLineEdit()

        hlayout2 = QHBoxLayout()
        hlayout2.addWidget(self.cb_SVD_filter)
        hlayout2.addWidget(self.txb_SVD_filter)
        self.main_layout.addLayout(hlayout2)

        self.grid2 = QGridLayout()
        self.main_layout.addLayout(self.grid2)

        self.grid2.addWidget(QLabel("Heat map levels:"), 0, 0, Qt.AlignLeft, 1)

        self.txb_z0 = MyLineEdit()
        self.txb_z1 = MyLineEdit()

        self.grid2.addWidget(QLabel("z0"), 1, 0, Qt.AlignLeft, 1)
        self.grid2.addWidget(QLabel("z1"), 1, 1, Qt.AlignLeft, 1)

        self.grid2.addWidget(self.txb_z0, 2, 0, Qt.AlignLeft, 1)
        self.grid2.addWidget(self.txb_z1, 2, 1, Qt.AlignLeft, 1)

        self.btn_center_levels = QPushButton("Center levels")
        self.main_layout.addWidget(self.btn_center_levels)

        self.main_layout.addStretch(1)

    def btn_crop_matrix_clicked(self):
        pass

    def btn_restore_matrix_clicked(self):
        pass
