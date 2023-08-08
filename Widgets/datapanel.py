from PyQt6.QtGui import *
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QSizePolicy, QLabel, QPushButton, QGridLayout, QVBoxLayout, QHBoxLayout, QCheckBox, QSpacerItem, QTabWidget
from .mylineedit import MyLineEdit

from misc import setup_size_policy
import random


class DataPanel(QTabWidget):

    def _QLabel(self, text, **kwargs):
        name = "qlabel_" + "".join([random.choice('abcdefghijklmnopqrstuvwxyz') for i in range(10)])
        qlabel = QLabel(text, **kwargs)
        setattr(self, name, qlabel)

        return qlabel

    def initialize(self, matrices):

        for i in range(self.i):
            self.removeTab(0)

        self.i = 0

        for i, m in enumerate(matrices):
            self.addTab(self.create_panels(m, i), m.get_filename())

    def set_range(self, index, x0=None, x1=None, y0=None, y1=None):
        assert type(index) is int

        if y0 is not None:
            getattr(self, f'txb_t0{index}').setText(f'{y0:.4g}')
        if y1 is not None:
            getattr(self, f'txb_t1{index}').setText(f'{y1:.4g}')
        if x0 is not None:
            getattr(self, f'txb_w0{index}').setText(f'{x0:.4g}')
        if x1 is not None:
            getattr(self, f'txb_w1{index}').setText(f'{x1:.4g}')

    def create_panels(self, matrix, index):

        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        grid = QGridLayout()
        main_layout.addLayout(grid)

        grid.addWidget(self._QLabel("Matrix size:"), 0, 0, Qt.AlignmentFlag.AlignLeft, 1)
        grid.addWidget(self._QLabel("Cropped matrix size:"), 1, 0, Qt.AlignmentFlag.AlignLeft, 1)

        lbl_matrix_size = QLabel(f"{matrix.D.shape[0]} x {matrix.D.shape[1]}")
        lbl_cr_matrix_size = QLabel(f"{matrix.D.shape[0]} x {matrix.D.shape[1]}")
        lbl_visible_area_msize = QLabel("")

        grid.addWidget(lbl_matrix_size, 0, 1, Qt.AlignmentFlag.AlignLeft, 1)
        grid.addWidget(lbl_cr_matrix_size, 1, 1, Qt.AlignmentFlag.AlignLeft, 1)

        grid.addWidget(self._QLabel("Visible area:"), 2, 0, Qt.AlignmentFlag.AlignLeft, 1)
        grid.addWidget(lbl_visible_area_msize, 2, 1, Qt.AlignmentFlag.AlignLeft, 1)

        txb_t0 = MyLineEdit()
        txb_t1 = MyLineEdit()
        txb_w0 = MyLineEdit()
        txb_w1 = MyLineEdit()

        grid.addWidget(self._QLabel("y0"), 3, 0, Qt.AlignmentFlag.AlignLeft, 1)
        grid.addWidget(self._QLabel("y1"), 3, 1, Qt.AlignmentFlag.AlignLeft, 1)

        grid.addWidget(txb_t0, 4, 0, Qt.AlignmentFlag.AlignLeft, 1)
        grid.addWidget(txb_t1, 4, 1, Qt.AlignmentFlag.AlignLeft, 1)

        grid.addWidget(self._QLabel("x0"), 5, 0, Qt.AlignmentFlag.AlignLeft, 1)
        grid.addWidget(self._QLabel("x1"), 5, 1, Qt.AlignmentFlag.AlignLeft, 1)

        grid.addWidget(txb_w0, 6, 0, Qt.AlignmentFlag.AlignLeft, 1)
        grid.addWidget(txb_w1, 6, 1, Qt.AlignmentFlag.AlignLeft, 1)

        # # self.btn_crop_matrix = QPushButton("Crop to visible area")
        # # self.btn_restore_matrix = QPushButton("Restore original matrix")
        #
        # hlayout_crop = QHBoxLayout()
        # # hlayout_crop.addWidget(self.btn_crop_matrix)
        # # hlayout_crop.addWidget(self.btn_restore_matrix)
        # self.main_layout.addLayout(hlayout_crop)
        #
        # # self.btn_crop_matrix.clicked.connect(self.btn_crop_matrix_clicked)
        # # self.btn_restore_matrix.clicked.connect(self.btn_restore_matrix_clicked)

        txb_n_spectra = MyLineEdit()
        btn_redraw_spectra = QPushButton("Redraw")

        hlayout = QHBoxLayout()
        hlayout.addWidget(self._QLabel("Number of spectra shown:"))
        hlayout.addWidget(txb_n_spectra)
        hlayout.addWidget(btn_redraw_spectra)
        main_layout.addLayout(hlayout)

        cb_SVD_filter = QCheckBox("SVD filter:")
        txb_SVD_filter = MyLineEdit()

        hlayout2 = QHBoxLayout()
        hlayout2.addWidget(cb_SVD_filter)
        hlayout2.addWidget(txb_SVD_filter)
        main_layout.addLayout(hlayout2)

        cb_ICA_filter = QCheckBox("ICA subtract filter:")
        txb_ICA_filter = MyLineEdit()

        hlayout3 = QHBoxLayout()
        hlayout3.addWidget(cb_ICA_filter)
        hlayout3.addWidget(txb_ICA_filter)
        main_layout.addLayout(hlayout3)

        grid2 = QGridLayout()
        main_layout.addLayout(grid2)

        grid2.addWidget(self._QLabel("Heat map levels:"), 0, 0, Qt.AlignmentFlag.AlignLeft, 1)

        txb_z0 = MyLineEdit()
        txb_z1 = MyLineEdit()

        grid2.addWidget(self._QLabel("z0"), 1, 0, Qt.AlignmentFlag.AlignLeft, 1)
        grid2.addWidget(self._QLabel("z1"), 1, 1, Qt.AlignmentFlag.AlignLeft, 1)

        grid2.addWidget(txb_z0, 2, 0, Qt.AlignmentFlag.AlignLeft, 1)
        grid2.addWidget(txb_z1, 2, 1, Qt.AlignmentFlag.AlignLeft, 1)

        # self.btn_center_levels = QPushButton("Center levels")
        # self.main_layout.addWidget(self.btn_center_levels)

        cb_show_chirp_points = QCheckBox("Show chirp points")
        btn_fit_chirp_params = QPushButton("Fit chirp params")

        scp_name = f"cb_show_cp_{index}"
        fcp_name = f"btn_fit_cp_{index}"

        setattr(self, scp_name, cb_show_chirp_points)
        setattr(self, fcp_name, btn_fit_chirp_params)

        main_layout.addWidget(self._QLabel("For femto fitting:"))

        hlayout2 = QHBoxLayout()
        hlayout2.addWidget(cb_show_chirp_points)
        hlayout2.addWidget(btn_fit_chirp_params)
        main_layout.addLayout(hlayout2)

        spacerItem = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        main_layout.addItem(spacerItem)

        for key, attr in locals().items():
            if key == 'self' or key == 'matrix':
                continue

            setattr(self, f'{key}{self.i}', attr)

        self.i += 1
        return main_widget

    def __init__(self, parent=None):
        super(DataPanel, self).__init__(parent=parent)

        self.i = 0  # index of added attributes

        # self.main_layout = QVBoxLayout()
        # self.main_layout.addWidget(self.tab_widget)
        #
        # self.setLayout(self.main_layout)

        # self.grid = QGridLayout()
        # self.main_layout.addLayout(self.grid)
        #
        # self.grid.addWidget(self._QLabel("Matrix size:"), 0, 0, Qt.AlignmentFlag.AlignLeft, 1)
        # self.grid.addWidget(self._QLabel("Cropped matrix size:"), 1, 0, Qt.AlignmentFlag.AlignLeft, 1)
        #
        # self.lbl_matrix_size = QLabel("")
        # self.lbl_cr_matrix_size = QLabel("")
        # self.lbl_visible_area_msize = QLabel("")
        #
        # self.grid.addWidget(self.lbl_matrix_size, 0, 1, Qt.AlignmentFlag.AlignLeft, 1)
        # self.grid.addWidget(self.lbl_cr_matrix_size, 1, 1, Qt.AlignmentFlag.AlignLeft, 1)
        #
        # self.grid.addWidget(self._QLabel("Visible area:"), 2, 0, Qt.AlignmentFlag.AlignLeft, 1)
        # self.grid.addWidget(self.lbl_visible_area_msize, 2, 1, Qt.AlignmentFlag.AlignLeft, 1)
        #
        # self.txb_t0 = MyLineEdit()
        # self.txb_t1 = MyLineEdit()
        # self.txb_w0 = MyLineEdit()
        # self.txb_w1 = MyLineEdit()
        #
        # self.grid.addWidget(self._QLabel("y0"), 3, 0, Qt.AlignmentFlag.AlignLeft, 1)
        # self.grid.addWidget(self._QLabel("y1"), 3, 1, Qt.AlignmentFlag.AlignLeft, 1)
        #
        # self.grid.addWidget(self.txb_t0, 4, 0, Qt.AlignmentFlag.AlignLeft, 1)
        # self.grid.addWidget(self.txb_t1, 4, 1, Qt.AlignmentFlag.AlignLeft, 1)
        #
        # self.grid.addWidget(self._QLabel("x0"), 5, 0, Qt.AlignmentFlag.AlignLeft, 1)
        # self.grid.addWidget(self._QLabel("x1"), 5, 1, Qt.AlignmentFlag.AlignLeft, 1)
        #
        # self.grid.addWidget(self.txb_w0, 6, 0, Qt.AlignmentFlag.AlignLeft, 1)
        # self.grid.addWidget(self.txb_w1, 6, 1, Qt.AlignmentFlag.AlignLeft, 1)
        #
        # # self.btn_crop_matrix = QPushButton("Crop to visible area")
        # # self.btn_restore_matrix = QPushButton("Restore original matrix")
        #
        # hlayout_crop = QHBoxLayout()
        # # hlayout_crop.addWidget(self.btn_crop_matrix)
        # # hlayout_crop.addWidget(self.btn_restore_matrix)
        # self.main_layout.addLayout(hlayout_crop)
        #
        # # self.btn_crop_matrix.clicked.connect(self.btn_crop_matrix_clicked)
        # # self.btn_restore_matrix.clicked.connect(self.btn_restore_matrix_clicked)
        #
        # self.txb_n_spectra = MyLineEdit()
        # self.btn_redraw_spectra = QPushButton("Redraw")
        #
        # hlayout = QHBoxLayout()
        # hlayout.addWidget(self._QLabel("Number of spectra shown:"))
        # hlayout.addWidget(self.txb_n_spectra)
        # hlayout.addWidget(self.btn_redraw_spectra)
        # self.main_layout.addLayout(hlayout)
        #
        # self.cb_SVD_filter = QCheckBox("SVD filter:")
        # self.txb_SVD_filter = MyLineEdit()
        #
        # hlayout2 = QHBoxLayout()
        # hlayout2.addWidget(self.cb_SVD_filter)
        # hlayout2.addWidget(self.txb_SVD_filter)
        # self.main_layout.addLayout(hlayout2)
        #
        # self.cb_ICA_filter = QCheckBox("ICA subtract filter:")
        # self.txb_ICA_filter = MyLineEdit()
        #
        # hlayout3 = QHBoxLayout()
        # hlayout3.addWidget(self.cb_ICA_filter)
        # hlayout3.addWidget(self.txb_ICA_filter)
        # self.main_layout.addLayout(hlayout3)
        #
        # self.grid2 = QGridLayout()
        # self.main_layout.addLayout(self.grid2)
        #
        # self.grid2.addWidget(self._QLabel("Heat map levels:"), 0, 0, Qt.AlignmentFlag.AlignLeft, 1)
        #
        # self.txb_z0 = MyLineEdit()
        # self.txb_z1 = MyLineEdit()
        #
        # self.grid2.addWidget(self._QLabel("z0"), 1, 0, Qt.AlignmentFlag.AlignLeft, 1)
        # self.grid2.addWidget(self._QLabel("z1"), 1, 1, Qt.AlignmentFlag.AlignLeft, 1)
        #
        # self.grid2.addWidget(self.txb_z0, 2, 0, Qt.AlignmentFlag.AlignLeft, 1)
        # self.grid2.addWidget(self.txb_z1, 2, 1, Qt.AlignmentFlag.AlignLeft, 1)
        #
        # # self.btn_center_levels = QPushButton("Center levels")
        # # self.main_layout.addWidget(self.btn_center_levels)
        #
        # self.cb_show_chirp_points = QCheckBox("Show chirp points")
        # self.btn_fit_chirp_params = QPushButton("Fit chirp params")
        #
        # self.main_layout.addWidget(self._QLabel("For femto fitting:"))
        #
        # hlayout2 = QHBoxLayout()
        # hlayout2.addWidget(self.cb_show_chirp_points)
        # hlayout2.addWidget(self.btn_fit_chirp_params)
        # self.main_layout.addLayout(hlayout2)
        #
        # spacerItem = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        # self.main_layout.addItem(spacerItem)

        setup_size_policy(self)

        # self.main_layout.addStretch(1)

    def btn_crop_matrix_clicked(self):
        pass

    def btn_restore_matrix_clicked(self):
        pass
