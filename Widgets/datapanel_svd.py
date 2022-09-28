from PyQt6.QtGui import *
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QSpacerItem, QSizePolicy, QLabel, QPushButton, QGridLayout, QVBoxLayout, QHBoxLayout, QCheckBox, QSpinBox
from .mylineedit import MyLineEdit
from misc import setup_size_policy

import random



class DataPanelSVD(QWidget):

    def _QLabel(self, text, **kwargs):
        name = "qlabel_" + "".join([random.choice('abcdefghijklmnopqrstuvwxyz') for i in range(10)])
        qlabel = QLabel(text, **kwargs)
        setattr(self, name, qlabel)

        return qlabel

    def __init__(self, parent=None):
        super(DataPanelSVD, self).__init__(parent=parent)

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.cb_SVD = QCheckBox("From selection")
        self.main_layout.addWidget(self.cb_SVD)

        # EFA settings

        self.main_layout.addWidget(self._QLabel('Evolving Factor Analysis setting:'))

        self.sb_n_t_points = QSpinBox()
        self.sb_n_t_points.setMinimum(2)
        self.sb_n_t_points.setMaximum(1000)
        self.sb_n_t_points.setValue(40)

        self.sb_n_svals = QSpinBox()
        self.sb_n_svals.setMinimum(1)
        self.sb_n_svals.setMaximum(100)
        self.sb_n_svals.setValue(7)

        hlayout = QHBoxLayout()
        hlayout.addWidget(self._QLabel("Number of time points:"))
        hlayout.addWidget(self.sb_n_t_points)
        self.main_layout.addLayout(hlayout)

        hlayout2 = QHBoxLayout()
        hlayout2.addWidget(self._QLabel("Number of EFA singular values shown:"))
        hlayout2.addWidget(self.sb_n_svals)
        self.main_layout.addLayout(hlayout2)

        self.btn_fEFA = QPushButton("Run forward EFA")
        # self.btn_bEFA = QPushButton("Run backward EFA")

        self.main_layout.addWidget(self.btn_fEFA)
        # self.main_layout.addWidget(self.btn_bEFA)

        self.sb_n_vectors = QSpinBox()
        self.sb_n_vectors.setMinimum(1)
        self.sb_n_vectors.setMaximum(1000)
        self.sb_n_vectors.setValue(3)

        self.main_layout.addWidget(self._QLabel('Independent Component Analysis (ICA) setting:'))

        self.sb_n_ICA = QSpinBox()
        self.sb_n_ICA.setMinimum(1)
        self.sb_n_ICA.setMaximum(100)
        self.sb_n_ICA.setValue(5)

        hlayout3 = QHBoxLayout()
        hlayout3.addWidget(self._QLabel("Number of ICA components:"))
        hlayout3.addWidget(self.sb_n_ICA)
        self.main_layout.addLayout(hlayout3)

        # self.btn_ICA = QPushButton("Run ICA")
        # self.main_layout.addWidget(self.btn_ICA)

        self.cb_show_ICA_ins_SVD = QCheckBox("Show ICA instead of SVD vectors")
        self.main_layout.addWidget(self.cb_show_ICA_ins_SVD)

        hlayout = QHBoxLayout()
        hlayout.addWidget(self._QLabel("Displayed vectors:"))
        hlayout.addWidget(self.sb_n_vectors)
        self.main_layout.addLayout(hlayout)

        self.cb_show_all = QCheckBox("Show all")
        self.cb_show_all.setChecked(True)
        self.main_layout.addWidget(self.cb_show_all)

        spacerItem = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.main_layout.addItem(spacerItem)

        setup_size_policy(self)

        # self.main_layout.addStretch(1)
