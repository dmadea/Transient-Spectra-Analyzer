
from PyQt5.QtWidgets import QMenuBar, QAction, QMenu

from user_namespace import copy_plot_to_clipboard, save_fit
from PyQt5.QtWidgets import *

from Widgets.fit_widget import FitWidget


class MenuBar(QMenuBar):

    MAX_RECENT_FILES = 30

    def __init__(self, parent=None):
        super(MenuBar, self).__init__(parent=parent)

        # ---File Menu---

        self.file_menu = self.addMenu("&File")

        # self.open_project_act = QAction("&Open Project", self)
        # self.open_project_act.setShortcut("Ctrl+O")
        # # self.open_project_act.triggered.connect(self.parent().open_project)
        # self.file_menu.addAction(self.open_project_act)

        # self.save_project_act = QAction("&Save Project", self)
        # self.save_project_act.setShortcut("Ctrl+S")
        # # self.save_project_act.triggered.connect(self.parent().save_project)
        # self.file_menu.addAction(self.save_project_act)
        #
        # self.save_project_as_act = QAction("Save Project &As", self)
        # self.file_menu.addAction(self.save_project_as_act)

        self.open_recent_menu = QMenu("Open Recent", self.file_menu)
        self.file_menu.addAction(self.open_recent_menu.menuAction())

        # add blank actions
        self.recent_file_actions = []
        for i in range(self.MAX_RECENT_FILES):
            act = self.open_recent_menu.addAction('')
            act.setVisible(False)
            self.recent_file_actions.append(act)
            act.triggered.connect(self.open_recent_file)

        self.file_menu.addSeparator()

        self.open_file_act = QAction("&Open File", self)
        self.open_file_act.triggered.connect(self.parent().open_file)
        # self.import_files.setShortcut("Ctrl+I")
        self.file_menu.addAction(self.open_file_act)

        self.save_fit_act = QAction("&Save Fit to File", self)
        self.save_fit_act.triggered.connect(self.save_fit_action)
        # self.import_files.setShortcut("Ctrl+I")
        self.file_menu.addAction(self.save_fit_act)

        # self.save_fit_MCR_act = QAction("&Save MCR Fit to File", self)
        # self.save_fit_MCR_act.triggered.connect(self.save_fit_MCR_action)
        # # self.import_files.setShortcut("Ctrl+I")
        # self.file_menu.addAction(self.save_fit_MCR_act)

        self.export_selected_spectra_as_act = QAction("&Export Selected Spectra As", self)
        # self.export_selected_spectra_as.setShortcut("Ctrl+E")
        # self.export_selected_spectra_as_act.triggered.connect(self.parent().export_selected_spectra_as)
        self.file_menu.addAction(self.export_selected_spectra_as_act)

        self.file_menu.addSeparator()

        self.settings_act = QAction("Se&ttings", self)
        # self.settings_act.triggered.connect(self.parent().open_settings)
        self.file_menu.addAction(self.settings_act)

        self.file_menu.addSeparator()

        self.exit_act = QAction("E&xit", self)
        self.exit_act.triggered.connect(self.parent().close)
        self.file_menu.addAction(self.exit_act)

        # ---Calculations Menu---
        #
        # self.calc_menu = self.addMenu("&Calculations")
        #
        # self.SVD_act = QAction('&Singular Value Decomposition')
        # self.SVD_act.triggered.connect(self.parent().perform_SVD)
        # self.calc_menu.addAction(self.SVD_act)
        #
        # self.global_fit_act = QAction('&Global Fit')
        # self.global_fit_act.triggered.connect(self.parent().open_fit_dialog)
        # self.calc_menu.addAction(self.global_fit_act)

        # ---About Menu---

        self.about_menu = self.addMenu("&About")

        self.about_act = QAction("&About", self)
        self.about_act.triggered.connect(self.show_about_window)
        self.about_menu.addAction(self.about_act)

        # ---- Heat Map Menu

        self.heat_map_menu = self.addMenu("&Heat Map")

        self.heat_map_copy_image = QAction("&Copy to clipboard as image", self)
        self.heat_map_copy_image.triggered.connect(lambda: copy_plot_to_clipboard('heat_map', 'img'))
        self.heat_map_menu.addAction(self.heat_map_copy_image)

        self.heat_map_copy_svg = QAction("&Copy to clipboard as SVG", self)
        self.heat_map_copy_svg.triggered.connect(lambda: copy_plot_to_clipboard('heat_map', 'svg'))
        self.heat_map_menu.addAction(self.heat_map_copy_svg)

        # ---- Trace Menu

        self.trace_menu = self.addMenu("&Trace")

        self.trace_copy_image = QAction("&Copy to clipboard as image", self)
        self.trace_copy_image.triggered.connect(lambda: copy_plot_to_clipboard('trace', 'img'))
        self.trace_menu.addAction(self.trace_copy_image)

        self.trace_copy_svg = QAction("&Copy to clipboard as SVG", self)
        self.trace_copy_svg.triggered.connect(lambda: copy_plot_to_clipboard('trace', 'svg'))
        self.trace_menu.addAction(self.trace_copy_svg)

        # ---- Spectrum Menu

        self.spectrum_menu = self.addMenu("&Spectrum")

        self.spectrum_copy_image = QAction("&Copy to clipboard as image", self)
        self.spectrum_copy_image.triggered.connect(lambda: copy_plot_to_clipboard('spectrum', 'img'))
        self.spectrum_menu.addAction(self.spectrum_copy_image)

        self.spectrum_copy_svg = QAction("&Copy to clipboard as SVG", self)
        self.spectrum_copy_svg.triggered.connect(lambda: copy_plot_to_clipboard('spectrum', 'svg'))
        self.spectrum_menu.addAction(self.spectrum_copy_svg)

    def open_recent_file(self):
        if self.sender():
            self.parent().open_file(filepath=self.sender().data())

    def show_about_window(self):
        self.parent().console.setVisible(True)

        about_message = """Some about message to be done....
        <span style='color:red'>Some Red text</span>
        <span style='color:red'><strong>Bold Red text</strong></span>
        """

        self.parent().console.print_html(about_message)

    def save_fit_action(self):

        filepath = QFileDialog.getSaveFileName(self, caption="Save Fit")

        if filepath[0] == '':
            return

        FitWidget.instance.matrix.save_fit(filepath[0])

