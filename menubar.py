
from PyQt6.QtWidgets import QMenu
from PyQt6.QtGui import QAction, QActionGroup


from user_namespace import copy_plot_to_clipboard, save_fit
from PyQt6.QtWidgets import *

from Widgets.fitwidget import FitWidget

from Widgets.maindisplaydockarea import CommonDimension


class MenuBar(QMenuBar):

    MAX_RECENT_FILES = 30

    def __init__(self, parent=None, common_dimension_triggered=None):
        super(MenuBar, self).__init__(parent=parent)

        # ---File Menu---

        self.file_menu = self.addMenu("&File")

        self.common_dimension_triggered = common_dimension_triggered  # argument is the common dimension

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

        # ---Settings Menu---

        self.dim_menu = self.addMenu("Common Dimension")
        self.group = QActionGroup(self)

        self.none_act = QAction("None")
        self.first_act = QAction("First Dimension")
        self.second_act = QAction("Second Dimension")
        self.both_act = QAction("Both Dimensions")

        self.dim_menu.addAction(self.none_act)
        self.dim_menu.addAction(self.first_act)
        self.dim_menu.addAction(self.second_act)
        self.dim_menu.addAction(self.both_act)

        self.none_act.triggered.connect(lambda: self.common_dimension_triggered(CommonDimension.Not))
        self.first_act.triggered.connect(lambda: self.common_dimension_triggered(CommonDimension.First))
        self.second_act.triggered.connect(lambda: self.common_dimension_triggered(CommonDimension.Second))
        self.both_act.triggered.connect(lambda: self.common_dimension_triggered(CommonDimension.Both))

        self.none_act.setCheckable(True)
        self.first_act.setCheckable(True)
        self.second_act.setCheckable(True)
        self.both_act.setCheckable(True)

        self.none_act.setChecked(True)
        # row.setChecked(self.mode == DockDisplayMode.Row)
        # column.setChecked(self.mode == DockDisplayMode.Column)
        # second_act.setChecked(self.mode == DockDisplayMode.Matrix)

        self.none_act.setActionGroup(self.group)
        self.first_act.setActionGroup(self.group)
        self.second_act.setActionGroup(self.group)
        self.both_act.setActionGroup(self.group)

        # ---About Menu---

        self.about_menu = self.addMenu("&About")

        self.about_act = QAction("&About", self)
        self.about_act.triggered.connect(self.show_about_window)
        self.about_menu.addAction(self.about_act)

        # ---- Heat Map Menu

        # self.heat_map_menu = self.addMenu("&Heat Map")
        #
        # self.heat_map_copy_image = QAction("&Copy to clipboard as image", self)
        # self.heat_map_copy_image.triggered.connect(lambda: copy_plot_to_clipboard('heat_map', 'img'))
        # self.heat_map_menu.addAction(self.heat_map_copy_image)
        #
        # self.heat_map_copy_svg = QAction("&Copy to clipboard as SVG", self)
        # self.heat_map_copy_svg.triggered.connect(lambda: copy_plot_to_clipboard('heat_map', 'svg'))
        # self.heat_map_menu.addAction(self.heat_map_copy_svg)

        # ---- Trace Menu

        # self.trace_menu = self.addMenu("&Trace")
        #
        # self.trace_copy_image = QAction("&Copy to clipboard as image", self)
        # self.trace_copy_image.triggered.connect(lambda: copy_plot_to_clipboard('trace', 'img'))
        # self.trace_menu.addAction(self.trace_copy_image)
        #
        # self.trace_copy_svg = QAction("&Copy to clipboard as SVG", self)
        # self.trace_copy_svg.triggered.connect(lambda: copy_plot_to_clipboard('trace', 'svg'))
        # self.trace_menu.addAction(self.trace_copy_svg)

        # ---- Spectrum Menu

        # self.spectrum_menu = self.addMenu("&Spectrum")
        #
        # self.spectrum_copy_image = QAction("&Copy to clipboard as image", self)
        # self.spectrum_copy_image.triggered.connect(lambda: copy_plot_to_clipboard('spectrum', 'img'))
        # self.spectrum_menu.addAction(self.spectrum_copy_image)
        #
        # self.spectrum_copy_svg = QAction("&Copy to clipboard as SVG", self)
        # self.spectrum_copy_svg.triggered.connect(lambda: copy_plot_to_clipboard('spectrum', 'svg'))
        # self.spectrum_menu.addAction(self.spectrum_copy_svg)

    def set_common_dimension(self, dim: CommonDimension):
        if dim == CommonDimension.Not:
            self.none_act.setChecked(True)
        elif dim == CommonDimension.First:
            self.first_act.setChecked(True)
        elif dim == CommonDimension.Second:
            self.second_act.setChecked(True)
        else:
            self.both_act.setChecked(True)

    def open_recent_file(self):
        if self.sender():
            self.parent().open_file(filepaths=[self.sender().data()])

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

