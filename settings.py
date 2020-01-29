import json
from logger import Logger

import numpy as np


class Settings(object):
    """Static class containing user settings with static functions for saving and loading to/from json file."""

    # settings that starts with '_' will not be saved into a file
    # static variables

    _filepath = "config.json"
    _inst_default_settings = None  # default settings instance

    # Simple Spectra Manipulator version
    __version__ = "0.1.0"

    # --------SETTINGS----------

    default_project_filename = "Untilted"
    # default_group_name = "--Group Name--"

    # import options

    csv_imp_delimiter = ','
    csv_imp_decimal_sep = '.'
    general_imp_delimiter = '\t'
    general_imp_decimal_sep = '.'
    dx_imp_delimiter = ' '
    dx_imp_decimal_sep = '.'
    dx_import_spectra_name_from_filename = False
    dx_if_title_is_empty_use_filename = True
    general_import_spectra_name_from_filename = False
    general_if_header_is_empty_use_filename = True

    clip_imp_delimiter = '\t'
    clip_imp_decimal_sep = '.'

    excel_imp_as_text = False

    # file export options

    files_exp_include_group_name = False
    files_exp_include_header = True
    csv_exp_delimiter = ','
    csv_exp_decimal_sep = '.'
    general_exp_delimiter = '\t'
    general_exp_decimal_sep = '.'

    # clipboard export options

    clip_exp_include_group_name = True
    clip_exp_include_header = True
    clip_exp_delimiter = '\t'
    clip_exp_decimal_sep = '.'

    # plotting settings

    graph_title = ""

    antialiasing = True
    left_axis_label = "Absorbance"
    left_axis_unit = None

    bottom_axis_label = "Wavelength (nm)"
    bottom_axis_unit = None

    show_grid = False
    grid_alpha = 0.1

    line_width = 1

    graph_title_font_size = 20
    bottom_axis_font_size = 20
    left_axis_font_size = 20

    same_color_in_group = True
    different_line_style_among_groups = False

    legend_spacing = 13


    # saved ranges

    baseline_correct_range = (300.0, 800.0)
    normalize_range = (300.0, 800.0)
    cut_range = (300.0, 800.0)
    integrate_range = (300.0, 800.0)
    last_rename_expression = ""
    last_rename_take_name_from_list = False

    # files dialog last paths

    import_files_dialog_path = ""
    open_project_dialog_path = ""
    save_project_dialog_path = ""
    export_spectra_as_dialog_path = ""
    export_spectra_as_dialog_ext = '.txt'

    # recent projects filepaths

    recent_project_filepaths = []

    gui_settings_last_tab_index = 0


    # --- FITTING MODELS----

    fitting_models = [
        {
            'name': 'A->B (lifetime)',
            'equation': 'return A * exp(-x / tau) + y0',
            'params': ['A', 'tau', 'y0'],
            'lower_bounds': [-np.inf, 0, -np.inf],
            'upper_bounds': [np.inf, np.inf, np.inf],
            'init_func': 'lambda x_data, y_data: (y_data[0], 1, 0)'
        },
        {
            'name': 'A->B (rate constant)',
            'equation': 'return A * exp(-k * x) + y0',
            'params': ['A', 'k', 'y0'],
            'lower_bounds': [-np.inf, 0, -np.inf],
            'upper_bounds': [np.inf, np.inf, np.inf],
            'init_func': 'lambda x_data, y_data: (y_data[0], 1, 0)'
        },
        {
            'name': 'A->B->C (B visible)',
            'equation': """# if k1~k2, we have to change the equation in order not to get zero division error
if abs(k1 - k2) < 1e-8:
    return A * k1 * x * exp(-k1 * x) + y0
else:
    return A * (k1 / (k2 - k1)) * (exp(-k1 * x) - exp(-k2 * x)) + y0""",
            'params': ['A', 'k1', 'k2', 'y0'],
            'lower_bounds': [-np.inf, 0, 0, -np.inf],
            'upper_bounds': [np.inf, np.inf, np.inf, np.inf],
            'init_func': 'lambda x_data, y_data: (max(np.abs(y_data)), 1, 0.5, 0)'
        },
        {
            'name': 'A->B->C (A + B visible)',
            'equation': """# if k1~k2, we have to change the equation in order not to get zero division error
if abs(k1 - k2) < 1e-8:
    return A1 * exp(-k1 * x) + A2 * k1 * x * exp(-k1 * x) + y0
else:
    return A1 * exp(-k1 * x) + A2 * (k1 / (k2 - k1)) * (exp(-k1 * x) - exp(-k2 * x)) + y0""",
            'params': ['A1', 'A2', 'k1', 'k2', 'y0'],
            'lower_bounds': [-np.inf, -np.inf, 0, 0, -np.inf],
            'upper_bounds': [np.inf, np.inf, np.inf, np.inf, np.inf],
            'init_func': 'lambda x_data, y_data: (0, max(np.abs(y_data)), 1, 0.5, 0)'
        },
        {
            'name': 'A->B->C->D (C visible)',
            'equation': """# if k1~k2~k3 and their combinations, we have to change the equation in order not to get zero division error
_ = 1e-8
if abs(k1 - k2) < _ and abs(k1 - k3) < _ and abs(k2 - k3) < _:
    #print("k1=k2=k3")
    return A * k1 * k1 * x * x * exp(-k1 * x) / 2 + y0
if abs(k1 - k2) < _:
    #print("k1=k2")
    return A * k1 * k1 * exp(-x * (k3 + k1)) * (exp(k3 * x) * (k3 * x - k1 * x - 1) + exp(k1 * x)) / (k3 - k1) ** 2 + y0
if abs(k1 - k3) < _:
    #print("k1=k3")
    return A * k2 * k1 * exp(-x * (k2 + k1)) * (exp(k2 * x) * (k2 * x - k1 * x - 1) + exp(k1 * x)) / (k2 - k1) ** 2 + y0
if abs(k2 - k3) < _:
    #print("k2=k3")
    return A * k1 * k2 * exp(-x * (k1 + k2)) * (exp(k1 * x) * (k1 * x - k2 * x - 1) + exp(k2 * x)) / (k1 - k2) ** 2 + y0
#print("different")
return A * k1 * k2 * exp(-x * (k1 + k2 + k3)) * ((k1 - k2) * exp(x * (k1 + k2)) + (k3 - k1) * exp(x * (k1 + k3)) + (k2 - k3) * exp(x * (k2 + k3))) / ((k1 - k2) * (k1 - k3) * (k2 - k3)) + y0""",
            'params': ['A', 'k1', 'k2', 'k3', 'y0'],
            'lower_bounds': [-np.inf, 0, 0, 0, -np.inf],
            'upper_bounds': [np.inf, np.inf, np.inf, np.inf, np.inf],
            'init_func': 'lambda x_data, y_data: (max(np.abs(y_data)), 1, 0.5, 0.2, 0)'
        },
        {
            'name': 'A->B->C->D (A + C visible)',
            'equation': """# if k1~k2~k3 and their combinations, we have to change the equation in order not to get zero division error
_ = 1e-8
if abs(k1 - k2) < _ and abs(k1 - k3) < _ and abs(k2 - k3) < _:
    #print("k1=k2=k3")
    return A1 * exp(-k1 * x) + A2 * k1 * k1 * x * x * exp(-k1 * x) / 2 + y0
if abs(k1 - k2) < _:
    #print("k1=k2")
    return A1 * exp(-k1 * x) + A2 * k1 * k1 * exp(-x * (k3 + k1)) * (exp(k3 * x) * (k3 * x - k1 * x - 1) + exp(k1 * x)) / (k3 - k1) ** 2 + y0
if abs(k1 - k3) < _:
    #print("k1=k3")
    return A1 * exp(-k1 * x) + A2 * k2 * k1 * exp(-x * (k2 + k1)) * (exp(k2 * x) * (k2 * x - k1 * x - 1) + exp(k1 * x)) / (k2 - k1) ** 2 + y0
if abs(k2 - k3) < _:
    #print("k2=k3")
    return A1 * exp(-k1 * x) + A2 * k1 * k2 * exp(-x * (k1 + k2)) * (exp(k1 * x) * (k1 * x - k2 * x - 1) + exp(k2 * x)) / (k1 - k2) ** 2 + y0
#print("different")
return A1 * exp(-k1 * x) + A2 * k1 * k2 * exp(-x * (k1 + k2 + k3)) * ((k1 - k2) * exp(x * (k1 + k2)) + (k3 - k1) * exp(x * (k1 + k3)) + (k2 - k3) * exp(x * (k2 + k3))) / ((k1 - k2) * (k1 - k3) * (k2 - k3)) + y0""",
            'params': ['A1', 'A2', 'k1', 'k2', 'k3', 'y0'],
            'lower_bounds': [-np.inf, -np.inf, 0, 0, 0, -np.inf],
            'upper_bounds': [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
            'init_func': 'lambda x_data, y_data: (max(np.abs(y_data)), max(np.abs(y_data)), 1, 0.5, 0.2, 0)'
        },
        {
            'name': 'A->B->C->D (B + C visible)',
            'equation': """# if k1~k2~k3 and their combinations, we have to change the equation in order not to get zero division error
_ = 1e-8
if abs(k1 - k2) < _ and abs(k1 - k3) < _ and abs(k2 - k3) < _:
    #print("k1=k2=k3")
    return A1 * k1 * x * exp(-k1 * x) + A2 * k1 * k1 * x * x * exp(-k1 * x) / 2 + y0
if abs(k1 - k2) < _:
    #print("k1=k2"
    return A1 * k1 * x * exp(-k1 * x) + A2 * k1 * k1 * exp(-x * (k3 + k1)) * (exp(k3 * x) * (k3 * x - k1 * x - 1) + exp(k1 * x)) / (k3 - k1) ** 2 + y0
if abs(k1 - k3) < _:
    #print("k1=k3")
    return A1 * (k1 / (k2 - k1)) * (exp(-k1 * x) - exp(-k2 * x)) + A2 * k2 * k1 * exp(-x * (k2 + k1)) * (exp(k2 * x) * (k2 * x - k1 * x - 1) + exp(k1 * x)) / (k2 - k1) ** 2 + y0
if abs(k2 - k3) < _:
    #print("k2=k3")
    return A1 * (k1 / (k2 - k1)) * (exp(-k1 * x) - exp(-k2 * x)) + A2 * k1 * k2 * exp(-x * (k1 + k2)) * (exp(k1 * x) * (k1 * x - k2 * x - 1) + exp(k2 * x)) / (k1 - k2) ** 2 + y0
#print("different")
return A1 * (k1 / (k2 - k1)) * (exp(-k1 * x) - exp(-k2 * x)) + A2 * k1 * k2 * exp(-x * (k1 + k2 + k3)) * ((k1 - k2) * exp(x * (k1 + k2)) + (k3 - k1) * exp(x * (k1 + k3)) + (k2 - k3) * exp(x * (k2 + k3))) / ((k1 - k2) * (k1 - k3) * (k2 - k3)) + y0""",
            'params': ['A1', 'A2', 'k1', 'k2', 'k3', 'y0'],
            'lower_bounds': [-np.inf, -np.inf, 0, 0, 0, -np.inf],
            'upper_bounds': [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
            'init_func': 'lambda x_data, y_data: (max(np.abs(y_data)), max(np.abs(y_data)), 1, 0.5, 0.2, 0)'
        },
        {
            'name': 'A->B->C->D (A + B + C visible)',
            'equation': """# if k1~k2~k3 and their combinations, we have to change the equation in order not to get zero division error
_ = 1e-8
if abs(k1 - k2) < _ and abs(k1 - k3) < _ and abs(k2 - k3) < _:
    #print("k1=k2=k3")
    return A1 * exp(-k1 * x) + A2 * k1 * x * exp(-k1 * x) + A3 * k1 * k1 * x * x * exp(-k1 * x) / 2 + y0
if abs(k1 - k2) < _:
    #print("k1=k2")
    return A1 * exp(-k1 * x) + A2 * k1 * x * exp(-k1 * x) + A3 * k1 * k1 * exp(-x * (k3 + k1)) * (exp(k3 * x) * (k3 * x - k1 * x - 1) + exp(k1 * x)) / (k3 - k1) ** 2 + y0
if abs(k1 - k3) < _:
    #print("k1=k3")
    return A1 * exp(-k1 * x) + A2 * (k1 / (k2 - k1)) * (exp(-k1 * x) - exp(-k2 * x)) + A3 * k2 * k1 * exp(-x * (k2 + k1)) * (exp(k2 * x) * (k2 * x - k1 * x - 1) + exp(k1 * x)) / (k2 - k1) ** 2 + y0
if abs(k2 - k3) < _:
    #print("k2=k3")
    return A1 * exp(-k1 * x) + A2 * (k1 / (k2 - k1)) * (exp(-k1 * x) - exp(-k2 * x)) + A3 * k1 * k2 * exp(-x * (k1 + k2)) * (exp(k1 * x) * (k1 * x - k2 * x - 1) + exp(k2 * x)) / (k1 - k2) ** 2 + y0
#print("different")
return A1 * exp(-k1 * x) + A2 * (k1 / (k2 - k1)) * (exp(-k1 * x) - exp(-k2 * x)) + A3 * k1 * k2 * exp(-x * (k1 + k2 + k3)) * ((k1 - k2) * exp(x * (k1 + k2)) + (k3 - k1) * exp(x * (k1 + k3)) + (k2 - k3) * exp(x * (k2 + k3))) / ((k1 - k2) * (k1 - k3) * (k2 - k3)) + y0""",
            'params': ['A1', 'A2', 'A3', 'k1', 'k2', 'k3', 'y0'],
            'lower_bounds': [-np.inf, -np.inf, -np.inf, 0, 0, 0, -np.inf],
            'upper_bounds': [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
            'init_func': 'lambda x_data, y_data: (max(np.abs(y_data)), max(np.abs(y_data)), max(np.abs(y_data)), 1, 0.5, 0.2, 0)'
        },
        {
            'name': 'A->B 2nd order',
            'equation': 'return A / (1 + A * k1 * x) + y0',
            'params': ['A', 'k1', 'y0'],
            'lower_bounds': [-np.inf, 0, -np.inf],
            'upper_bounds': [np.inf, np.inf, np.inf],
            'init_func': 'lambda x_data, y_data: (y_data[0], 1, 0)'
        },
        {
            'name': 'A->B 2nd order (with c0)',
            'equation': 'return A * c0 / (1 + c0 * k1 * x) + y0',
            'params': ['A', 'c0', 'k1', 'y0'],
            'lower_bounds': [-np.inf, 0, 0, -np.inf],
            'upper_bounds': [np.inf, np.inf, np.inf, np.inf],
            'init_func': 'lambda x_data, y_data: (y_data[0], 1, 1, 0)'
        },
        {
            'name': 'A->B, A->B 2nd order',
            'equation': 'return A * k1 / ( (A*k2+k1) * exp(k1 * x) - A*k2) + y0',
            'params': ['A', 'k1', 'k2', 'y0'],
            'lower_bounds': [-np.inf, 0, 0, -np.inf],
            'upper_bounds': [np.inf, np.inf, np.inf, np.inf],
            'init_func': 'lambda x_data, y_data: (y_data[0], 1, 1, 0)'
        },
        {
            'name': 'A->B, A->B 2nd order (with c0)',
            'equation': 'return A * c0*k1 / ((c0*k2+k1) * exp(k1 * x) - c0*k2) + y0',
            'params': ['A', 'c0', 'k1', 'k2', 'y0'],
            'lower_bounds': [-np.inf, 0, 0, 0, -np.inf],
            'upper_bounds': [np.inf, np.inf, np.inf, np.inf, np.inf],
            'init_func': 'lambda x_data, y_data: (y_data[0], 1, 1, 1, 0)'
        },
        {
            'name': 'A->B, C->D (lifetime)',
            'equation': 'return A1 * exp(-x / tau1) + A2 * exp(-x / tau2) + y0',
            'params': ['A1', 'A2', 'tau1', 'tau2', 'y0'],
            'lower_bounds': [-np.inf, -np.inf, 0, 0, -np.inf],
            'upper_bounds': [np.inf, np.inf, np.inf, np.inf, np.inf],
            'init_func': 'lambda x_data, y_data: (y_data[0], y_data[0], 1, 1, 0)'
        },
        {
            'name': 'A->B, C->D (rate constant)',
            'equation': 'return A1 * exp(-k1 * x) + A2 * exp(-k2 * x) + y0',
            'params': ['A1', 'A2', 'k1', 'k2', 'y0'],
            'lower_bounds': [-np.inf, -np.inf, 0, 0, -np.inf],
            'upper_bounds': [np.inf, np.inf, np.inf, np.inf, np.inf],
            'init_func': 'lambda x_data, y_data: (y_data[0], y_data[0], 1, 1, 0)'
        }
    ]

    def __init__(self):
        self.attr = Settings._get_attributes()

        # delete settings that are project independent

        del self.attr['fitting_models']
        del self.attr['recent_project_filepaths']
        del self.attr['import_files_dialog_path']
        del self.attr['open_project_dialog_path']
        del self.attr['save_project_dialog_path']

    def set_settings(self):

        for key, value in self.attr.items():
            setattr(Settings, key, value)

        del self

    @classmethod
    def set_default_settings(cls):
        if Settings._inst_default_settings is not None:
            Settings._inst_default_settings.set_settings()

    @classmethod
    def _get_attributes(cls):
        members = vars(cls)
        filtered = {attr: key for attr, key in members.items() if not attr.startswith('_') and
                    not callable(getattr(Settings, attr))}
        return filtered

    @classmethod
    def save(cls):

        # if Settings._inst_default_settings is None:
        #     Settings._inst_default_settings = Settings()

        sett_dict = cls._get_attributes()

        try:

            with open(cls._filepath, "w") as file:
                json.dump(sett_dict, file, sort_keys=False, indent=4, separators=(',', ': '))

        except Exception as ex:
            Logger.message("Error saving settings to file {}. Error message:\n{}".format(
                Settings._filepath, ex.__str__()))

    @classmethod
    def load(cls):
        # basically static constructor method, set up default settings instance
        if Settings._inst_default_settings is None:
            Settings._inst_default_settings = Settings()

        try:
            with open(cls._filepath, "r") as file:
                data = json.load(file)

            # instance = object.__new__(cls)

            for key, value in data.items():
                setattr(cls, key, value)

        except Exception as ex:
            Logger.message("Error loading settings from file {}, setting up default settings. Error message:\n{}".format(
                Settings._filepath, ex.__str__()))

# Settings.save()
#
# inst = Settings()
#
# # inst.save_project_dialog_path = "asaps pasdasdpa sdpaosapsdo apsdopaso pasokd aosdkaps paskdpaoskdpaoskdpaos"
#
# # inst.attr['normalize_range'] = (0, 0)
#
# inst.set_settings()
#
# print(Settings._get_attributes())
