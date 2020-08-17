import json
from logger import Logger


class Settings(object):
    """Static class containing user settings with static functions for saving and loading to/from json file."""

    # settings that starts with '_' will not be saved into a file
    # static variables

    _filepath = "config.json"
    _inst_default_settings = None  # default settings instance

    # Transient Spectra Analyzer version
    __version__ = "0.1.0"
    __last_release__ = "17.08.2020"

    # --------SETTINGS----------


    # files dialog last paths

    import_files_dialog_path = ""
    open_project_dialog_path = ""
    save_project_dialog_path = ""
    export_spectra_as_dialog_path = ""
    export_spectra_as_dialog_ext = '.txt'
    export_spectra_as_dialog_delimiter = '\t'
    export_spectra_as_dialog_decimal_sep = '.'

    # recent projects filepaths

    recent_project_filepaths = []

    gui_settings_last_tab_index = 0

    def __init__(self):
        self.attr = Settings._get_attributes()

        # delete settings that are project independent

        del self.attr['export_spectra_as_dialog_path']
        del self.attr['export_spectra_as_dialog_ext']
        del self.attr['export_spectra_as_dialog_delimiter']
        del self.attr['export_spectra_as_dialog_decimal_sep']
        del self.attr['recent_project_filepaths']
        del self.attr['import_files_dialog_path']
        del self.attr['open_project_dialog_path']
        del self.attr['save_project_dialog_path']

    def set_settings(self):
        """Sets static settings from this instance object (project settings)."""

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
            Logger.message(
                "Error loading settings from file {}, setting up default settings. Error message:\n{}".format(
                    Settings._filepath, ex.__str__()))
            Settings.save()

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
