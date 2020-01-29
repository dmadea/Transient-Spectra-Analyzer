
import os
import sys
from abc import abstractmethod
from logger import Logger


# abstract class that every parser must inherits
class GenericParser(object):

    def __init__(self, filepath=None, str_data=None, delimiter='\t', decimal_sep='.'):
        if filepath is None and str_data is None:
            raise ValueError("At least one argument (filepath or str_data) must not be None.")

        if str_data is None:
            self.filepath = filepath
            self.name_of_file = self.get_filename()
            self._data = None
        else:
            self._data = str_data
            self.filepath = None
            self.name_of_file = None

        self.delimiter = delimiter
        self.decimal_sep = decimal_sep

    @abstractmethod
    def parse(self, name=None):
        # it = self.__iter__()  # get iterator

        # for line in it:
        #     # Do stuff...
        pass

    def __iter__(self):
        """Return an line iterator that iterates through loaded data, either opened file, or string data"""
        if self._data is None:

            encoding = self.get_encoding()

            if encoding is None:
                raise Exception("Problem with encoding of the file.\n{}".format(self.filepath))

            return open(self.filepath, 'r', encoding=encoding, newline='').__iter__()
        else:
            return iter(self._data.splitlines(keepends=True))

    def get_encoding(self):
        if self.filepath is None:
            return None

        default_encoding = sys.getfilesystemencoding()
        # utf-8-sig is better, since it does not matter if BOM is on the begining of the file or not
        encoding = 'utf-8-sig' if default_encoding == 'utf-8' else default_encoding
        try:
            with open(self.filepath, 'r', encoding=encoding) as f:
                f.readline()
            return encoding
        except UnicodeDecodeError:
            encoding = 'utf-16'
            try:
                with open(self.filepath, 'r', encoding=encoding) as f:
                    f.readline()
                return encoding
            except Exception as ex:
                # Logger.console_message(ex.__str__())
                return None
        except Exception as ex:
            # Logger.console_message(ex.__str__())
            return None


    def get_filename(self):
        # head = os.path.split(path_to_file)[0]
        tail = os.path.split(self.filepath)[1]

        return os.path.splitext(tail)[0]  # without extension

    def float_try_parse(self, num):
        try:
            return float(num.replace(self.decimal_sep, '.').strip())
        except ValueError:
            return None
