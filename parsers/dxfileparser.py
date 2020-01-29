from spectrum import Spectrum
import numpy as np
from parsers.genericparser import GenericParser

from logger import Logger


class DXFileParser(GenericParser):

    def __init__(self, filepath=None, str_data=None, delimiter=' ', decimal_sep='.',
                 dx_import_spectra_name_from_filename=False, dx_if_title_is_empty_use_filename=True):
        super(DXFileParser, self).__init__(filepath, str_data, delimiter, decimal_sep)

        self.dx_import_spectra_name_from_filename = dx_import_spectra_name_from_filename
        self.dx_if_title_is_empty_use_filename = dx_if_title_is_empty_use_filename

    def parse(self, name=''):

        it = self.__iter__()  # get line iterator

        data = []

        # read first header title line
        title_header = it.__next__().strip()
        if name is None:
            try:
                name = title_header.split('=')[1][1:]
            except:
                pass

        # if the title field in the dx file is empty, or some error occurred while reading it,
        # use user defined settings

        if self.dx_import_spectra_name_from_filename:
            name = self.name_of_file

        if self.dx_if_title_is_empty_use_filename:
            name = self.name_of_file if name == '' else name

        # skip 17 lines of header
        for i in range(17):
            it.__next__()

        count = 0

        # read the data itself
        for line in it:
            line = line.strip()
            split = line.split(self.delimiter)
            if len(split) < 2:
                continue

            wavelength = self.float_try_parse(split[0])
            absorbance = self.float_try_parse(split[1])
            if wavelength is None or absorbance is None:
                count += 1
                continue

            data.append((wavelength, absorbance))

        if len(data) < 2:
            return

        # one row will be skipped every time - it is the end line of DX file
        if count > 1:
            Logger.console_message("Warning, {} rows were skipped due to unsuccessful "
                                   "float value parsing of DX file. Some data may be lost.\nFile: {}".format(count - 1, self.filepath))

        sp = Spectrum(np.asarray(data, dtype=np.float64), self.filepath, name)
        return [sp]  # returns a list
