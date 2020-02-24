
from spectrum import Spectrum
from parsers.genericparser import GenericParser
import numpy as np
# from abc import abstractmethod



class GeneralParser(GenericParser):

    def __init__(self, filepath=None, str_data=None, delimiter='\t', decimal_sep='.', ):
        super(GeneralParser, self).__init__(filepath, str_data, delimiter, decimal_sep)

        self._spectra_buffer = []


    def float_try_parse(self, num_list):
        return [GenericParser.float_try_parse(self, num) for num in num_list]

        # if isinstance(num_list, list):
        #     return [GenericParser.float_try_parse(num.replace(self.decimal_sep, '.').strip()) for num in num_list]
        # else:
        #     return GenericParser.float_try_parse(num_list.replace(self.decimal_sep, '.').strip())

    def _parse_data(self, data, names, error_columns):
        if len(data) < 2 or len(error_columns) < 2:
            data.clear()
            names.clear()
            return False

        col_count = len(data[0])
        row_count = len(data)

        if len(names) < len(data[0]):  # resize names of spectra
            names += [''] * (col_count - len(names))

        spectra = []

        # print(error_columns, col_count, row_count)

        for i in range(1, col_count):
            # skip the wrong column
            if error_columns[i]:
                continue

            sp_data = []
            group_name = self.get_filename() if self.filepath is not None else ''

            for j in range(row_count):
                sp_data.append((data[j][0], data[j][i]))

            sp = Spectrum(np.asarray(sp_data, dtype=np.float32), self.filepath, names[i], group_name)

            spectra.append(sp)

        if len(spectra) == 1:
            spectra[0].name = self.name_of_file if self.name_of_file is not None else spectra[0].name

        self._spectra_buffer.append(spectra[0] if len(spectra) == 1 else spectra)

        return True

    def _fill_error_values(self, error_columns, l_values):
        row_count = len(l_values)
        l_e = len(error_columns)
        if row_count > l_e:  # resize list to the size of l_values
            error_columns += [False] * (row_count - l_e)

        for i in range(row_count):
            if l_values[i] is None:
                error_columns[i] = True

        return error_columns

    def line2list_iterator(self):
        """Iterator method that can be overridden. Get iterator that will iterate through a parsed line into a LIST"""

        for line in self.__iter__():
            split_line = line.strip().split(self.delimiter)

            yield split_line

    def parse(self, name=None):

        it = self.line2list_iterator()

        data = []
        names = []
        error_columns = []

        for line in it:
            # line is a parsed list - ['entry1', 'entry2', .... ]

            l = len(line)

            # if l == 0:
            #     self._parse_data(data, names, error_columns)
            #     error_columns = []
            #     continue

            if l < 2:
                self._parse_data(data, names, error_columns)
                error_columns = []
                continue

            l_values = self.float_try_parse(line)
            # if l >= len(error_columns):
            error_columns = self._fill_error_values(error_columns, l_values)

            if l_values[0] is None:
                self._parse_data(data, names, error_columns)
                names = line
                error_columns = []

                continue

            data.append(l_values)

        self._parse_data(data, names, error_columns)

        return self._spectra_buffer

