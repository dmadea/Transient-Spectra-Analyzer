
import csv
from parsers.generalparser import GeneralParser


class CSVFileParser(GeneralParser):

    def __init__(self, filepath=None, str_data=None, delimiter=',', decimal_sep='.', doublequote=True, skipinitialspace=True):
        super(CSVFileParser, self).__init__(filepath, str_data, delimiter, decimal_sep)

        self.doublequote = doublequote
        self.skipinitialspace = skipinitialspace

    def line2list_iterator(self):
        # get line iterator, return parsed list iterator, like eg. ['190', '1.5', '1.3', 'some value']
        it = self.__iter__()
        return csv.reader(it, doublequote=self.doublequote,
                          skipinitialspace=self.skipinitialspace,
                          delimiter=self.delimiter)
