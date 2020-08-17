import numpy as np
import os
from LFP_matrix import LFP_matrix
from misc import find_nearest_idx
import glob


def float_try_parse(num):
    try:
        return float(num)
    except ValueError:
        return 0

def parse_file(path_to_file):
    file, ext = os.path.splitext(path_to_file)
    ext = ext.lower()
    head = os.path.split(path_to_file)[0]
    tail = os.path.split(path_to_file)[1]
    name_of_file = os.path.splitext(tail)[0]  #without extension
    # CSV file
    delimiter = ','
    transpose = False

    if ext == '.csv':
        delimiter = ','

    elif ext.startswith('.a'):  # a0-an from femto
        delimiter = '\t'

    else:
        delimiter = '\t'
        transpose = True
    try:
        data = np.genfromtxt(path_to_file, dtype=np.float32, skip_header=0, delimiter=delimiter, filling_values=0,
                         autostrip=True)
    except ValueError:
        data = np.genfromtxt(path_to_file, dtype=np.float32, skip_header=0, delimiter=delimiter, filling_values=0,
                             autostrip=True, encoding='utf16')

    data = np.nan_to_num(data)

    # with open(path_to_file, mode='r') as tsf:
    #     buffer = [[float_try_parse(num.strip()) for num in line.strip().split(delimiter)]
    #               for line in tsf]

    # data = np.asarray(buffer)
    lfp_matrix = LFP_matrix(data.T if transpose else data, path_to_file, 'name...')
    return lfp_matrix

def get_spectra(filepath, lin_space, count):
    # l_space = np.linspace(0.05, 9, 50)
    lfp_matrix = parse_file(filepath)

    # lfp_matrix.slice(np.s_[:], np.s_[2:])
    # print(lfp_matrix.wavelengths)
    # print(lfp_matrix.times)
    # print(lfp_matrix.data)

    print(lfp_matrix.Y.shape)
    t_len = lfp_matrix.get_time_dimension()
    min, max = lfp_matrix.times[0], lfp_matrix.times[t_len - 1]
    print(t_len, min, max)
    idx_buffer = [find_nearest_idx(lfp_matrix.times, i) for i in lin_space]
    # count = 50

    # DiMe-CP+BP, 395nm-40us, 128 flashes

    list = []
    # number of spectra before and after the chosen spectrum that average will be calculated from
    if count == 0:
        slice_matrix = lfp_matrix.Y[idx_buffer, :]
    else:
        for i in idx_buffer:
            avrg_slice_matrix = lfp_matrix.Y[i - count:i + count, :]
            avrg_array = np.average(avrg_slice_matrix, axis=0)
            list.append(avrg_array)
        slice_matrix = np.asarray(list)
    # slice_matrix = lfp_matrix.data[idx_buffer, :]
    time_slice = lfp_matrix.times[idx_buffer]
    ret_matrix = np.vstack((time_slice, slice_matrix.T))
    # wavelengths = ['Wavelength'] + list(lfp_matrix.wavelengths)
    wavelengths = np.concatenate([[0], lfp_matrix.wavelengths])
    ret_matrix = np.hstack((wavelengths.reshape(wavelengths.shape[0], 1), ret_matrix))
    str = lfp_matrix.to_string(ret_matrix)
    new_filename = os.path.join(os.path.split(filepath)[0], os.path.splitext(filepath)[0] + '-choosed_spectra.txt')
    with open(new_filename, 'w') as tsf:
        tsf.write(str)




if __name__ == "__main__":

    dir = r"C:\Users\Dominik\Documents\MUNI\Organic Photochemistry\Projects\2018-19_Japan-C-C bond homolysis\LFP\2019-07-11 Xan + CP2\\"

    for filepath in glob.glob(dir + "*.csv"):
        tail = os.path.split(filepath)[1]

        # lin_space = np.linspace(0.1, 15, 70)
        lin_space = np.arange(0, 2, step=0.05)

        count = 0

        if "spectra" in tail:
            print("parsing {}...".format(tail))
            get_spectra(filepath, lin_space, count)
            print("Done")








