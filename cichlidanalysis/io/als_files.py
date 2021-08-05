import glob
import os
import datetime as dt

import pandas as pd

from cichlidanalysis.analysis.processing import remove_cols


def load_als_files(folder, suffix="*als.csv"):
    os.chdir(folder)
    files = glob.glob(suffix)
    files.sort()
    first_done = 0

    for file in files:
        if first_done:
            data_s = pd.read_csv(os.path.join(folder, file), sep=',', error_bad_lines=False, warn_bad_lines=True)
            # data_s = pd.read_csv(os.path.join(folder, file), sep='/')
            # str.split(',' expand = T)
            # #, error_bad_lines=False, warn_bad_lines=True)
            print("loaded file {}".format(file))
            data_s['FishID'] = file[0:-8]
            data_s['ts'] = adjust_old_time(file, pd.to_datetime(data_s['tv_ns'], unit='ns'))
            data = pd.concat([data, data_s])

        else:
            # initiate data frames for each of the fish, beside the time series,
            # also add in the species name and ID at the start
            data = pd.read_csv(os.path.join(folder, file), sep=',', error_bad_lines=False, warn_bad_lines=True)
            print("loaded file {}".format(file))
            data['FishID'] = file[0:-8]
            data['ts'] = adjust_old_time(file, pd.to_datetime(data['tv_ns'], unit='ns'))
            first_done = 1

    # workaround to deal with Removed index_col=0, as is giving Type error ufunc "isnan'
    data.drop(data.filter(regex="Unname"), axis=1, inplace=True)
    data = data.drop(data[data.ts < dt.datetime.strptime("1970-1-2 00:00:00", '%Y-%m-%d %H:%M:%S')].index)
    data = data.reset_index(drop=True)
    data = remove_cols(data, ['vertical_pos', 'horizontal_pos', 'speed_bl', 'activity'])

    print("All als.csv files loaded")
    return data


def load_ds_als_files(folder, suffix="*als.csv"):
    os.chdir(folder)
    files = glob.glob(suffix)
    files.sort()
    first_done = 0

    for file in files:
        if first_done:
            data_s = pd.read_csv(os.path.join(folder, file), sep=',')
            print("loaded file {}".format(file))
            data = pd.concat([data, data_s])

        else:
            # initiate data frames for each of the fish, beside the time series,
            data = pd.read_csv(os.path.join(folder, file), sep=',')
            print("loaded file {}".format(file))
            first_done = 1

    # workaround to deal with Removed index_col=0, as is giving Type error ufunc "isnan'
    data.drop(data.filter(regex="Unname"), axis=1, inplace=True)

    print("All down sampled als.csv files loaded")
    return data


def load_vertical_rest_als_files(folder, suffix="*als_vertical_pos_hist_rest-non-rest.csv"):
    os.chdir(folder)
    files = glob.glob(suffix)
    files.sort()
    first_done = 0

    for file in files:
        if first_done:
            data_s = pd.read_csv(os.path.join(folder, file), sep=',')
            print("loaded file {}".format(file))
            data = pd.concat([data, data_s])

        else:
            # initiate data frames for each of the fish, beside the time series,
            data = pd.read_csv(os.path.join(folder, file), sep=',')
            print("loaded file {}".format(file))
            first_done = 1

    # workaround to deal with Removed index_col=0, as is giving Type error ufunc "isnan'
    data.drop(data.filter(regex="Unname"), axis=1, inplace=True)

    print("All down sampled als.csv files loaded")
    return data


def load_vertical_rest_als_files(folder, suffix="*als_vertical_pos_hist_rest-non-rest.csv"):
    """ load *als_vertical_pos_hist_rest-non-rest.csv files and add species name

    :param folder:
    :param suffix:
    :return:
    """
    os.chdir(folder)
    files = glob.glob(suffix)
    files.sort()
    first_done = 0

    for file in files:
        if first_done:
            data_s = pd.read_csv(os.path.join(folder, file), sep=',')
            print("loaded file {}".format(file))
            data_s['species'] = file[0:-40]
            data = pd.concat([data, data_s])

        else:
            # initiate data frames for each of the species
            data = pd.read_csv(os.path.join(folder, file), sep=',')
            print("loaded file {}".format(file))
            data['species'] = file[0:-40]
            first_done = 1

    # workaround to deal with Removed index_col=0, as is giving Type error ufunc "isnan'
    data.drop(data.filter(regex="Unname"), axis=1, inplace=True)

    print("All down sampled als_vertical_pos_hist_rest-non-rest.csv files loaded")
    return data
