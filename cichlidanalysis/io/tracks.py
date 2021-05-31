import glob
import os
import copy
import datetime

import numpy as np
import pandas as pd


def load_track(csv_file_path):
    """Takes file path, loads  the csv track, computes speed from this, returns  both
    """
    track_internal = np.genfromtxt(csv_file_path, delimiter=',')

    if track_internal.size == 0:
        # if empty return empty
        displacement_internal = []
        track_internal = []
        return displacement_internal, track_internal
    else:
        # find displacement
        b = np.diff(track_internal[:, 1])
        c = np.diff(track_internal[:, 2])
        displacement_internal = np.sqrt(b ** 2 + c ** 2)
        return displacement_internal, track_internal


def remove_tags(input_files, remove=["exclude", "meta.csv", "als.csv"]):
    """ Input is a list of strings, this function will  go through each of the strings and remove any which have any
    of the tags
    >>> remove_tags(["test_exc_.csv", "test_meta.csv", "b_als.c", "file_als_.csv"], ["exc", "meta.csv", "als.c"])
    ['file_als_.csv']
    >>> remove_tags(["test_exc_.csv", "test_meta.csv", "b_als.c", "file_als.csv"], ["exc", "meta.csv", "als.c"])
    []
    >>> remove_tags(["test_exc_.csv", "test_meta.csv", "b_als.c", "file_als_.csv", "p"], ["exc", "meta.csv", "als.c"])
    ['file_als_.csv', 'p']
    >>> remove_tags(["a.csv", "b.csv", "b_als.c", "c.csv", "d"], ["excude", "abc.csv", "not.c"])
    ['a.csv', 'b.csv', 'b_als.c', 'c.csv', 'd']
    """
    # remove files with  certain tags
    files = []
    for file_a in input_files:
        counting = 0
        for tag in remove:
            if tag in file_a:
                counting += 1
        if counting == 0:
            files.append(file_a)
    files.sort()
    return files


def get_latest_tracks(folder_path, file_pattern):
    os.chdir(folder_path)
    all_files = glob.glob("*{}*.csv".format(file_pattern))

    # remove files with  certain tags
    files = remove_tags(all_files, ["exclude", "meta.csv", "als.csv"])

    # prioritise cleaned version of these files
    file_clean = glob.glob("*{}_cleaned.csv".format(file_pattern))
    file_clean.sort()

    for file_clean in file_clean:
        for file in files:
            if file[0:-4] == file_clean[0:-12]:
                pos = files.index(file)
                replaced = files.pop(pos)
                files.insert(pos, file_clean)

    return file_clean, files


def extract_tracks_from_fld(folder, file_ending):
    """Asks you for a folder path which is the fish roi, find all csv files in the folder which have the
    "file_ending". Will exclude all files with "exclude". Replaces tracks with "Range" and "Cleaned"
    Returns appended tracks and speed for that fish, Timestamp in nS
    """
    track_full = np.empty([0, 4])

    file_cleaned, files = get_latest_tracks(folder, file_ending)

    # prioritise range version of these files
    files_split = glob.glob("*Range*_.csv")
    files_split.sort()

    movie_nums = []
    for file_split in files_split:
        if file_split in files:
            movie_nums.append(file_split.split("_")[1])

    # need to extract the movie name which was spilt and replace it with all the "range" files
    for movie in set(movie_nums):
        all_with_movie_num = glob.glob("*_{}_*.csv".format(movie))
        select_with_movie_num = copy.copy(all_with_movie_num)
        movies_with_range = []
        for file_with_movie_num in all_with_movie_num:
            if file_with_movie_num.find("Range") > -1:
                movies_with_range.append(select_with_movie_num.pop(select_with_movie_num.index(file_with_movie_num)))
        movies_with_range.sort()

        #but also remove the ones with "exclude" tag
        select_with_movie_num = remove_tags(select_with_movie_num, ["exclude"])

        if len(select_with_movie_num) > 1:
            print("two  options for replacement for split movie... exiting")
            return False

        replacing_movie_idx = files.index(select_with_movie_num[0])
        files.pop(replacing_movie_idx)
        for inserting in movies_with_range:
            files.insert(replacing_movie_idx, inserting)
            replacing_movie_idx += 1
    files_to_load = list(dict.fromkeys(files))
    files_to_load.sort()

    for file in files_to_load:
        print(file)
        na, track_single = load_track(os.path.join(folder, file))
        if len(track_single) > 0:
            track_full = np.append(track_full, track_single, axis=0)

    print("All files loaded")

    # find displacement
    b = np.diff(track_full[:, 1])
    c = np.diff(track_full[:, 2])
    speed_full = np.sqrt(b ** 2 + c ** 2)

    return track_full, speed_full


def adjust_old_time_ns(recname, timevector_ns):
    """ some older recordings use 7:30-19:30 light  instead of 7-19 light, subtract 30min from timestamps"""
    thirty_min_ns = 30*60*1000000000
    # finding old recordings
    if int(recname[4:12]) < 20201127:
        adjusted_time_ns = timevector_ns - thirty_min_ns
        print("old recording from before 20201127 so adjusting time from 7.30am to 7am")
    else:
        adjusted_time_ns = timevector_ns

    return adjusted_time_ns


def adjust_old_time(recname, timestamps):
    """ some older recordings use 7:30-19:30 light  instead of 7-19 light, subtract 30min from timestamps"""
    thirty_minutes = datetime.timedelta(minutes=30)

    # finding old recordings
    if int(recname[4:12]) < 20201127:
        adjusted_timestamps = timestamps - thirty_minutes
        print("old recording from before 20201127 so adjusting time from 7.30am to 7am")
    else:
        adjusted_timestamps = timestamps

    return adjusted_timestamps


def load_als_files(folder, suffix="*als.csv"):
    os.chdir(folder)
    files = glob.glob(suffix)
    files.sort()
    first_done = 0

    for file in files:
        if first_done:
            data_s = pd.read_csv(os.path.join(folder, file), sep=',')
            # Removed index_col=0, as is giving Type error ufunc "isnan'
            print("loaded file {}".format(file))
            data_s['FishID'] = file[0:-8]
            # data_s['species'] = file[0:-8].split("_")[3]
            # data_s['sex'] = file[0:-8].split("_")[4]
            data_s['ts'] = adjust_old_time(file, pd.to_datetime(data_s['tv_ns'], unit='ns'))
            data = pd.concat([data, data_s])

        else:
            # initiate data frames for each of the fish, beside the time series,
            # also add in the species name and ID at the start
            data = pd.read_csv(os.path.join(folder, file), sep=',')
            # Removed index_col=0, as is giving Type error ufunc "isnan'
            print("loaded file {}".format(file))
            data['FishID'] = file[0:-8]
            # data['species'] = file[0:-8].split("_")[3]
            # data['sex'] = file[0:-8].split("_")[4]

            data['ts'] = adjust_old_time(file, pd.to_datetime(data['tv_ns'], unit='ns'))
            first_done = 1

    # workaround to deal with Removed index_col=0, as is giving Type error ufunc "isnan'
    data.drop(data.filter(regex="Unname"), axis=1, inplace=True)
    # also change how the csv is saved in run_fish_als.py

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
    # also change how the csv is saved in run_fish_als.py

    print("All down sampled als.csv files loaded")
    return data


def save_fishtracks_30m(fish_tracks_30m, rootdir):
    """ Save out combined fish_tracks_30m

    :param fish_tracks_30m:
    :param rootdir:
    :return:
    """
    date = datetime.datetime.now().strftime("%Y%m%d")
    # save out 30m data (all adjusted to 7am-7pm)
    fish_tracks_30m.to_csv(os.path.join(rootdir, "combined_{0}_als_30m.csv".format(date)))
    print("Finished saving out 30min data")


if __name__ == '__main__':
    import doctest
    doctest.testmod()
