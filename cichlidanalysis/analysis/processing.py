import copy

import numpy as np
import pandas as pd


def smooth_speed(speed, win_size=2):
    """ Rolling average of win_size for the speed array

    :param speed: array of speed to smooth
    :param win_size: over how many frames to smooth over
    :return: smoothed speed
    """
    df = pd.DataFrame(speed)
    smooth_speed = (df.rolling(window=win_size, min_periods=1).mean()).values
    return smooth_speed


def neg_values(array):
    """ make values in array negative (useful for some plotting)

    :param array: array to make negative
    :return: negative array
    """
    new_array = copy.copy(array)
    for index, element in enumerate(array):
        if not np.isnan(element):
            new_array[index] = -abs(element)
    return new_array


def interpolate_nan_streches(data_input):
    """ finds nan streches and interporlates the strech, pads start as continuous"""
    data = copy.copy(data_input)

    # taking care of ends
    # determine if data started with nan or not
    if np.isnan(data[0]):
        # find where the data starts
        j = 0
        found_start = 0
        while found_start == 0:
            if ~np.isnan(data[j]):
                start_notnan = j
                found_start = 1
            j = j + 1
        data[0:start_notnan] = data[start_notnan]

    if np.isnan(data[-1]):
        # find where the last nan strech starts
        j = -1
        found_start = 0
        while found_start == 0:
            if ~np.isnan(data[j]):
                start_notnan = j
                found_start = 1
            j = j - 1
        data[start_notnan:] = data[start_notnan]

    # finds indices of NaNs
    data_nans = np.isnan(data) * 1

    # cleaning data of spurious positions
    spurious = [1, 1, 1, 0, 1, 1, 1]

    for idx, data_point in enumerate(data_nans[0:-2], start=2):
        if np.array_equiv(spurious, data_nans[idx - 3:idx + 4]):
            data_nans[idx - 3:idx + 4] = [1, 1, 1, 1, 1, 1, 1]

    # finds start or finsh of NaN streches
    changes = np.diff(data_nans, axis=0)

    # added 1 to bout_start as otherwise it is the last timepoint that was below the threshold.
    # Also did it to ends so a peak of one timepoint would have a length of 1.
    bout_start = np.where(changes == 1)[0] + 1
    bout_end = np.where(changes == -1)[0] + 1

    for idx, strech in enumerate(bout_start):
        data[bout_start[idx]:bout_end[idx]] = np.interp(np.arange(bout_start[idx], bout_end[idx]),
                                                        [bout_start[idx] - 1, bout_end[idx]],
                                                        [data[bout_start[idx] - 1], data[bout_end[idx]]])

    return data


def remove_high_spd_xy(speed_raw, x, y):
    """ takes the raw speed and will replace any value over threshold with the mean of the values < threshold -5:+5.
    This is to get rid of the massive speed jumps caused by roi jumps

    :param speed_raw:
    :param x:
    :param y:
    :return:
    """

    speed_t = copy.copy(speed_raw)
    x_t = copy.copy(x)
    y_t = copy.copy(y)

    # threshold = np.nanpercentile(speed_raw, 95) * 2
    threshold = 200
    ind_high = np.where(speed_raw > threshold)[0]

    # for each index > threshold find the values on the side and take the average of all of those values < threshold
    for index_n in range(0, ind_high.shape[0]):
        win_min = ind_high[index_n] - 5
        win_max = ind_high[index_n] + 5

        if win_min < 0:
            win_min = 0
        if win_max > speed_t.shape[0]:
            win_max = speed_t.shape[0]

        values = speed_t[win_min:win_max]
        speed_t[ind_high[index_n]] = np.nanmean(values[values < threshold])
        x_t[ind_high[index_n]] = np.nanmean(x_t[win_min:win_max][values < threshold])
        y_t[ind_high[index_n]] = np.nanmean(y_t[win_min:win_max][values < threshold])
    return speed_t, x_t, y_t


def threshold_data(speed, threshold):
    """ gives back a np.array with 1 for where the input array is above the threshold. Puts back NaNs

    :param speed: np.array or pd.series
    :param threshold: float
    :return: super_threshold_indices: np.array

    >>> threshold_data(np.array([0,0,0,3,3,0,1,0]),2)
    array([0., 0., 0., 1., 1., 0., 0., 0.])
    >>> threshold_data(np.array([0,0,0,2,2,0,1,0]),2)
    array([0., 0., 0., 0., 0., 0., 0., 0.])
    """
    # apply movement threshold to define active bouts
    super_threshold_indices = (speed > threshold).astype(np.float)
    super_threshold_indices[np.isnan(speed)] = np.nan
    super_threshold_indices = np.array(super_threshold_indices)
    super_threshold_indices.resize(super_threshold_indices.shape[0],)

    # plt.figure()
    # plt.imshow(np.expand_dims(super_threshold_indices, axis=0), extent=[0, 54000/30, 0, 100])

    return super_threshold_indices


def add_col(df, col_str, fish_IDs_i, meta_i):
    """ Adds column to fish_tracks type data from meta data

    :param df: fish_tracks
    :param col_str: column to add back e.g. "species"
    :param fish_IDs_i:
    :param meta_i:
    :return:
    """
    if col_str in meta_i.index:
        if col_str not in df.columns:
            df[col_str] = 'blank'
            for fish in fish_IDs_i:
                df.loc[df['FishID'] == fish, col_str] = meta_i.loc[col_str, fish]
        else:
            print("column {} already in given dataframe".format(col_str))
    else:
        print("column {} doesn't exist in meta".format(col_str))


def feature_daily(averages_feature):
    """ Calculates the  daily average, note must be organised with datetime as index and columns as different
    fish/species"""

    # get time of day so that the same time of day for each fish can be averaged
    averages_feature['time_of_day'] = averages_feature.apply(lambda row: str(row.name)[11:16], axis=1)
    aves_ave_feature = averages_feature.groupby('time_of_day').mean()
    return aves_ave_feature


def standardise_cols(input_pd_df):
    """ Calculate z-scores for every column"""

    first = 1
    cols = input_pd_df.columns
    for col in cols:
        col_zscore = col + '_zscore'
        if first:
            output_pd_df = ((input_pd_df[col] - input_pd_df[col].mean()) / input_pd_df[col].std()).to_frame().\
                rename(columns={'spd_mean': col_zscore})
            first = 0
        else:
            output_pd_df[col_zscore] = (input_pd_df[col] - input_pd_df[col].mean()) / input_pd_df[col].std()

    return output_pd_df


def remove_cols(fish_tracks_i, remove):
    """ removing cols from fish_tracks

    :param fish_tracks_i: fish_tracks
    :param remove: list of ccolumn names to remove
    :return: fish_tracks
    """
    for remove_name in remove:
        if remove_name in fish_tracks_i.columns:
            fish_tracks_i = fish_tracks_i.drop(remove_name, axis=1)
            print("old track, removed {}".format(remove_name))
    return fish_tracks_i


def norm_hist(input_d):
    """ Normalise input by total number e.g. fraction"""
    input_d_norm = input_d / sum(input_d)
    return input_d_norm


def add_daytime(fish_df, time_m_names, times_m_dict):
    """ Add daytime column with given time_m_names, works on fish_tracks and fish_bouts if they have a time_of_day_m
    column

    :param fish_df:
    :param time_m_names:
    :param times_m_dict:
    :return:
    """
    fish_df['daytime'] = "night"
    for epoque_n, epoque in enumerate(time_m_names[0:-1]):
        fish_df.loc[(fish_df.time_of_day_m > times_m_dict[epoque]) &
                            (fish_df.time_of_day_m < times_m_dict[time_m_names[epoque_n + 1]]), 'daytime'] = epoque
    return fish_df


def ave_daily_fish(fish_tracks_30m, fish, measure):
    """ To find the daily average for something (measure e.g. 'rest' or 'movement')

    :param fish_tracks_30m:
    :param fish:
    :param measure:
    :return: daily_ave, daily_ave_std, daily_ave_total
    """
    days = fish_tracks_30m[fish_tracks_30m.FishID == fish][[measure, 'ts']]
    days = days.set_index('ts')

    # get time of day so that the same time of day for each day can be averaged
    days['time_of_day'] = days.apply(lambda row: str(row.name)[11:16], axis=1)
    daily_ave = days.groupby('time_of_day').mean()
    daily_ave_std = days.groupby('time_of_day').std()
    daily_ave_total = sum(daily_ave.iloc[:, 0])/2
    return daily_ave, daily_ave_std, daily_ave_total


def species_feature_fish_daily_ave(fish_tracks_ds_i, species_name, feature):
    """ Gets fish of species "species_name", finds the daily average of the given feature

    :param fish_tracks_ds_i:
    :param species_name:
    :param feature:
    :return: fish_daily_ave_feature
    """
    feature_i = fish_tracks_ds_i.loc[fish_tracks_ds_i.species == species_name, [feature, 'FishID', 'ts']]
    fish_feature = feature_i.pivot(columns='FishID', values=feature, index='ts')
    fish_daily_ave_feature = feature_daily(fish_feature)

    return fish_daily_ave_feature


def fish_tracks_add_day_twilight_night(fish_tracks_ds):
    """ Adds daytime column based off time_of_day_m changes (6-8am and 6-8pm are crepuscular"""

    dcn_times_m = [6*60, 8*60, 18*60, 20*60]

    fish_tracks_ds['daytime'] = "n"
    fish_tracks_ds.loc[(fish_tracks_ds.time_of_day_m >= dcn_times_m[0]) & (fish_tracks_ds.time_of_day_m < dcn_times_m[1])
    , 'daytime'] = "c"
    fish_tracks_ds.loc[(fish_tracks_ds.time_of_day_m >= dcn_times_m[1]) & (fish_tracks_ds.time_of_day_m < dcn_times_m[2])
    , 'daytime'] = "d"
    fish_tracks_ds.loc[(fish_tracks_ds.time_of_day_m >= dcn_times_m[2]) & (fish_tracks_ds.time_of_day_m < dcn_times_m[3])
    , 'daytime'] = "c"
    print("added night and day column")
    return fish_tracks_ds


def add_day_number_fish_tracks(fish_tracks_ds):
    """ Add day number to fish. Time stamp (ts) neds to be in '1970-01-02 00:00:00' format

    :param fish_tracks_ds:
    :return:
    """
    # add new column with day number (starting from 1)
    fish_tracks_ds['day_n'] = fish_tracks_ds.ts.apply(lambda row: int(str(row)[8:10]) - 1)
    return fish_tracks_ds


if __name__ == "__main__":
    import doctest
    doctest.testmod()
