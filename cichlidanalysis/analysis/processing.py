import copy

import numpy as np
import pandas as pd


def smooth_speed(speed, win_size=2):
    df = pd.DataFrame(speed)
    smooth_speed = (df.rolling(window=win_size, min_periods=1).mean()).values
    return smooth_speed


def neg_values(array):
    new_array = copy.copy(array)
    for index, element in enumerate(array):
        if not np.isnan(element):
            new_array[index] = -abs(element)
    return new_array


def remove_high_spd(speed_raw):
    # takes the raw speed and will replace any value over threshold with the mean of the values < threshold -5:+5.
    # This is to get rid of the massive speed jumps caused by roi jumps
    speed_t = copy.copy(speed_raw)
    threshold = np.nanpercentile(speed_raw, 95) * 2
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
    return speed_t


def coord_smooth(x_coords, y_coords, win_size):
    # type: # (np.array, np.array, int) -> np.array, np.array
    df_x = pd.DataFrame(x_coords)
    smooth_x = (df_x.rolling(window=win_size).apply(lambda x: median(x)[0])).values

    df_y = pd.DataFrame(y_coords)
    smooth_y = (df_y.rolling(window=win_size).apply(lambda x: median(x)[0])).values
    return smooth_x, smooth_y


def binner(input_data, bin_width, axis_d):
    """ takes a 1D np array and bins it with bin size of bin_width, input needs to be with data in dim 0 e.g.
    [10,0] or [10,] """
    # input must be np.array, if pandas.series change it to np.array
    if isinstance(input_data, pd.core.series.Series):
        print("correcting input from pd.series to np.array")
        input_data = input_data.to_numpy()

    if input_data.shape[0] % bin_width != 0:
        rest = input_data.shape[0] % bin_width
        output_data = np.reshape(input_data[0:-rest], [int(input_data[0:-rest].shape[0] / bin_width), bin_width])
    else:
        output_data = np.reshape(input_data, [int(input_data.shape[0] / bin_width), bin_width])

    output_data_mean = np.nanmean(output_data, axis=axis_d)

    return output_data_mean, output_data


def int_nan_streches(data_input):
    """ finds nan streches and interporlates the strech, pads start as continuous"""
    data = copy.copy(data_input)
    # data = speed_full[99000:103000]

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
    # takes the raw speed and will replace any value over threshold with the mean of the values < threshold -5:+5.
    # This is to get rid of the massive speed jumps caused by roi jumps
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
    """

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
    """ Calculates the  daily average, note must be organised with datetime ass index and columns as different
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


if __name__ == "__main__":
    import doctest
    doctest.testmod()
