import numpy as np
import pandas as pd

from cichlidanalysis.analysis.processing import threshold_data


def find_bout_start_ends(bout_array):
    """ Takes a np.array of zeros and ones and determines the start/stop of the one streches. Assumes no NaNs.

    :param bout_array:
    :return: bout_start_t, bout_end_t
    """
    # test that the  array has no NaNs
    if max(np.isnan(bout_array)):
        print("NaN in bout_array therefore cannot run bout_speeds")
        return False
    else:
        # determine bout starts and finishes
        changes = np.diff(bout_array, axis=0)

        # added 1 to active_bout_start as otherwise it is the last timepoint that was below the threshold.
        # Also did it to ends so a peak of one timepoint would have a length of 1.
        bout_start = np.asarray(np.where(changes == 1)) + 1
        bout_end = np.asarray(np.where(changes == -1)) + 1

        # determine if array started with a bout
        if bout_array[0] == 1:
            # first bout is ongoing, remove first bout as it is incomplete
            bout_start_t = bout_start[0, ]
            bout_end_t = bout_end[0, 1:]
        else:
            # take all starts (and ends)
            bout_start_t = bout_start[0, ]
            bout_end_t = bout_end[0, ]

        # remove incomplete bouts (e.g. those that do not end), in this case there will be one less end than start
        if bout_start_t.shape != bout_end_t.shape:
            if bout_start_t.shape > bout_end_t.shape:
                bout_start_t = bout_start_t[0:-1]
            else:
                print("something weird with number of bouts?")
                return False

        # determine active inter-bout interval
        bout_lengths = bout_end_t - bout_start_t

        return bout_start_t, bout_end_t, bout_lengths


def find_bout_start_ends_inclusive(bout_array):
    """ Takes a np.array of zeros and ones and determines the start/stop of the one streches. Assumes no NaNs.
    Includes streches which are at the edge

    :param bout_array:
    :return: bout_start, bout_end
    """
    # test that the  array has no NaNs
    if max(np.isnan(bout_array)):
        print("NaN in bout_array therefore cannot run bout_speeds")
        return False
    else:
        # determine bout starts and finishes
        changes = np.diff(bout_array, axis=0)

        # added 1 to active_bout_start as otherwise it is the last timepoint that was below the threshold.
        # Also did it to ends so a peak of one timepoint would have a length of 1.
        bout_start = (np.asarray(np.where(changes == 1)) + 1)[0]
        bout_end = (np.asarray(np.where(changes == -1)) + 1)[0]

        # determine if array ends with a bout
        if bout_array[-1] == 1:
            # if so  add in a end
            bout_end = np.concatenate((bout_end, np.array([len(bout_array)])), axis=0)

        # determine if array started with a bout
        if bout_array[0] == 1:
            # first bout is ongoing, add first bout as it is incomplete
            bout_start = np.concatenate((np.array([0]), bout_start), axis=0)

        return bout_start, bout_end


def bout_speeds(bout_array, speed):
    """ For each bout (1 in array, not a zero, assumes no NaNs in data), find the speed of that bout
    :param bout_array:
    :param speed:
    :return: speed_active, bout_max, bout_speed
    """
    # test that the  array has no NaNs
    if max(np.isnan(bout_array)):
        print("NaN in bout_array therefore cannot run bout_speeds")
        return False
    else:
        # find global speed within active bouts
        speed_active = speed[bout_array > 0.5]

        # find bout starts, ends and lengths
        bout_start, bout_end, bout_lengths = find_bout_start_ends(bout_array)
        bout_number = bout_start.shape[0]

        # for every bout, find the max speed
        bout_max = np.zeros(bout_start.shape[0])
        for bout_n in np.linspace(0, bout_number - 1, bout_number):
            bout_max[int(bout_n)] = np.max(speed[bout_start[int(bout_n)]:(bout_start[int(bout_n)] + bout_lengths[int(bout_n)])])

    return speed_active, bout_max


def triggered_bout_speed(bout_array, speed, pre, post):
    """ for every bout extract the speed  "pre" time points before to "post" time points after.
    :param bout_array
    :param speed
    :param pre
    :param post
    :return: trig_bout_spd
    """
    # test that the  array has no NaNs
    if max(np.isnan(bout_array)):
        print("NaN in bout_array therefore cannot run bout_speeds")
        return False
    else:
        # find bout starts, ends and lengths
        bout_start, bout_end, bout_lengths = find_bout_start_ends(bout_array)
        bout_number = bout_start.shape[0]

        # for every bout extract the speed  "pre" time points before to "post" time points after.
        trig_bout_spd = np.empty([bout_start.shape[0], np.max(bout_lengths)])  # max(bout_lengths)+15]) # fps*10])
        trig_bout_spd[:] = np.nan

        for bout in np.linspace(0, bout_number - 1, bout_number):
            # extract out speed data from "pre" time points before to "post" time points after.
            if ((bout_start[int(bout)] - pre) > 0) & ((bout_start[int(bout)] + post) < speed.shape[0]):
                trig_bout_spd[int(bout), 0:(pre + post)] = (speed[(bout_start[int(bout)] - pre):(bout_start[int(bout)]
                                                                                        + post)]).reshape(pre + post)

        return trig_bout_spd


def find_bouts(speed, threshold):
    """ Finds active and quiescent bouts, including where they start, how long they are etc
    :param speed (smoothed)
    :param threshold: speed threshold to determine active/quiescent
    :return: active_bout_lengths, active_bout_end_t, active_bout_start_t, quiescent_bout_lengths, quiescent_bout_end_t,
           quiescent_bout_start_t, active_bout_max

    assume no NaNs??
    """
    # improvements to do: deal with nans in the middle of data
    # one way to do that would be to break apart blocks at NaNs. So there would be a loop to add in uninterrupted blocks
    # need to keep track and accumulate blocks in same category (e.g. night)

    active_indices = threshold_data(speed, threshold)
    inactive_indices = (active_indices != 1) * 1

    # for active
    active_bout_start, active_bout_end, active_bout_lengths = find_bout_start_ends(active_indices)
    active_speed, active_bout_max = bout_speeds(active_indices, speed)

    # for inactive
    inactive_bout_start, inactive_bout_end, inactive_bout_lengths = find_bout_start_ends(inactive_indices)
    inactive_speed, inactive_bout_max = bout_speeds(inactive_indices, speed)

    return active_bout_lengths, active_bout_end, active_bout_start, inactive_bout_lengths, inactive_bout_end, \
           inactive_bout_start, active_speed, active_bout_max, active_indices, inactive_speed, inactive_bout_max, \
           inactive_indices


def find_bout_start_ends_pd(bout_array):
    """ Takes a np.array of zeros and ones and determines the start/stop of the one streches. Assumes no NaNs.

    :param bout_array:
    :return: bout_start_t, bout_end_t
    """
    # test that the  array has no NaNs
    if max(np.isnan(bout_array)):
        print("NaN in bout_array therefore cannot run find_bout_start_ends_pd")
        return False
    else:
        # determine bout starts and finishes
        changes = np.diff(bout_array, axis=0)

        # added 1 to active_bout_start as otherwise it is the last timepoint that was below the threshold.
        # Also did it to ends so a peak of one timepoint would have a length of 1.
        bout_start = np.asarray(np.where(changes == 1)) + 1
        bout_end = np.asarray(np.where(changes == -1)) + 1

        # determine if array started with a bout
        if bout_array[0] == 1:
            # first bout is ongoing, remove first bout as it is incomplete
            bout_start_t = bout_start[0, ]
            bout_end_t = bout_end[0, 1:]
        else:
            # take all starts (and ends)
            bout_start_t = bout_start[0, ]
            bout_end_t = bout_end[0, ]

        # remove incomplete bouts (e.g. those that do not end), in this case there will be one less end than start
        if bout_start_t.shape != bout_end_t.shape:
            if bout_start_t.shape > bout_end_t.shape:
                bout_start_t = bout_start_t[0:-1]
            else:
                print("something weird with number of bouts?")
                return False

        # determine active inter-bout interval
        bout_lengths = bout_end_t - bout_start_t

        return bout_start_t, bout_end_t, bout_lengths


def find_bouts_input(fish_tracks_i, change_times_m,  measure='rest'):
    """ Finds active and inactive bouts, including where they start, how long they are etc
    :param fish_tracks_i:
    :param measure: what to measure in the fish_tracks
    :return: fish_bouts: a dataframe with time stamps of start and ends of "1" or "True" bouts in the given data.
    """
    fishes = fish_tracks_i['FishID'].unique()
    first = True

    for fish in fishes:
        all_bout_starts = pd.Series()
        all_bout_ends = pd.Series()

        # get individual fish
        fish_tracks_f = fish_tracks_i[fish_tracks_i.FishID == fish][['ts', measure]]

        # check if there are NaNs
        if np.max(np.isnan(fish_tracks_f.iloc[:, 1])):
            # break up NaN stretches
            non_nan_array = abs(((np.isnan(fish_tracks_f.iloc[:, 1])) * 1)-1)
            non_nan_array = non_nan_array.to_numpy()
            data_start, data_end = find_bout_start_ends_inclusive(non_nan_array)
        else:
            data_start, data_end = [0], [len(fish_tracks_f)]

        for strech_n in np.arange(0, len(data_start)):
            # calulate data stretches starts and ends
            data_stretch = fish_tracks_f.iloc[data_start[strech_n]:data_end[strech_n], 1]
            data_stetch_ts = fish_tracks_f.iloc[data_start[strech_n]:data_end[strech_n], 0]
            bout_start, bout_end, _ = find_bout_start_ends(data_stretch.to_numpy())
            # add the time stamps of found starts and ends to pd.Series
            all_bout_starts = pd.concat([all_bout_starts.reset_index(drop=True), data_stetch_ts.iloc[bout_start].
                                        reset_index(drop=True)])
            all_bout_ends = pd.concat([all_bout_ends.reset_index(drop=True), data_stetch_ts.iloc[bout_end].
                                      reset_index(drop=True)])

        # import matplotlib.pyplot as plt
        # plt.plot(fish_tracks_f.iloc[data_start[strech_n]:data_end[strech_n], 0], data_stretch)
        # plt.scatter(all_bout_starts, np.zeros([1, len(all_bout_starts)]), color='r')
        # plt.scatter(all_bout_ends, np.zeros([1, len(all_bout_starts)]), color='b')

        # find bout lengths for measure and nonmeasure
        all_bout_measure_lengths = all_bout_ends - all_bout_starts
        all_bout_nonmeasure_lengths = all_bout_starts.to_numpy()[1:] - all_bout_ends.to_numpy()[0:-1]

        # make fish_bouts df
        fish_bouts_i = pd.concat([all_bout_starts.reset_index(drop=True), all_bout_ends.reset_index(drop=True),
                            all_bout_measure_lengths.reset_index(drop=True), pd.Series(all_bout_nonmeasure_lengths)],
                                 axis=1)
        fish_bouts_i.columns = ['bout_start', 'bout_end', measure + '_len', 'non' + measure + '_len']
        fish_bouts_i['FishID'] = fish

        # combine with the other fish
        if first:
            fish_bouts = fish_bouts_i
            first = False
        else:
            fish_bouts = pd.concat([fish_bouts, fish_bouts_i], axis=0)

    fish_bouts = fish_bouts.reset_index(drop=True)

    # add new column with Day or Night
    fish_bouts['time_of_day_m'] = fish_bouts.bout_start.apply(lambda row: int(str(row)[11:16][:-3]) * 60 +
                                                                          int(str(row)[11:16][-2:]))

    fish_bouts['daynight'] = "d"
    fish_bouts.loc[fish_bouts.time_of_day_m < change_times_m[0], 'daynight'] = "n"
    fish_bouts.loc[fish_bouts.time_of_day_m > change_times_m[3], 'daynight'] = "n"

    fish_bouts["bout_start"].groupby(fish_bouts["bout_start"].dt.hour).count().plot(kind="bar")
    fish_bouts.loc[fish_bouts['FishID'] == fish, "bout_start"].groupby(fish_bouts["bout_start"].dt.hour).count().plot(kind="bar")

    return fish_bouts


def names_bouts():
    data_names = ['spd_mean', 'move_mean', 'rest_mean', 'y_mean', 'spd_std', 'move_std', 'rest_std', 'y_std',
                  'move_bout_mean', 'nonmove_bout_mean', 'rest_bout_mean', 'nonrest_bout_mean', 'move_bout_std',
                  'nonmove_bout_std', 'rest_bout_std', 'nonrest_bout_std']
    time_v2_m_names = ['predawn', 'dawn', 'day', 'dusk', 'postdusk', 'night']

    spd_means = ['spd_mean_predawn', 'spd_mean_dawn', 'spd_mean_day', 'spd_mean_dusk', 'spd_mean_postdusk',
                 'spd_mean_night']
    rest_means = ['rest_mean_predawn', 'rest_mean_dawn', 'rest_mean_day', 'rest_mean_dusk', 'rest_mean_postdusk',
                  'rest_mean_night']
    move_means = ['move_mean_predawn', 'move_mean_dawn', 'move_mean_day', 'move_mean_dusk', 'move_mean_postdusk',
                  'move_mean_night']
    rest_b_means = ['rest_bout_mean_predawn', 'rest_bout_mean_dawn', 'rest_bout_mean_day', 'rest_bout_mean_dusk',
                    'rest_bout_mean_postdusk', 'rest_bout_mean_night']
    nonrest_b_means = ['nonrest_bout_mean_predawn', 'nonrest_bout_mean_dawn', 'nonrest_bout_mean_day',
                       'nonrest_bout_mean_dusk',
                       'nonrest_bout_mean_postdusk', 'nonrest_bout_mean_night']
    move_b_means = ['move_bout_mean_predawn', 'move_bout_mean_dawn', 'move_bout_mean_day', 'move_bout_mean_dusk',
                    'move_bout_mean_postdusk', 'move_bout_mean_night']
    nonmove_b_means = ['nonmove_bout_mean_predawn', 'nonmove_bout_mean_dawn', 'nonmove_bout_mean_day',
                       'nonmove_bout_mean_dusk',
                       'nonmove_bout_mean_postdusk', 'nonmove_bout_mean_night']

    # movement_bouts = ['move_bout_mean', 'nonmove_bout_mean', 'move_bout_std']
    # rest_bouts = ['rest_bout_mean', 'nonrest_bout_mean']

    return data_names, time_v2_m_names, spd_means, rest_means, move_means, rest_b_means, nonrest_b_means, move_b_means, nonmove_b_means


if __name__ == "__main__":
    import doctest
    doctest.testmod()
