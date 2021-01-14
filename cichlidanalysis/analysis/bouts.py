import numpy as np

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


if __name__ == "__main__":
    import doctest
    doctest.testmod()
