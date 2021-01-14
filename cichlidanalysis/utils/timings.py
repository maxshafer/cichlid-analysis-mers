import math
import numpy as np

def infer_tv(fps, filechunk):
    """Takes fps and the length of the video to compute an inferred time vector that starts from 0
    >>> infer_tv(1, 3)
    array([0., 1., 2.])
    """
    tv = np.arange(0, filechunk, 1 / fps)
    return tv

def output_timings():
    # set sunrise, day, sunset, night times (ns, s, m, h)
    change_times_s = [7*60*60 + 30*60, 8*60*60, 19*60*60, 19*60*60 + 30*60]
    change_times_ns = [i * 10**9 for i in change_times_s]
    change_times_m = [i / 60 for i in change_times_s]
    change_times_h = [i / 60 / 60 for i in change_times_s]
    change_times_d = [i / 24 for i in change_times_h]

    # set day in ns
    day_ns = 24 * 60 * 60 * 10**9
    day_s = 24 * 60 * 60
    return change_times_s, change_times_ns, change_times_m, change_times_h, day_ns, day_s, change_times_d


def load_timings(vector_length):
    fps = 10

    # get time variables
    change_times_s, change_times_ns, _, change_times_h, day_ns, day_s,  change_times_d = output_timings()
    tv_ns = np.arange(0, day_ns * 8, 10 ** 9 / 10)
    tv_ns = tv_ns[0:vector_length]
    tv_sec = tv_ns / 10 ** 9

    # correct to 24h time
    tv_24h_sec = tv_ns / 10 ** 9
    num_days = 7
    tv_s_type = get_time_state(tv_sec, day_s, change_times_s, fps)

    return fps, tv_ns, tv_sec, tv_24h_sec, num_days, tv_s_type, change_times_s, change_times_ns, change_times_h, \
           day_ns, day_s, change_times_d


def get_time_state(tv_i_s, day_unit_s, change_times_unit_s, fps):
    """ Input must be in seconds!!! state of day, night =  0, dawn/dusk = 1, daylight = 2
    """

    days_to_plot = math.ceil(tv_i_s[-1]/day_unit_s)
    time_state = np.zeros([len(tv_i_s), 1])
    change_times_unit_fps = [element * fps for element in change_times_unit_s]
    day_unit = day_unit_s*fps

    for day_n in range(days_to_plot):
        # night times
        time_state[0+day_unit * day_n:change_times_unit_fps[0] + day_unit * day_n] = 0
        time_state[change_times_unit_fps[3]+day_unit * day_n:day_unit + day_unit * day_n] = 0

        # dawn
        time_state[change_times_unit_fps[0] + day_unit * day_n:change_times_unit_fps[1]+day_unit * day_n] = 1

        # day
        time_state[change_times_unit_fps[1] + day_unit * day_n:change_times_unit_fps[2]+day_unit * day_n] = 2

        # dusk
        time_state[change_times_unit_fps[2] + day_unit * day_n:change_times_unit_fps[3]+day_unit * day_n] = 1

    return time_state


if __name__ == "__main__":
    import doctest
    doctest.testmod()
