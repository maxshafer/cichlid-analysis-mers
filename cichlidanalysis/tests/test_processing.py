import pytest
import numpy as np
import pandas as pd

from cichlidanalysis.analysis.processing import smooth_speed, neg_values, interpolate_nan_streches, remove_high_spd_xy, \
    fish_tracks_add_day_twilight_night, add_day_number_fish_tracks

@pytest.mark.parametrize("win_size, speed_raw, speed_smooth", [
    (2, np.array([0, 0, 0, 1, 0, 0]).reshape(6, 1), np.array([0, 0, 0, 0.5, 0.5, 0]).reshape(6, 1)),
    (3, np.array([0, 0, 0, 1, 0, 0, 0]).reshape(7, 1), np.array([0, 0, 0, 1/3, 1/3, 1/3, 0]).reshape(7, 1))])
def test_smooth_speed(win_size, speed_raw, speed_smooth):
    assert (smooth_speed(speed_raw, win_size) == speed_smooth).all()


@pytest.mark.parametrize("input_array, output_array", [
    (np.array([0, 0, 0, 1, 2, 20]).reshape(6, 1), np.array([0, 0, 0, -1, -2, -20]).reshape(6, 1)),
    (np.array([0, 0, 0, 1, 2, -20]).reshape(6, 1), np.array([0, 0, 0, -1, -2, -20]).reshape(6, 1))])
def test_neg_values(input_array, output_array):
    assert (neg_values(input_array) == output_array).all()


@pytest.mark.parametrize("input_data, binned_data", [
    (np.array([0, 0, 0, 1, 0, 0]), np.array([0, 0, 0, 1, 0, 0])),
    (np.array([0, 0, np.nan, 0, 1, np.nan, np.nan, np.nan,  0, 0]), np.array([0, 0, 0, 0, 1, 0.75, 0.5, 0.25, 0, 0])),
    (np.array([0, 0, np.nan, 0, 1, np.nan, np.nan, 1, 0]), np.array([0, 0, 0, 0, 1, 1, 1, 1, 0])),
    (np.array([np.nan, np.nan, 0, np.nan, 0, 1, np.nan, 0, 0]), np.array([0, 0, 0, 0, 0, 1, 0.5, 0, 0])),
    (np.array([np.nan, np.nan, 0, 0, 1, np.nan, 0, np.nan, np.nan]), np.array([0, 0, 0, 0, 1, 0.5, 0, 0, 0]))])
def test_int_nan_streches(input_data, binned_data):
    output_data = interpolate_nan_streches(input_data)
    assert (output_data == binned_data).all()


@pytest.mark.parametrize("speed_raw, x,  y, thresh_speed, thresh_x, thresh_y", [
    (np.array([0, 0, 0, 1, 0, 0]), np.array([0, 0, 0, 1, 0, 0]), np.array([0, 0, 0, 1, 0, 0]),
     np.array([0, 0, 0, 1, 0, 0]),  np.array([0, 0, 0, 1, 0, 0]), np.array([0, 0, 0, 1, 0, 0])),
    (np.array([0, 0, 0, 201, 0, 0]), np.array([0, 0, 0, 1, 0, 0]), np.array([0, 0, 0, 1, 0, 0]),
     np.array([0, 0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0, 0])),
    (np.array([12, 12, 12, 12, 12, 205, 10, 10, 10, 10, 10, 201, 10, 10]), np.ones(14), np.ones(14),
     np.array([12, 12, 12, 12, 12, 11, 10, 10, 10, 10, 10, 10, 10, 10]), np.ones(14), np.ones(14)),
    (np.array([10, 204, 205, 205, 205, 205, 10, 10, 10, 10, 10, 201, 10, 10]), np.ones(14), np.ones(14),
     np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]), np.ones(14), np.ones(14)),
    (np.array([10, 204, float("nan"), 205, 205, 205, 10, 10, 10, 10, 10, 201, 10, 10]), np.ones(14), np.ones(14),
     np.array([10, 10, -1, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]), np.ones(14), np.ones(14))])
def test_remove_high_spd_xy(speed_raw, x, y, thresh_speed, thresh_x, thresh_y):
    speed_t, x_t, y_t = remove_high_spd_xy(speed_raw, x, y)
    errors = []

    # as NaNs do not == NaN, need to find NaNs and replace ( with -1)
    speed_t_2 = np.where(np.isnan(speed_t), -1, speed_t)
    x_t_2 = np.where(np.isnan(x_t), -1, x_t)
    y_t_2 = np.where(np.isnan(y_t), -1, y_t)

    if not (speed_t_2 == thresh_speed).all():
        errors.append("speed not thresholded properly")
        print(speed_t)
    if not (x_t_2 == thresh_x).all():
        errors.append("x not thresholded properly")
    if not (y_t_2 == thresh_y).all():
        errors.append("y not thresholded properly")

    assert not errors, "errors occured:\n{}".format("\n".join(errors))


@pytest.mark.parametrize("input_fish_tracks_ds, expected", [(
    pd.DataFrame(np.array([350, 359, 360, 361, 479, 480, 481, 1079, 1080, 1081, 1199, 1200, 1201]), columns=['time_of_day_m']),
    pd.DataFrame([[350, 'n'], [359, 'n'], [360, 'c'],  [361, 'c'], [479, 'c'], [480, 'd'], [481, 'd'], [1079, 'd'],
                  [1080, 'c'], [1081, 'c'], [1199, 'c'], [1200, 'n'], [1201, 'n']],
                 columns=['time_of_day_m', 'daytime']))])
def test_fish_tracks_add_day_twilight_night(input_fish_tracks_ds, expected):
    output_fish_tracks_ds = fish_tracks_add_day_twilight_night(input_fish_tracks_ds)
    assert output_fish_tracks_ds.equals(expected)


@pytest.mark.parametrize("input_fish_tracks_ds, expected", [(
    pd.DataFrame(['1970-01-02 00:00:00', '1970-01-02 01:30:00', '1970-01-03 06:00:00', '1970-01-03 10:30:00'], columns=['ts']),
    pd.DataFrame([['1970-01-02 00:00:00', 1], ['1970-01-02 01:30:00', 1], ['1970-01-03 06:00:00', 2], ['1970-01-03 10:30:00', 2]],
                 columns=['ts', 'day_n']))])
def test_add_day_number_fish_tracks(input_fish_tracks_ds, expected):
    output_fish_tracks_ds = add_day_number_fish_tracks(input_fish_tracks_ds)
    assert output_fish_tracks_ds.equals(expected)
