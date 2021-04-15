import copy
import os
import glob
from tkinter.filedialog import askdirectory
from tkinter import Tk

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from cichlidanalysis.io.meta import load_yaml
from cichlidanalysis.io.tracks import extract_tracks_from_fld, adjust_old_time_ns
from cichlidanalysis.utils.timings import output_timings
from cichlidanalysis.analysis.processing import int_nan_streches, remove_high_spd_xy, smooth_speed, neg_values
from cichlidanalysis.plotting.single_plots import filled_plot, plot_hist_2, image_minmax, sec_axis_h


def full_analysis(rootdir):
    parts = os.path.split(rootdir)
    config = load_yaml(parts[0], "config")
    meta = load_yaml(rootdir, "meta_data")

    if len(parts[1]) < 19:
        print("old recording naming, reconstructing name")
        rec_folder = os.path.split(os.path.split(parts[0])[0])[1]
        cam_n = os.path.split(parts[0])[1][-1]
        roi_n = parts[1][-1]

        fish_ID = "{0}_c{1}_r{2}_{3}_s{4}".format(rec_folder, cam_n, roi_n, meta["species"].replace(' ', '-'),
                                                  meta["sex"])
    else:
        fish_ID = parts[1]
        roi_n = fish_ID.split("_")[2][1]

    file_ending = roi_n
    # load tracks
    track_full, speed_full = extract_tracks_from_fld(rootdir, file_ending)

    # for old recordings update time (subtract 30min)
    track_full[:, 0] = adjust_old_time_ns(fish_ID, track_full[:, 0])

    # get starting time of video
    os.chdir(rootdir)
    files = glob.glob("*.csv")
    files.sort()
    start_time = files[0][9:15]
    start_total_sec = (int(start_time[0:2]) * 60 * 60 + int(start_time[2:4]) * 60 + int(start_time[4:]))

    # set sunrise, day, sunset, night times (ns, s, m, h) and set day length in ns
    change_times_s, change_times_ns, change_times_m, change_times_h, day_ns, day_s, change_times_d = output_timings()

    # set time vector
    if track_full[0, 0] == 0:
        tv = np.arange(start_total_sec * 10 ** 9,
                       ((track_full.shape[0] / config['fps']) * 10 ** 9 + start_total_sec * 10 ** 9),
                       ((1 / config['fps']) * 10 ** 9))
        print("using retracked data so using interpolated time vector")
    else:
        tv = track_full[:, 0] - track_full[0, 0] + start_total_sec * 10 ** 9

    # correct to seconds
    tv_sec = tv / 10 ** 9
    tv_24h_sec = tv / 10 ** 9
    num_days = 7

    # get time vector with 24h time
    for i in range(num_days):
        tv_24h_sec[np.where(tv_24h_sec > day_ns / 10 ** 9)] -= day_ns / 10 ** 9
    min_bins = 30

    # interpolate between NaN streches
    x_n = int_nan_streches(track_full[:, 1])
    y_n = int_nan_streches(track_full[:, 2])

    # replace bad track NaNs (-1) -> these are manually defined as artifacts by "split_tracking"
    x_n[np.where(x_n == -1)] = np.nan
    y_n[np.where(y_n == -1)] = np.nan

    # find displacement
    speed_full_i = np.sqrt(np.diff(x_n) ** 2 + np.diff(y_n) ** 2)
    speed_t, x_nt, y_nt = remove_high_spd_xy(speed_full_i, x_n, y_n)

    speed_sm = smooth_speed(speed_t, win_size=5)
    speed_sm_mm = speed_sm * config["mm_per_pixel"]
    speed_sm_mm_ps = speed_sm_mm * config['fps']
    speed_sm_tbl = speed_sm_mm / meta["fish_length_mm"]
    speed_sm_tbl_ps = speed_sm_tbl * config['fps']

    # smoothing on coordinates
    fig1, ax1 = plt.subplots()
    plt.hist(np.diff(x_nt), 1000)
    plt.yscale('log')
    plt.xlabel("pixels")
    plt.ylabel("frequency")
    plt.title("X_{0}".format(meta["species"]))
    plt.savefig(os.path.join(rootdir, "{0}_X_jumps_{1}.png".format(fish_ID, meta["species"].replace(' ', '-'))))

    fig1, ax1 = plt.subplots()
    plt.hist(np.diff(y_nt), 1000)
    plt.yscale('log')
    plt.xlabel("pixels")
    plt.ylabel("frequency")
    plt.title("Y_{0}".format(meta["species"]))
    plt.savefig(os.path.join(rootdir, "{0}_Y-jumps_{1}.png".format(fish_ID, meta["species"].replace(' ', '-'))))

    y_sm = smooth_speed(y_nt, win_size=5)
    smooth_win = 10 * 60 * min_bins
    y_bin = smooth_speed(y_sm, win_size=smooth_win)

    plt.close()
    fig2, ax2 = filled_plot(tv / 10 ** 9 / 60 / 60, y_bin[0:-1], change_times_h, day_ns / 10 ** 9 / 60 / 60)
    plt.xlabel("Time (h)")
    ax2.invert_yaxis()
    plt.ylabel("average y position")
    plt.title("Y position_{0}_smoothed_by_{1}".format(meta["species"], min_bins))
    plt.savefig(os.path.join(rootdir, "{0}_Y-position_{1}.png".format(fish_ID, meta["species"].replace(' ', '-'))))

    # area
    area_sm = smooth_speed(track_full[0:-1, 3], win_size=5)
    area_bin = smooth_speed(area_sm, win_size=smooth_win)

    plt.close()
    fig2, ax2 = filled_plot(tv / 10 ** 9 / 60 / 60, area_bin, change_times_h, day_ns / 10 ** 9 / 60 / 60)
    plt.xlabel("Time (h)")
    plt.ylabel("average area size")
    plt.title("Area_{0}_smoothed_by_{1}".format(meta["species"], min_bins))

    # split data into day and  night
    tv_night = np.empty([0, 0])
    speed_sm_night = np.empty([0, 0])

    tv_night = np.append(tv_night, tv_24h_sec[np.where(change_times_s[0] > tv_24h_sec)])
    tv_night = np.append(tv_night, tv[np.where(tv_24h_sec[0:-1] > change_times_s[3])])

    speed_sm_night = np.append(speed_sm_night, speed_sm_mm_ps[np.where(change_times_s[0] > tv_24h_sec[0:-1]), 0])
    speed_sm_night = np.append(speed_sm_night, speed_sm_mm_ps[np.where(tv_24h_sec[0:-1] > change_times_s[3]), 0])

    tv_day = np.empty([0, 0])
    speed_sm_day = np.empty([0, 0])

    tv_day = np.append(tv_day, tv[np.where((change_times_s[0] < tv_24h_sec[0:-1]) &
                                           (tv_24h_sec[0:-1] < change_times_s[3]))])
    speed_sm_day = np.append(speed_sm_day, speed_sm_mm_ps[np.where((change_times_s[0] < tv_24h_sec[0:-1]) &
                                                             (tv_24h_sec[0:-1] < change_times_s[3])), 0])

    # plot speed distributions
    bin_edges_plot = np.linspace(0, 200, 101)
    # bin_edges_plot = np.logspace(0, 1.2, 10)
    plot_hist_2(bin_edges_plot, speed_sm_day, "day", speed_sm_night, "night", "speed mm/s", 1)
    plt.savefig(os.path.join(rootdir, "{0}_hist_D_vs_N_{1}_spd_mms.png".format(fish_ID, meta["species"].replace(' ', '-'))))

    # split data into day and night
    position_night_x = np.empty([0, 0])
    position_night_y = np.empty([0, 0])

    position_night_x = np.append(position_night_x, x_nt[np.where(change_times_s[0] > tv_24h_sec)])
    position_night_x = np.append(position_night_x, x_nt[np.where(tv_24h_sec[0:-1] > change_times_s[3])])

    position_night_y = np.append(position_night_y, y_nt[np.where(change_times_s[0] > tv_24h_sec)])
    position_night_y = np.append(position_night_y, y_nt[np.where(tv_24h_sec[0:-1] > change_times_s[3])])

    position_day_x = np.empty([0, 0])
    position_day_y = np.empty([0, 0])

    position_day_x = np.append(position_day_x, x_nt[np.where((change_times_s[0] < tv_24h_sec[0:-1]) &
                                                             (tv_24h_sec[0:-1] < change_times_s[3]))])
    position_day_y = np.append(position_day_y, y_nt[np.where((change_times_s[0] < tv_24h_sec[0:-1]) &
                                                             (tv_24h_sec[0:-1] < change_times_s[3]))])

    # plot position (fix = remove x,y when they were over threshold)
    bin_edges_plot = np.linspace(0, 800, 101)
    plot_hist_2(bin_edges_plot, position_day_y, "day", position_night_y, "night", "Y position", 1)

    xmin = np.nanmin(x_nt[:])
    xmax = np.nanmax(x_nt[:])
    ymin = np.nanmin(y_nt[:])
    ymax = np.nanmax(y_nt[:])
    image_minmax(rootdir, ymin, ymax, fish_ID, meta)

    fig, (ax1, ax2) = plt.subplots(2, 7, sharey=True)
    for day in range(num_days):
        position_night_x = x_nt[np.where((tv_sec > (change_times_s[3] + day_s * day)) &
                                         (tv_sec < (change_times_s[0] + day_s * (day + 1))))]
        position_night_y = y_nt[np.where((tv_sec > (change_times_s[3] + day_s * day)) &
                                         (tv_sec < (change_times_s[0] + day_s * (day + 1))))]

        position_day_x = x_nt[np.where((tv_sec > (change_times_s[0] + day_s * day)) &
                                       (tv_sec < (change_times_s[3] + day_s * day)))]
        position_day_y = y_nt[np.where((tv_sec > (change_times_s[0] + day_s * day)) &
                                       (tv_sec < (change_times_s[3] + day_s * day)))]

        ax1[day].hist2d(position_day_x[~np.isnan(position_day_x)],
                        neg_values(position_day_y[~np.isnan(position_day_y)]),
                        bins=10, range=[[xmin, xmax], [-ymax, ymin]], cmap='inferno')
        ax2[day].hist2d(position_night_x[~np.isnan(position_night_x)],
                        neg_values(position_night_y[~np.isnan(position_night_y)]),
                        bins=10, range=[[xmin, xmax], [-ymax, ymin]], cmap='inferno')
    plt.savefig(os.path.join(rootdir, "{0}_hist2d_D_vs_N_split_days_spt.png".format(fish_ID)))

    # # speed vs y position
    # # plt.scatter(speed_sm_mm_ps, y_nt[0:-1])
    # spd_max = np.percentile(speed_sm_mm_ps, 95)
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # test_s = speed_sm_mm_ps[~np.isnan(speed_sm_mm_ps)]
    # test_y = y_nt[~np.isnan(y_nt)]
    # ax1.hist2d(test_s, neg_values(test_y[0:-1]), bins=10, range=[[0, spd_max], [-ymax, ymin]], cmap='inferno')

    # looking at correlations
    # covariance = np.cov(test_s, test_y[0:-1])
    # from scipy.stats import pearsonr
    # corr, _ = pearsonr(test_s, test_y[0:-1])
    # print(corr)

    # ax1.hist2d(speed_sm_mm_ps[~np.isnan(speed_sm_mm_ps)], neg_values(position_day_y[0:-1][~np.isnan(position_day_y)]),
    #            bins=10, range=[[0, spd_max], [-ymax, ymin]], cmap='inferno')

    # Distribute position into categories: y [top, centre, bottom], or x [centre side].
    # Assume the fish explores the whole area over the video
    y_bins = 10
    x_bins = 5
    y_bin_size = (ymax - ymin) / y_bins
    x_bin_size = (xmax - xmin) / x_bins

    vertical_pos = np.empty([y_nt.shape[0]])
    previous_y_bin = 0
    for y_bin in range(1, y_bins + 1):
        vertical_pos[np.where(
            np.logical_and((y_nt - ymin) >= previous_y_bin * y_bin_size, (y_nt - ymin) <= y_bin * y_bin_size))] = y_bin
        previous_y_bin = copy.copy(y_bin)

    horizontal_pos = np.empty([x_nt.shape[0]])
    previous_x_bin = 0
    for x_bin in range(1, x_bins + 1):
        horizontal_pos[np.where(
            np.logical_and((x_nt - xmin) >= previous_x_bin * x_bin_size, (x_nt - xmin) <= x_bin * x_bin_size))] = x_bin
        previous_x_bin = copy.copy(x_bin)

    move_thresh = 15

    # Bin thresholded data (10fps = seconds, 60 seconds = min e.g. 10*60*10 = 10min bins
    fraction_active = (speed_sm_mm_ps > move_thresh) * 1
    super_threshold_indices_bin = smooth_speed(fraction_active, 10 * 60 * min_bins)

    # filled plot in s
    plt.close()
    fig1, ax1 = filled_plot(tv / 10 ** 9 / 60 / 60, super_threshold_indices_bin, change_times_h,
                            day_ns / 10 ** 9 / 60 / 60)
    ax1.set_ylim([0, 1])
    sec_axis_h(ax1, start_total_sec)
    plt.xlabel("Time (h)")
    plt.ylabel("Fraction active in {} min sliding windows".format(min_bins))
    plt.title("Fraction_active_{}_thresh_{}_mmps".format(meta["species"], move_thresh))
    plt.savefig(os.path.join(rootdir, "{0}_wake_{1}_spt.png".format(fish_ID, meta["species"].replace(' ', '-'))))

    # win_size = fps * sec/min * mins (was 30*60)heatm
    smooth_win = 10 * 60 * min_bins
    speed_sm_bin = smooth_speed(speed_sm_tbl_ps, win_size=smooth_win)
    plt.close()
    fig2, ax2 = filled_plot(tv / 10 ** 9 / 60 / 60, speed_sm_bin, change_times_h, day_ns / 10 ** 9 / 60 / 60)
    sec_axis_h(ax2, start_total_sec)
    plt.xlabel("Time (h)")
    plt.ylabel("Speed body lengths/s")
    plt.title("Speed_{0}_smoothed_by_{1}".format(meta["species"], min_bins))
    plt.savefig(os.path.join(rootdir, "{0}_speed_{1}_30m_spt.png".format(fish_ID, meta["species"].replace(' ', '-'))))

    # win_size = fps * sec/min * mins (was 30*60)heatm
    smooth_win = 10 * 60 * min_bins
    speed_sm_mm_bin = smooth_speed(speed_sm_mm_ps, win_size=smooth_win)
    plt.close()
    fig2, ax2 = filled_plot(tv / 10 ** 9 / 60 / 60, speed_sm_mm_bin, change_times_h, day_ns / 10 ** 9 / 60 / 60)
    sec_axis_h(ax2, start_total_sec)
    plt.xlabel("Time (h)")
    plt.ylabel("Speed mm/s")
    plt.title("Speed_{0}_smoothed_by_{1}".format(meta["species"], min_bins))
    plt.savefig(os.path.join(rootdir, "{0}_speed_{1}_30m_spt.png".format(fish_ID, meta["species"].replace(' ', '-'))))

    plt.close()
    fig2, ax2 = filled_plot(tv / 10 ** 9 / 60 / 60, speed_sm_mm_bin, change_times_h, day_ns / 10 ** 9 / 60 / 60)
    sec_axis_h(ax2, start_total_sec)
    plt.xlabel("Time (h)")
    plt.ylabel("Speed mm/s")
    plt.title("Speed_{0}_smoothed_by_{1}".format(meta["species"], min_bins))
    ax2.set_ylim(0, 60)
    plt.savefig(
        os.path.join(rootdir, "{0}_speed_{1}_30m_spt_0-60ylim.png".format(fish_ID, meta["species"].replace(' ', '-'))))

    smooth_win = 10 * 60 * 10
    speed_sm_mm_bin = smooth_speed(speed_sm_mm_ps, win_size=smooth_win)
    plt.close()
    fig2, ax2 = filled_plot(tv / 10 ** 9 / 60 / 60, speed_sm_mm_bin, change_times_h, day_ns / 10 ** 9 / 60 / 60)
    sec_axis_h(ax2, start_total_sec)
    plt.xlabel("Time (h)")
    plt.ylabel("Speed mm/s")
    plt.title("Speed_{0}_smoothed_by_{1}".format(meta["species"], min_bins))
    plt.savefig(
        os.path.join(rootdir, "{0}_speed_{1}_{2}m_spt.png".format(fish_ID, meta["species"].replace(' ', '-'), 10)))

    plt.close()
    fig2, ax2 = filled_plot(tv / 10 ** 9 / 60 / 60, speed_full, change_times_h, day_ns / 10 ** 9 / 60 / 60)
    plt.plot((tv / 10 ** 9 / 60 / 60)[0:-1], speed_t)
    sec_axis_h(ax2, start_total_sec)
    plt.xlabel("Time (h)")
    plt.ylabel("Speed pixels/0.1s")
    plt.title("Speed_{0}_raw-black_thresholded-blue".format(fish_ID,meta["species"].replace(' ', '-')))
    plt.savefig(os.path.join(rootdir, "{0}_speed_{1}_speed_full_speed_thresholded.png".format(fish_ID,
                                                                                              meta["species"].replace(
                                                                                                  ' ', '-'))))

    # area
    plt.close()
    fig2, ax2 = filled_plot(tv / 10 ** 9 / 60 / 60, track_full[0:-1, 3], change_times_h, day_ns / 10 ** 9 / 60 / 60)
    plt.xlabel("Time (h)")
    plt.ylabel("Area pixels/0.1s")
    plt.title("Area_{0}".format(meta["species"]))
    sec_axis_h(ax2, start_total_sec)
    plt.savefig(os.path.join(rootdir, "{0}_{1}_area.png".format(fish_ID, meta["species"].replace(' ', '-'))))

    plt.close()
    fig3, ax3 = filled_plot(tv / 10 ** 9 / 60 / 60, np.diff(tv) / 10 ** 9, change_times_h, day_ns / 10 ** 9 / 60 / 60)
    sec_axis_h(ax3, start_total_sec)
    plt.xlabel("Time (h)")
    plt.ylabel("Inter frame time difference (s)")
    plt.title("TV_{0}".format(meta["species"]))
    plt.savefig(os.path.join(rootdir, "{0}_{1}_TV_diff.png".format(fish_ID, meta["species"].replace(' ', '-'))))

    # save out track file
    # track file needs: FISH20200727_c1_r1_Genus-species_sex_mmpp_fishlength-mm
    # speed_sm_tbl_ps, tv, x, y, fraction_active
    # speed_sm_mm_ps, tv, x, y

    track_meta = {'ID': fish_ID, 'species': meta["species"], 'sex': meta["sex"],
                  'fish_length_mm': meta["fish_length_mm"], 'mm_per_pixel': config["mm_per_pixel"]}
    meta_df = pd.DataFrame(track_meta, columns=['ID', 'species', 'sex', 'fish_length_mm', 'mm_per_pixel'], index=[0])
    meta_df.to_csv(os.path.join(rootdir, "{0}_meta.csv".format(fish_ID)))

    # start from midnight (so they all start at the same time) - need to adjust "midnight" depending on if ts were
    # adjusted for 30min shift (all recordings before 20201127).
    if int(fish_ID[4:12]) < 20201127:
        thirty_min_ns = 30 * 60 * 1000000000
        adjusted_day_ns = day_ns - thirty_min_ns
        print("old recording from before 20201127 so adjusting back time before saving out als")
    else:
        adjusted_day_ns = day_ns

    midnight = np.max(np.where(tv < adjusted_day_ns))

    track_als = np.vstack((tv[midnight:-1], speed_sm_mm_ps[midnight:, 0], x_nt[midnight:-1], y_nt[midnight:-1]))

    als_df = pd.DataFrame(track_als.T, columns=['tv_ns', 'speed_mm', 'x_nt', 'y_nt'],
                          index=pd.Index(np.arange(0, len(speed_sm_tbl_ps[midnight:]))))
    als_df.to_csv(os.path.join(rootdir, "{}_als.csv".format(fish_ID)))
    plt.close('all')


if __name__ == '__main__':
    analyse_multiple_folders = 'm'
    while analyse_multiple_folders not in {'y', 'n'}:
        analyse_multiple_folders = input("Analyse multiple folders (ROIs) (y) or only one ROI (n)?: \n")

    if analyse_multiple_folders == 'n':
        # Allows a user to select top directory
        root = Tk()
        root.withdraw()
        root.update()
        rootdir = askdirectory(parent=root, title="Select roi folder (which has the movies and tracks)")
        root.destroy()

        full_analysis(rootdir)
    else:
        # Allows a user to select top directory
        root = Tk()
        root.withdraw()
        root.update()
        topdir = askdirectory(parent=root, title="Select top recording folder (which has the camera folders)")
        root.destroy()

        list_subfolders_with_paths = [f.path for f in os.scandir(topdir) if f.is_dir()]

        for camera_folder in list_subfolders_with_paths:
            # find the camera roi folders
            list_subsubfolders_with_paths = [f.path for f in os.scandir(camera_folder) if f.is_dir()]
            for roi_folder in list_subsubfolders_with_paths:
                full_analysis(roi_folder)
