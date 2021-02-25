############################
# This module loads als and meta data of individual fish and plots the following for each species:
# speed_mm (30m bins, daily ave) for each fish (lines single and average as well as heatmap)
# x,y position (binned day/night, and average day/night)
# fraction mobile/immobile
# fraction active/quiescent
# bout structure (MI and AQ, bout fraction in 30min bins, bouts D/N over days)

# For combined data this module will plot:
# speed_mm (30m bins, daily ave) for each species (lines and heatmap)
# x,y position (binned day/night,  and average day/night)

from tkinter.filedialog import askdirectory
from tkinter import *
import os
import warnings
import time

import pandas as pd
import numpy as np
import datetime as dt

from cichlidanalysis.io.meta import load_meta_files
from cichlidanalysis.io.tracks import load_als_files
from cichlidanalysis.utils.timings import load_timings
from cichlidanalysis.analysis.processing import add_col, threshold_data
from cichlidanalysis.plotting.position_plots import spd_vs_y, plot_position_maps
from cichlidanalysis.plotting.speed_plots import plot_speed_30m_individuals, plot_speed_30m_mstd
from cichlidanalysis.plotting.movement_plots import plot_movement_30m_individuals, plot_movement_30m_mstd
from cichlidanalysis.plotting.daily_plots import plot_daily
# from cichlidanalysis.analysis.behavioural_state import define_bs, bout_play

# debug pycharm problem
warnings.simplefilter(action='ignore', category=FutureWarning)

# pick folder
# Allows user to select top directory and load all als files here
root = Tk()
root.withdraw()
root.update()
rootdir = askdirectory(parent=root)
root.destroy()

t0 = time.time()
fish_tracks = load_als_files(rootdir)
t1 = time.time()
print("time to load tracks {}".format(t1-t0))

# drop any time points < dt.datetime.strptime("1970-1-2 00:00:00", '%Y-%m-%d %H:%M:%S') (since some are adjusted)
fish_tracks = fish_tracks.drop(fish_tracks[fish_tracks.ts < dt.datetime.strptime("1970-1-2 00:00:00",
                                                                                 '%Y-%m-%d %H:%M:%S')].index)
fish_tracks.reset_index()

meta = load_meta_files(rootdir)
metat = meta.transpose()
remove = ['vertical_pos', 'horizontal_pos', 'speed_bl', 'activity']
for remove_name in remove:
    if remove_name in fish_tracks.columns:
        fish_tracks = fish_tracks.drop(remove_name, axis=1)
        print("old track, removed {}".format(remove_name))

# get each fish ID
fish_IDs = fish_tracks['FishID'].unique()

# get timings
fps, tv_ns, tv_sec, tv_24h_sec, num_days, tv_s_type, change_times_s, change_times_ns, change_times_h, day_ns, day_s,\
    change_times_d, change_times_m = load_timings(fish_tracks[fish_tracks.FishID == fish_IDs[0]].shape[0])
change_times_unit = [7*2, 7.5*2, 18.5*2, 19*2]

# add new column with Day or Night
t2 = time.time()
fish_tracks['time_of_day_m'] = fish_tracks.ts.apply(lambda row: int(str(row)[11:16][:-3]) * 60 + int(str(row)[11:16][-2:]))
t3 = time.time()
print("time to add time_of_day tracks {}".format(t3-t2))

fish_tracks['daynight'] = "d"
fish_tracks.loc[fish_tracks.time_of_day_m < change_times_m[0], 'daynight'] = "n"
fish_tracks.loc[fish_tracks.time_of_day_m > change_times_m[3], 'daynight'] = "n"

#### Movement moving/not-moving use 0.25bl threshold ####
fish_tracks['movement'] = np.nan
for fish in fish_IDs:
    # threshold the speed_mm with 0.25 of the body length of the fish
    fish_tracks.loc[(fish_tracks.FishID == fish), 'movement'] = \
        threshold_data(fish_tracks.loc[(fish_tracks.FishID == fish), "speed_mm"], metat.loc[fish, 'fish_length_mm']*0.25)

fish_tracks.info()

##### x,y position (binned day/night, and average day/night) #####
# resample data
horizontal_pos = fish_tracks.pivot(columns="FishID", values="x_nt")
vertical_pos = fish_tracks.pivot(columns="FishID", values="y_nt")

# scale each fish by min/max
horizontal_pos -= horizontal_pos.min()
horizontal_pos /= horizontal_pos.max()
vertical_pos -= vertical_pos.min()
vertical_pos /= vertical_pos.max()
# flip Y axis
vertical_pos = abs(1 - vertical_pos)

# put this data back into fish_tracks
fish_tracks['vertical_pos'] = np.nan
fish_tracks['horizontal_pos'] = np.nan
for fish in fish_IDs:
    fish_tracks.loc[fish_tracks.FishID == fish, 'vertical_pos'] = vertical_pos.loc[:, fish]
    fish_tracks.loc[fish_tracks.FishID == fish, 'horizontal_pos'] = horizontal_pos.loc[:, fish]

# data gets heavy so remove what is not necessary
remove = ['y_nt', 'x_nt', 'tv_ns']
for remove_name in remove:
    if remove_name in fish_tracks.columns:
        fish_tracks = fish_tracks.drop(remove_name, axis=1)
        print("removed {}".format(remove_name))

# resample data
fish_tracks_30m = fish_tracks.groupby('FishID').resample('30T', on='ts').mean()
fish_tracks_30m.reset_index(inplace=True)

# add back 'species', 'sex'
# for col_name in ['species', 'sex', 'fish_length_mm']:
for col_name in ['species']:
    add_col(fish_tracks_30m, col_name, fish_IDs, meta)
all_species = fish_tracks_30m['species'].unique()

fish_tracks_30m['daynight'] = "d"
fish_tracks_30m.loc[fish_tracks_30m.time_of_day_m < change_times_m[0], 'daynight'] = "n"
fish_tracks_30m.loc[fish_tracks_30m.time_of_day_m > change_times_m[3], 'daynight'] = "n"


# # ### Behavioural state - calculated from Movement ###
# time_window_s = 10
# fraction_threshold = 0.2
#
# testing1 = bout_play(fish_tracks, metat)
#
# testing = define_bs(fish_tracks, rootdir, time_window_s, fraction_threshold)



# ### plotting ### #

# speed_mm (30m bins) for each fish (individual lines)
plot_speed_30m_individuals(rootdir, fish_tracks_30m, change_times_d)

# speed_mm (30m bins) for each species (mean  +- std)
plot_speed_30m_mstd(rootdir, fish_tracks_30m, change_times_d)


# movement for each fish (individual lines)
plot_movement_30m_individuals(rootdir, fish_tracks_30m, change_times_d)

# movement (30m bins) for each species (mean  +- std)
plot_movement_30m_mstd(rootdir, fish_tracks_30m, change_times_d)


# get daily average
plot_daily(fish_tracks_30m, change_times_unit, rootdir)


# ##### x,y position (binned day/night, and average day/night) #####
plot_position_maps(meta, fish_tracks, rootdir)

# speed vs Y position, for each fish, for combine fish of species, separated between day and night
spd_vs_y(meta, fish_tracks_30m, fish_IDs, rootdir)



# feature vector: for each fish readout vector of feature values
# version 1: Day/Night for: speed -  mean, stdev, median; y position - mean, stdev, median;
# FM - mean, stdev, median;
# to add later:
#   BS - mean, stdev, median
#   30min bins of each data

column_names = ['spd_mean_d', 'spd_mean_n', 'spd_std_d', 'spd_std_n', 'spd_median_d', 'spd_median_n', 'move_mean_d',
                'move_mean_n', 'move_std_d', 'move_std_n', 'move_median_d', 'move_median_n', 'y_mean_d', 'y_mean_n',
                'y_std_d', 'y_std_n', 'y_median_d', 'y_median_n', 'fish_length_mm']

for species in all_species:
    df = pd.DataFrame([],  columns=column_names)

    for fish in fish_IDs:
        fish_v_d = fish_tracks.loc[(fish_tracks.FishID == fish) & (fish_tracks.daynight == "d"),
                                            ["speed_mm", "movement", "vertical_pos"]]
        fish_v_n = fish_tracks.loc[(fish_tracks.FishID == fish) & (fish_tracks.daynight == "n"),
                                            ["speed_mm", "movement", "vertical_pos"]]

        df_f = pd.DataFrame([[fish_v_d.mean()[0], fish_v_n.mean()[0], fish_v_d.std()[0], fish_v_n.std()[0],
                              fish_v_d.median()[0], fish_v_n.median()[0], fish_v_d.mean()[1], fish_v_n.mean()[1],
                              fish_v_d.std()[1], fish_v_n.std()[1], fish_v_d.median()[1], fish_v_n.median()[1],
                              fish_v_d.mean()[2], fish_v_n.mean()[2], fish_v_d.std()[2], fish_v_n.std()[2],
                              fish_v_d.median()[2], fish_v_n.median()[2], metat.loc[fish, 'fish_length_mm']]],
                            index=[fish], columns=column_names)
        df_f = df_f.round(4)
        df = pd.concat([df, df_f])

    df.to_csv(os.path.join(rootdir, "{}_als_fv.csv".format(species)))

# save out 30m data (all adjusted to 7am-7pm)
fish_tracks_30m.to_csv(os.path.join(rootdir, "{}_als_30m.csv".format(species)))
