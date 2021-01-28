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
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)
import seaborn as sns

from cichlidanalysis.io.meta import load_meta_files
from cichlidanalysis.io.tracks import load_als_files
from cichlidanalysis.utils.timings import load_timings
from cichlidanalysis.analysis.processing import add_col, threshold_data
from cichlidanalysis.plotting.position_plots import spd_vs_y
# from cichlidanalysis.analysis.bouts import find_bouts
from cichlidanalysis.plotting.speed_plots import plot_speed_30m_individuals, plot_speed_30m_mstd
from cichlidanalysis.plotting.movement_plots import plot_movement_30m_individuals, plot_movement_30m_mstd

# debug pycharm problem
warnings.simplefilter(action='ignore', category=FutureWarning)

# useful
# data.columns
# making a pandas dataframe with tv_i (from middnight until 7days), x, y, speed_sm, FISH-ID, species, sex, fish_mm

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

meta = load_meta_files(rootdir)
metat = meta.transpose()
remove = ['vertical_pos', 'horizontal_pos', 'speed_bl', 'activity']
for remove_name in remove:
    if remove_name in fish_tracks.columns:
        fish_tracks = fish_tracks.drop(remove_name, axis=1)
        print("removed {}".format(remove_name))

# get each fish ID
fish_IDs = fish_tracks['FishID'].unique()

# get timings
fps, tv_ns, tv_sec, tv_24h_sec, num_days, tv_s_type, change_times_s, change_times_ns, change_times_h, day_ns, day_s,\
    change_times_d, change_times_m = load_timings(fish_tracks[fish_tracks.FishID == fish_IDs[0]].shape[0])
change_times_unit = [7.5*2, 8*2, 19*2, 19.5*2]

# # if from 7am - 7pm # need  better way to deal with this change!
# if statement for date (then add 30min to times stamps? as some fish will have this and others won't
# change_times_unit = [7*2, 7.5*2, 18.5*2, 19*2]
# change_times_s = [i - 1800 for i in change_times_s]
# change_times_h = [i - 0.5 for i in change_times_h]
# change_times_d = [i - .5/24 for i in change_times_d]

# # convert tv into datetime and add as new column
# fish_tracks['ts'] = pd.to_datetime(fish_tracks['tv_ns'], unit='ns')

# add new column with Day or Night
t0 = time.time()
fish_tracks['time_of_day_m'] = fish_tracks.ts.apply(lambda row: int(str(row)[11:16][:-3]) * 60 + int(str(row)[11:16][-2:]))
t1 = time.time()
print("time to load tracks {}".format(t1-t0))

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
for fish in fish_IDs:
    fish_tracks.loc[fish_tracks.FishID == fish, 'vertical_pos'] = vertical_pos.loc[:, fish]

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

# speed_mm (30m bins) for each fish (individual lines)
plot_speed_30m_individuals(rootdir, fish_tracks_30m, change_times_d)

# speed_mm (30m bins) for each species (mean  +- std)
plot_speed_30m_mstd(rootdir, fish_tracks_30m, change_times_d)

# movement for each fish (individual lines)
plot_movement_30m_individuals(rootdir, fish_tracks_30m, change_times_d)
# movement (30m bins) for each species (mean  +- std)
plot_movement_30m_mstd(rootdir, fish_tracks_30m, change_times_d)

##### get daily average #####

# make a new col where the daily timestamp is (no year/ month/ day)
for species_f in all_species:
    spd = fish_tracks_30m[fish_tracks_30m.species == species_f][['speed_mm', 'FishID', 'ts']]
    sp_spd = spd.pivot(columns='FishID', values='speed_mm', index='ts')
    # get time of day so that the same tod for each fish can be averaged
    sp_spd['time_of_day'] = sp_spd.apply(lambda row: str(row.name)[11:16], axis=1)
    sp_spd_ave = sp_spd.groupby('time_of_day').mean()
    sp_spd_ave_std = sp_spd_ave.std(axis=1)

    # speed_mm (30m bins daily average) for each fish (individual lines)
    plt.figure(figsize=(6, 4))
    # ax = sns.lineplot(data=sp_spd_ave)
    for cols in np.arange(0, sp_spd_ave.columns.shape[0]):
        ax = sns.lineplot(x=sp_spd_ave.index, y=(sp_spd_ave).iloc[:, cols])
    ax.axvspan(0, change_times_unit[0], color='lightblue', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[0], change_times_unit[1], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[2], change_times_unit[3], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[3], 24*2, color='lightblue', alpha=0.5,linewidth=0)
    ax.set_ylim([0, 60])
    ax.set_xlim([0, 24*2])
    plt.xlabel("Time (h:m)")
    plt.ylabel("Speed (mm/s)")
    ax.xaxis.set_major_locator(MultipleLocator(6))
    plt.title(species_f)
    plt.savefig(os.path.join(rootdir, "speed_30min_ave_individual{0}.png".format(species_f.replace(' ', '-'))))

    # speed_mm (30m bins daily average) for each fish (mean  +- std)
    plt.figure(figsize=(6, 4))
    sp_spd_ave_ave = sp_spd_ave.mean(axis=1)
    ax = sns.lineplot(x=sp_spd_ave.index, y=(sp_spd_ave_ave))
    ax = sns.lineplot(x=sp_spd_ave.index, y=(sp_spd_ave_ave + sp_spd_ave_std), color='lightgrey')
    ax = sns.lineplot(x=sp_spd_ave.index, y=(sp_spd_ave_ave - sp_spd_ave_std), color='lightgrey')

    ax.axvspan(0, change_times_unit[0], color='lightblue', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[0], change_times_unit[1], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[2], change_times_unit[3], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[3], 24*2, color='lightblue', alpha=0.5,linewidth=0)
    ax.set_ylim([0, 60])
    ax.set_xlim([0, 24 * 2])
    plt.xlabel("Time (h:m)")
    plt.ylabel("Speed (mm/s)")
    ax.xaxis.set_major_locator(MultipleLocator(6))
    plt.title(species_f)
    plt.savefig(os.path.join(rootdir, "speed_30min_ave_ave-stdev{0}.png".format(species_f.replace(' ', '-'))))


##### x,y position (binned day/night, and average day/night) #####



# speed vs Y position, for each fish, for combine fish of species, separated between day and night
spd_vs_y(meta, fish_tracks_30m, fish_IDs, rootdir)


# #### Behavioural state - calculated from Movement ####





# feature vector: for each fish readout vector of feature values
# version 1: Day/Night for: speed -  mean, stdev, median; y position - mean, stdev, median;
# FM - mean, stdev, median;
# to add later:
#   BS - mean, stdev, median
#   30min bins of each data


for species in all_species:
    df = pd.DataFrame([],  columns=['spd_mean_d', 'spd_mean_n', 'spd_std_d', 'spd_std_n', 'spd_median_d',
                                    'spd_median_n', 'move_mean_d', 'move_mean_n', 'move_std_d', 'move_std_n',
                                    'move_median_d', 'move_median_n', 'y_mean_d', 'y_mean_n', 'y_std_d', 'y_std_n',
                                    'y_median_d', 'y_median_n'])
    for fish in fish_IDs:
        fish_v_d = fish_tracks.loc[(fish_tracks.FishID == fish) & (fish_tracks.daynight == "d"),
                                            ["speed_mm", "movement", "vertical_pos"]]
        fish_v_n = fish_tracks.loc[(fish_tracks.FishID == fish) & (fish_tracks.daynight == "n"),
                                            ["speed_mm", "movement", "vertical_pos"]]

        df_f = pd.DataFrame([[fish_v_d.mean()[0], fish_v_n.mean()[0], fish_v_d.std()[0], fish_v_n.std()[0],
                              fish_v_d.median()[0], fish_v_n.median()[0], fish_v_d.mean()[1], fish_v_n.mean()[1],
                              fish_v_d.std()[1], fish_v_n.std()[1], fish_v_d.median()[1], fish_v_n.median()[1],
                              fish_v_d.mean()[2], fish_v_n.mean()[2], fish_v_d.std()[2], fish_v_n.std()[2],
                              fish_v_d.median()[2], fish_v_n.median()[2]]], index=[fish],
                            columns=['spd_mean_d', 'spd_mean_n', 'spd_std_d', 'spd_std_n', 'spd_median_d',
                                                 'spd_median_n', 'move_mean_d', 'move_mean_n', 'move_std_d',
                                                 'move_std_n', 'move_median_d', 'move_median_n', 'y_mean_d', 'y_mean_n',
                                                 'y_std_d', 'y_std_n', 'y_median_d', 'y_median_n'])
        df_f = df_f.round(4)
        df = pd.concat([df, df_f])

    df.to_csv(os.path.join(rootdir, "{}_als_fv.csv".format(species)))



# # group by
# plt.plot(fish_tracks.groupby["species"])
# fish_tracks_30m.groupby('species')['speed_mm'].plot(kind='line', x='ts', y='speed_mm')
# fish_tracks_30m.groupby('species').plot(kind='line', x='ts', y='speed_mm')
#
# date_form = DateFormatter("%H")
# ax = sns.lineplot(data=fish_tracks_30m, x='ts', y='speed_mm', hue='FishID')
# ax.xaxis.set_major_locator(MultipleLocator(0.2))
# ax.xaxis.set_major_formatter(date_form)
# fill_plot(ax, change_times_d, fish_tracks_30m[fish_tracks_30m.FishID == fish_IDs[0]].ts)
#
# fish_tracks_30m[fish_tracks_30m.FishID == fish_IDs[0]].ts

