############################
# This module loads als and meta data of individual fish and plots the following for each species:
# speed_mm (30m bins, daily ave) for each fish (lines single and average as well as heatmap)
# x,y position (binned day/night, and average day/night)
# fraction mobile/immobile
# fraction active/quiescent
# bout structure (MI and AQ, bout fraction in 30min bins, bouts D/N over days)

# For combined data this module  will plot:
# speed_mm (30m bins, daily ave) for each species (lines and heatmap)
# x,y position (binned day/night,  and average day/night)

from tkinter.filedialog import askdirectory
from tkinter import *
import os
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.ticker import (MultipleLocator)
import seaborn as sns


from cichlidanalysis.io.meta import load_meta_files
from cichlidanalysis.io.tracks import load_als_files
from cichlidanalysis.utils.timings import load_timings
from cichlidanalysis.analysis.processing import add_col
from cichlidanalysis.plotting.single_plots import fill_plot_ts
from cichlidanalysis.plotting.position_plots import spd_vs_y


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

fish_tracks = load_als_files(rootdir)
meta = load_meta_files(rootdir)
metat = meta.transpose()

# get each fish ID
fish_IDs = fish_tracks['FishID'].unique()

# get timings
fps, tv_ns, tv_sec, tv_24h_sec, num_days, tv_s_type, change_times_s, change_times_ns, change_times_h, day_ns, day_s,\
    change_times_d, change_times_m = load_timings(fish_tracks[fish_tracks.FishID == fish_IDs[0]].shape[0])
change_times_unit = [7.5*2, 8*2, 19*2, 19.5*2]

# # if from 7am - 7pm # need  better way to deal with this change!
# if statement for date (then add 30min to times stamps? as some fish will have ethis and others won't
# change_times_unit = [7*2, 7.5*2, 18.5*2, 19*2]
# change_times_s = [i - 1800 for i in change_times_s]
# change_times_h = [i - 0.5 for i in change_times_h]
# change_times_d = [i - .5/24 for i in change_times_d]

# convert tv into datetime and add as new column
fish_tracks['ts'] = pd.to_datetime(fish_tracks['tv_ns'], unit='ns')

# add new column with Day or Night
# fish_tracks['time_of_day'] = fish_tracks.ts.apply(lambda row: str(row)[11:16])
# fish_tracks['time_of_day_m'] = fish_tracks.time_of_day.apply(lambda row: int(row[:-3]) * 60 + int(row[-2:]))
fish_tracks['time_of_day_m'] = fish_tracks.ts.apply(lambda row: int(str(row)[11:16][:-3]) * 60 + int(str(row)[11:16][-2:]))

fish_tracks['daynight'] = "day"
fish_tracks.loc[fish_tracks.time_of_day_m < change_times_m[0], 'daynight'] = "night"
fish_tracks.loc[fish_tracks.time_of_day_m > change_times_m[3], 'daynight'] = "night"

# resample data
fish_tracks_30m = fish_tracks.groupby('FishID').resample('30T', on='ts').mean()
fish_tracks_30m.reset_index(inplace=True)

# add back 'species', 'sex'
for col_name in ['species', 'sex', 'fish_length_mm']:
    add_col(fish_tracks_30m, col_name, fish_IDs, meta)
all_species = fish_tracks_30m['species'].unique()

fish_tracks_30m['daynight'] = "day"
fish_tracks_30m.loc[fish_tracks_30m.time_of_day_m < change_times_m[0], 'daynight'] = "night"
fish_tracks_30m.loc[fish_tracks_30m.time_of_day_m > change_times_m[3], 'daynight'] = "night"

# speed_mm (30m bins) for each fish (individual lines)
date_form = DateFormatter("%H")
for species_f in all_species:
    plt.figure(figsize=(10, 4))
    ax = sns.lineplot(data=fish_tracks_30m[fish_tracks_30m.species == species_f], x='ts', y='speed_mm', hue='FishID')
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_major_formatter(date_form)
    fill_plot_ts(ax, change_times_d, fish_tracks_30m[fish_tracks_30m.FishID == fish_IDs[0]].ts)
    ax.set_ylim([0, 60])
    plt.xlabel("Time (h)")
    plt.ylabel("Speed (mm/s)")
    plt.title(species_f)
    plt.savefig(os.path.join(rootdir, "speed_30min_individual{0}.png".format(species_f.replace(' ', '-'))))

# speed_mm (30m bins) for each species (mean  +- std)
for species_f in all_species:
    # get speeds for each individual for a given species
    spd = fish_tracks_30m[fish_tracks_30m.species == species_f][['speed_mm', 'FishID', 'ts']]
    sp_spd = spd.pivot(columns='FishID', values='speed_mm', index='ts')

    # calculate ave and stdv
    average = sp_spd.mean(axis=1)
    stdv = sp_spd.std(axis=1)

    plt.figure(figsize=(10, 4))
    ax = sns.lineplot(x=sp_spd.index, y=average + stdv, color='lightgrey')
    sns.lineplot(x=sp_spd.index, y=average - stdv, color='lightgrey')
    sns.lineplot(x=sp_spd.index, y=average)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_major_formatter(date_form)
    fill_plot_ts(ax, change_times_d, fish_tracks_30m[fish_tracks_30m.FishID == fish_IDs[0]].ts)
    ax.set_ylim([0, 60])
    plt.xlabel("Time (h)")
    plt.ylabel("Speed (mm/s)")
    plt.title(species_f)
    plt.savefig(os.path.join(rootdir, "speed_30min_m-stdev{0}.png".format(species_f.replace(' ', '-'))))


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
# resample data
horizontal_pos = fish_tracks.pivot(columns="FishID", values="x_nt")
vertical_pos = fish_tracks.pivot(columns="FishID", values="y_nt")

# scale each fish by min/max
horizontal_pos -= horizontal_pos.min()
horizontal_pos /= horizontal_pos.max()
vertical_pos -= vertical_pos.min()
vertical_pos /= vertical_pos.max()
# flip Y axis
vertical_pos = abs(1- vertical_pos)

# split data into day and night
position_night_x = horizontal_pos.iloc[np.where(change_times_s[0] > tv_24h_sec)[0], ]
position_night_x = horizontal_pos.iloc[np.where(tv_24h_sec[0:-1] > change_times_s[3])[0], ]

position_night_y = vertical_pos.iloc[np.where(change_times_s[0] > tv_24h_sec)[0], ]
position_night_y = vertical_pos.iloc[np.where(tv_24h_sec[0:-1] > change_times_s[3])[0], ]

position_day_x = horizontal_pos.iloc[np.where((change_times_s[0] < tv_24h_sec[0:-1]) &
                                              (tv_24h_sec[0:-1] < change_times_s[3]))[0], ]
position_day_y = vertical_pos.iloc[np.where((change_times_s[0] < tv_24h_sec[0:-1]) &
                                            (tv_24h_sec[0:-1] < change_times_s[3]))[0], ]

# need to clean up data between fish, either use the vertical_pos/ horizontal_pos, or scale by x/ylim for x_nt, y_nt
individuals = True
fig1, ax1 = plt.subplots(2, len(meta.loc["species"].unique()))
for idx, species in enumerate(meta.loc["species"].unique()):
    position_day_x_sub = position_day_x.loc[:, (meta.loc["species"] == species)].to_numpy()
    position_day_y_sub = position_day_y.loc[:, (meta.loc["species"] == species)].to_numpy()
    position_night_x_sub = position_night_x.loc[:, (meta.loc["species"] == species)].to_numpy()
    position_night_y_sub = position_night_y.loc[:, (meta.loc["species"] == species)].to_numpy()

    if individuals:
        fig2, ax2 = plt.subplots(2,  position_day_x_sub.shape[1])
        fig2.suptitle("Individual fish averages for {}".format(species))
        for individ in np.arange(0, position_day_x_sub.shape[1]):
            position_day_xy, xedges_day, yedges_day, _ = plt.hist2d(position_day_x_sub[:, individ],
                                                                    position_day_y_sub[:, individ],
                                                                    bins=[3, 10], cmap='inferno', range=[[0, 1], [0, 1]])
            position_night_xy, xedges_night, yedges_night, _ = plt.hist2d(
                position_night_x_sub[:, individ],
                position_night_y_sub[:, individ],
                bins=[3, 10], cmap='inferno', range=[[0, 1], [0, 1]])

            # ax2[0, individ].set_title(individ)
            ax2[0, individ].imshow(position_day_xy.T)
            ax2[0, individ].invert_yaxis()
            ax2[0, individ].get_xaxis().set_ticks([])
            ax2[0, individ].get_yaxis().set_ticks([])
            ax2[1, individ].clear()
            ax2[1, individ].imshow(position_night_xy.T)
            ax2[1, individ].get_xaxis().set_ticks([])
            ax2[1, individ].get_yaxis().set_ticks([])
            ax2[1, individ].invert_yaxis()
            if individ == 0:
                ax2[0, individ].set_ylabel("Day")
                ax2[1, individ].set_ylabel("Night")
        plt.savefig(os.path.join(rootdir, "xy_ave_DN_individuals_{0}.png".format(species_f.replace(' ', '-'))))

    else:
    # reshape all the data
        position_day_x_sub = np.reshape(position_day_x_sub, position_day_x_sub.shape[0] * position_day_x_sub.shape[1])
        position_day_y_sub = np.reshape(position_day_y_sub, position_day_y_sub.shape[0] * position_day_y_sub.shape[1])
        position_night_x_sub = np.reshape(position_night_x_sub, position_night_x_sub.shape[0] * position_night_x_sub.shape[1])
        position_night_y_sub = np.reshape(position_night_y_sub, position_night_y_sub.shape[0] * position_night_y_sub.shape[1])

    # Creating bins
    x_min = 0
    x_max = np.nanmax(position_day_x_sub)

    y_min = 0
    y_max = np.nanmax(position_day_y_sub)

    x_bins = np.linspace(x_min, x_max, 4)
    y_bins = np.linspace(y_min, y_max, 11)

    fig3 = plt.figure(figsize=(4, 4))
    position_day_xy, xedges_day, yedges_day, _ = plt.hist2d(position_day_x_sub[~np.isnan(position_day_x_sub)],
                                                            position_day_y_sub[~np.isnan(position_day_y_sub)],
                     cmap='inferno', bins=[x_bins, y_bins])
    plt.close(fig3)

    # need to properly normalise by counts! To get frequency!!!!!!!!
    position_day_xy = (position_day_xy / sum(sum(position_day_xy)))*100
    fig3 = plt.figure(figsize=(4, 4))
    plt.imshow(position_day_xy.T, cmap='inferno', vmin=0, vmax=25)
    plt.title("Day")
    cbar = plt.colorbar(label="% occupancy")
    plt.gca().invert_yaxis()
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.savefig(os.path.join(rootdir, "xy_ave_Day_{0}.png".format(species_f.replace(' ', '-'))))

    fig4 = plt.figure(figsize=(4, 4))
    position_night_xy, xedges_night, yedges_night, _ = plt.hist2d(position_night_x_sub[~np.isnan(position_night_x_sub)],
                                                                  position_night_y_sub[~np.isnan(position_night_y_sub)],
                    bins=[3, 10], cmap='inferno')
    plt.close(fig4)

    position_night_xy = (position_night_xy / sum(sum(position_night_xy)))*100
    fig4 = plt.figure(figsize=(4, 4))
    plt.imshow(position_night_xy.T, cmap='inferno', vmin=0, vmax=25)
    plt.title("Night")
    cbar = plt.colorbar(label="% occupancy")
    plt.gca().invert_yaxis()
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.savefig(os.path.join(rootdir, "xy_ave_Night_{0}.png".format(species_f.replace(' ', '-'))))

    # find better way to deal with lack of second dimension when only one species
    if len(meta.loc["species"].unique()) == 1:
        ax1[0].set_title(species)
        ax1[0].set_ylabel("Day")
        ax1[0].imshow(position_day_xy.T, cmap='inferno')
        ax1[0].invert_yaxis()
        ax1[0].get_xaxis().set_ticks([])
        ax1[0].get_yaxis().set_ticks([])
        ax1[1].clear()
        ax1[1].imshow(position_night_xy.T, cmap='inferno')
        ax1[1].get_xaxis().set_ticks([])
        ax1[1].get_yaxis().set_ticks([])
        ax1[1].invert_yaxis()
        ax1[1].set_ylabel("Night")
    else:
        ax1[0, idx].title(species)
        ax1[0, idx].set_ylabel("Day")
        ax1[0, idx].imshow(position_day_xy.T)
        ax1[0, idx].invert_yaxis()
        ax1[0, idx].get_xaxis().set_ticks([])
        ax1[0, idx].get_yaxis().set_ticks([])
        ax1[1, idx].clear()
        ax1[1, idx].imshow(position_night_xy.T)
        ax1[1, idx].get_xaxis().set_ticks([])
        ax1[1, idx].get_yaxis().set_ticks([])
        ax1[1, idx].invert_yaxis()
        ax1[1, idx].set_ylabel("Night")
fig1.savefig(os.path.join(rootdir, "xy_ave_DN_all.png"))


# speed vs Y position, for each fish, for combine fish of species, separated between day and night
spd_vs_y(meta, fish_tracks, fish_tracks_30m, fish_IDs, rootdir)

# fraction mobile/immobile








# fraction active/quiescent




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

