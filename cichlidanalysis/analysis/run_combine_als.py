############################
# This module loads als and meta data of individual fish and plots the following for each species:
# speed_mm (30m bins, daily ave) for each fish (lines single and average as well as heatmap)
# x,y position (binned day/night, and average day/night)
# fraction mobile/immobile
# fraction active/quiescent
# bout structure (MI nad AQ, bout fraction in 30min bins, bouts D/N over days)

# For combined data this module  will plot:
# speed_mm (30m bins, daily ave) for each species (lines and heatmap)
# x,y position (binned day/night,  and average day/night)



from tkinter.filedialog import askdirectory, askopenfilename
from tkinter import *
import os
import glob
import warnings
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.dates import DateFormatter
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns


from cichlidanalysis.io.meta import load_meta_files
from cichlidanalysis.io.tracks import load_als_files
from cichlidanalysis.utils.timings import load_timings
from cichlidanalysis.analysis.processing import add_col
from cichlidanalysis.plotting.single_plots import fill_plot_ts


# debug pycharm problem
warnings.simplefilter(action='ignore', category=FutureWarning)

# useful
# data.columns
# making a pandas dataframe with tv_i (from middnight until 7days), x, y, speed_sm, FISH-ID, species, sex, fish_mm

# pick folder
# Allows a user to select top directory
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

fps, tv_ns, tv_sec, tv_24h_sec, num_days, tv_s_type, change_times_s, change_times_ns, change_times_h, day_ns, day_s,\
    change_times_d = load_timings(fish_tracks[fish_tracks.FishID == fish_IDs[0]].shape[0])

# convert tv into datetime and add as new column
fish_tracks['ts'] = pd.to_datetime(fish_tracks['tv_ns'], unit='ns')

# resample data
fish_tracks_30m = fish_tracks.groupby('FishID').resample('30T', on='ts').mean()
fish_tracks_30m.reset_index(inplace=True)

# add back 'species', 'sex'
for col_name in ['species', 'sex', 'fish_length_mm']:
    add_col(fish_tracks_30m, col_name, fish_IDs, meta)
all_species = fish_tracks_30m['species'].unique()

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

# speed_mm (30m bins) for each fish (mean  +- std)
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


# get daily average

# make a new col where the daily timestamp is (no year/ month/ day)
for species_f in all_species:
    spd = fish_tracks_30m[fish_tracks_30m.species == species_f][['speed_mm', 'FishID', 'ts']]
    sp_spd = spd.pivot(columns='FishID', values='speed_mm', index='ts')
    # get time of day so that the same tod for each fish can be averaged
    sp_spd['time_of_day'] = sp_spd.apply(lambda row: str(row.name)[11:20], axis=1)
    sp_spd_ave = sp_spd.groupby('time_of_day').mean()
    sp_spd_std = sp_spd.groupby('time_of_day').std()

    time_24h = sp_spd.time_of_day.apply(lambda row: datetime.datetime.strptime(row, '%H:%M:%S'))
    sp_spd = sp_spd.Index.rename(time_24h)

    plt.figure(figsize=(10, 4))
    ax = sns.lineplot(data=sp_spd_ave)
    # ax = sns.lineplot(data=sp_spd_ave, x=sp_spd.time_of_day, y=sp_spd_ave + sp_spd_std, hue='FishID')
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(date_form)
    fill_plot_ts(ax, change_times_d, time_24h)
    ax.set_ylim([0, 60])
    plt.xlabel("Time (h)")
    plt.ylabel("Speed (mm/s)")
    plt.title(species_f)

# speed_mm (30m bins daily average) for each fish (individual lines)



# speed_mm (30m bins daily average) for each fish (mean  +- std)




# fish_tracks_30m.groupby('FishID').plot(kind='line', x='ts', y='speed_mm')


# group by
plt.plot(fish_tracks.groupby["species"])
fish_tracks_30m.groupby('species')['speed_mm'].plot(kind='line', x='ts', y='speed_mm')
fish_tracks_30m.groupby('species').plot(kind='line', x='ts', y='speed_mm')

date_form = DateFormatter("%H")


ax = sns.lineplot(data=fish_tracks_30m, x='ts', y='speed_mm', hue='FishID')
ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.xaxis.set_major_formatter(date_form)
fill_plot(ax, change_times_d, fish_tracks_30m[fish_tracks_30m.FishID == fish_IDs[0]].ts)

fish_tracks_30m[fish_tracks_30m.FishID == fish_IDs[0]].ts

