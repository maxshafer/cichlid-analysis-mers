import os

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.ticker import (MultipleLocator)
import seaborn as sns

from cichlidanalysis.plotting.single_plots import fill_plot_ts


# speed_mm (30m bins) for each fish (individual lines)
def plot_speed_30m_individuals(rootdir, fish_tracks_30m, change_times_d):
    # get each species
    all_species = fish_tracks_30m['species'].unique()
    # get each fish ID
    fish_IDs = fish_tracks_30m['FishID'].unique()

    date_form = DateFormatter("%H")
    for species_f in all_species:
        plt.figure(figsize=(10, 4))
        ax = sns.lineplot(data=fish_tracks_30m[fish_tracks_30m.species == species_f], x='ts', y='speed_mm',
                          hue='FishID')
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_major_formatter(date_form)
        fill_plot_ts(ax, change_times_d, fish_tracks_30m[fish_tracks_30m.FishID == fish_IDs[0]].ts)
        ax.set_ylim([0, 60])
        plt.xlabel("Time (h)")
        plt.ylabel("Speed (mm/s)")
        plt.title(species_f)
        plt.savefig(os.path.join(rootdir, "speed_30min_individual{0}.png".format(species_f.replace(' ', '-'))))


# speed_mm (30m bins) for each species (mean  +- std)
def plot_speed_30m_mstd(rootdir, fish_tracks_30m, change_times_d):
    # get each species
    all_species = fish_tracks_30m['species'].unique()
    # get each fish ID
    fish_IDs = fish_tracks_30m['FishID'].unique()
    date_form = DateFormatter("%H")

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
