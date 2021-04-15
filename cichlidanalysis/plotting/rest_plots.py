import os

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
from matplotlib.ticker import (MultipleLocator)

from cichlidanalysis.plotting.speed_plots import fill_plot_ts


def plot_rest_ind(rootdir, fish_tracks_ds, change_times_d, fraction_threshold, time_window_s, ds_unit):
    # get each species
    all_species = fish_tracks_ds['species'].unique()
    date_form = DateFormatter("%H")

    for species_f in all_species:
        fig1, ax = plt.subplots(1, 1, figsize=(10, 4))
        date_form = DateFormatter("%H")
        ax = sns.lineplot(data=fish_tracks_ds[fish_tracks_ds.species == species_f], x='ts', y='rest', hue='FishID')
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_major_formatter(date_form)
        fill_plot_ts(ax, change_times_d, fish_tracks_ds.ts)
        ax.set_ylim([0, 1])
        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., prop={'size': 6})
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Fraction rest")
        ax.set_title("Rest calculated from thresh {} and window {}".format(fraction_threshold, time_window_s))
        plt.tight_layout()
        print("defining behavioural state")
        plt.show()
        plt.savefig(os.path.join(rootdir, "rest_{0}_individuals_{1}.png".format(ds_unit, species_f.replace(' ', '-'))))



def plot_rest_mstd(rootdir, fish_tracks_ds, change_times_d, ds_unit):
    # speed_mm (30m bins) for each species (mean  +- std)
    # get each species
    all_species = fish_tracks_ds['species'].unique()
    # get each fish ID
    fish_IDs = fish_tracks_ds['FishID'].unique()
    date_form = DateFormatter("%H")

    for species_f in all_species:
        # get rest for each individual for a given species
        rest = fish_tracks_ds[fish_tracks_ds.species == species_f][['rest', 'FishID', 'ts']]
        rest_piv = rest.pivot(columns='FishID', values='rest', index='ts')

        # calculate ave and stdv
        average = rest_piv.mean(axis=1)
        stdv = rest_piv.std(axis=1)

        plt.figure(figsize=(10, 4))
        ax = sns.lineplot(x=rest_piv.index, y=average + stdv, color='lightgrey')
        sns.lineplot(x=rest_piv.index, y=average - stdv, color='lightgrey')
        sns.lineplot(x=rest_piv.index, y=average)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_major_formatter(date_form)
        fill_plot_ts(ax, change_times_d, fish_tracks_ds[fish_tracks_ds.FishID == fish_IDs[0]].ts)
        ax.set_ylim([0, 1])
        plt.xlabel("Time (h)")
        plt.ylabel("Rest fraction")
        plt.title(species_f)
        plt.tight_layout()
        plt.savefig(os.path.join(rootdir, "rest_{0}_m-stdev{1}.png".format(ds_unit, species_f.replace(' ', '-'))))
