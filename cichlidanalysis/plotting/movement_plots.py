import os

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.ticker import (MultipleLocator)
import seaborn as sns
import numpy as np

from cichlidanalysis.plotting.single_plots import fill_plot_ts
from cichlidanalysis.io.meta import extract_meta
from cichlidanalysis.analysis.processing import norm_hist


def plot_movement_30m_individuals(rootdir, fish_tracks_30m, change_times_d, move_thresh):
    """ movement (30m bins) for each fish (individual lines)

    :param rootdir:
    :param fish_tracks_30m:
    :param change_times_d:
    :param move_thresh:
    :return:
    """
    # get each species
    all_species = fish_tracks_30m['species'].unique()
    # get each fish ID
    fish_IDs = fish_tracks_30m['FishID'].unique()

    date_form = DateFormatter("%H")
    for species_f in all_species:
        plt.figure(figsize=(10, 4))
        ax = sns.lineplot(data=fish_tracks_30m[fish_tracks_30m.species == species_f], x='ts', y='movement',
                          hue='FishID')
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_major_formatter(date_form)
        fill_plot_ts(ax, change_times_d, fish_tracks_30m[fish_tracks_30m.FishID == fish_IDs[0]].ts)
        ax.set_ylim([0, 1])
        plt.xlabel("Time (h)")
        plt.ylabel("Fraction moving")
        plt.title(species_f)
        plt.savefig(os.path.join(rootdir, "fraction_moving_30min_move_thresh-{0}_individual{1}.png".format(move_thresh,
                                          species_f.replace(' ', '-'))))
        plt.close()


def plot_movement_30m_sex(rootdir, fish_tracks_30m, change_times_d, move_thresh):
    """

    :param rootdir:
    :param fish_tracks_30m:
    :param change_times_d:
    :param move_thresh:
    :return:
    """
    # get each species
    all_species = fish_tracks_30m['species'].unique()
    # get each fish ID
    fish_IDs = fish_tracks_30m['FishID'].unique()

    date_form = DateFormatter("%H")
    for species_f in all_species:
        plt.figure(figsize=(10, 4))
        ax = sns.lineplot(data=fish_tracks_30m[fish_tracks_30m.species == species_f], x='ts', y='movement',
                          hue='sex', units="FishID", estimator=None)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_major_formatter(date_form)
        fill_plot_ts(ax, change_times_d, fish_tracks_30m[fish_tracks_30m.FishID == fish_IDs[0]].ts)
        ax.set_ylim([0, 1])
        plt.xlabel("Time (h)")
        plt.ylabel("Fraction moving")
        plt.title(species_f)
        plt.savefig(
            os.path.join(rootdir, "fraction_moving_30min_move_thresh-{0}_individual_by_sex_{1}.png".format(move_thresh,
                        species_f.replace(' ', '-'))))

        plt.figure(figsize=(10, 4))
        ax = sns.lineplot(data=fish_tracks_30m[fish_tracks_30m.species == species_f], x='ts', y='movement',
                          hue='sex')
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_major_formatter(date_form)
        fill_plot_ts(ax, change_times_d, fish_tracks_30m[fish_tracks_30m.FishID == fish_IDs[0]].ts)
        ax.set_ylim([0, 1])
        plt.xlabel("Time (h)")
        plt.ylabel("Fraction moving")
        plt.title(species_f)
        plt.savefig(os.path.join(rootdir, "fraction_moving_30min_move_thresh-{0}_mean_std_sex{1}.png".format(
            move_thresh, species_f.replace(' ', '-'))))
        plt.close()


# movement (30m bins) for each species (mean  +- std)
def plot_movement_30m_mstd(rootdir, fish_tracks_30m, change_times_d, move_thresh):
    # get each species
    all_species = fish_tracks_30m['species'].unique()
    # get each fish ID
    fish_IDs = fish_tracks_30m['FishID'].unique()
    date_form = DateFormatter("%H")

    for species_f in all_species:
        # get speeds for each individual for a given species
        spd = fish_tracks_30m[fish_tracks_30m.species == species_f][['movement', 'FishID', 'ts']]
        sp_spd = spd.pivot(columns='FishID', values='movement', index='ts')

        # calculate ave and stdv
        average = sp_spd.mean(axis=1)
        stdv = sp_spd.std(axis=1)

        plt.figure(figsize=(10, 4))
        ax = sns.lineplot(x=sp_spd.index, y=average + stdv, color='lightgrey')
        sns.lineplot(x=sp_spd.index, y=average - stdv, color='lightgrey')
        sns.lineplot(x=sp_spd.index, y=average, color='palevioletred')
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_major_formatter(date_form)
        fill_plot_ts(ax, change_times_d, fish_tracks_30m[fish_tracks_30m.FishID == fish_IDs[0]].ts)
        ax.set_ylim([0, 1])
        plt.xlabel("Time (h)")
        plt.ylabel("Fraction movement")
        plt.title(species_f)
        plt.savefig(os.path.join(rootdir, "fraction_movement_30min_move_thresh-{0}_m-stdev{1}.png".format(move_thresh,
                                          species_f.replace(' ', '-'))))
        plt.close()


def plot_bout_lengths_dn_move(fish_bouts, rootdir):
    """ Plot movement and immobile bouts for a species

    :param fish_bouts:
    :param rootdir:
    :return:
    """
    fishes = fish_bouts['FishID'].unique()
    species = set()
    for fish in fishes:
        fish_data = extract_meta(fish)
        species.add(fish_data['species'])

    for species_n in species:
        # counts of bout lengths for on and off bout
        fig1, ax1 = plt.subplots(2, 2)
        fish_on_bouts_d = fish_bouts.loc[(fish_bouts['FishID'] == fish) & (fish_bouts['daynight'] == 'd'), "movement_len"]
        fish_on_bouts_n = fish_bouts.loc[(fish_bouts['FishID'] == fish) & (fish_bouts['daynight'] == 'n'), "movement_len"]
        fish_off_bouts_d = fish_bouts.loc[(fish_bouts['FishID'] == fish) & (fish_bouts['daynight'] == 'd'), "nonmovement_len"]
        fish_off_bouts_n = fish_bouts.loc[(fish_bouts['FishID'] == fish) & (fish_bouts['daynight'] == 'n'), "nonmovement_len"]

        bin_boxes_on = np.arange(0, 100, 0.5)
        bin_boxes_off = np.arange(0, 100, 0.5)
        counts_on_bout_len_d, _, _ = ax1[0, 0].hist(fish_on_bouts_d.dt.total_seconds(), bins=bin_boxes_on, color='red')
        counts_on_bout_len_n, _, _ = ax1[1, 0].hist(fish_on_bouts_n.dt.total_seconds(), bins=bin_boxes_on, color='blue')
        counts_off_bout_len_d, _, _ = ax1[0, 1].hist(fish_off_bouts_d.dt.total_seconds(), bins=bin_boxes_off, color='red')
        counts_off_bout_len_n, _, _ = ax1[1, 1].hist(fish_off_bouts_n.dt.total_seconds(), bins=bin_boxes_off, color='blue')

        ax1[0, 0].set_ylabel("Day")
        ax1[1, 0].set_ylabel("Night")
        ax1[1, 0].set_xlabel("immobile bout lengths")
        ax1[1, 1].set_xlabel("movement bout lengths")
        plt.close()

        # normalised fractions of bout lengths for on and off bout
        counts_on_bout_len_d_norm = norm_hist(counts_on_bout_len_d)
        counts_on_bout_len_n_norm = norm_hist(counts_on_bout_len_n)
        counts_off_bout_len_d_norm = norm_hist(counts_off_bout_len_d)
        counts_off_bout_len_n_norm = norm_hist(counts_off_bout_len_n)

        fig2, ax2 = plt.subplots(1, 2)
        ax2[0].plot(bin_boxes_on[0:-1], counts_on_bout_len_d_norm, 'r', label='Day')
        ax2[0].plot(bin_boxes_on[0:-1], counts_on_bout_len_n_norm, color='blueviolet', label='Night')
        ax2[1].plot(bin_boxes_off[0:-1], counts_off_bout_len_d_norm, color='indianred', label='Day')
        ax2[1].plot(bin_boxes_off[0:-1], counts_off_bout_len_n_norm, color='darkblue', label='Night')
        ax2[0].set_ylabel("Fraction")
        ax2[0].set_xlabel("movement bouts lengths (s)")
        ax2[1].set_xlabel("immobile bouts lengths (s)")
        ax2[0].set_xlim([-1, 50])
        ax2[1].set_xlim([-1, 50])
        ax2[0].set_ylim([0, 1])
        ax2[1].set_ylim([0, 1])
        # ax2[0].set_yscale('log')
        # ax2[1].set_yscale('log')
        ax2[1].legend()
        plt.close()

        # cumulative sum of rest/active bout lengths
        fig3, ax3 = plt.subplots(1, 2)
        ax3[0].plot(bin_boxes_on[0:-1], np.cumsum(counts_on_bout_len_d_norm), color='red', label='Day')
        ax3[0].plot(bin_boxes_on[0:-1], np.cumsum(counts_on_bout_len_n_norm), color='blueviolet', label='Night')
        ax3[1].plot(bin_boxes_off[0:-1], np.cumsum(counts_off_bout_len_d_norm), color='indianred', label='Day')
        ax3[1].plot(bin_boxes_off[0:-1], np.cumsum(counts_off_bout_len_n_norm), color='darkblue', label='Night')
        ax3[0].set_xlabel("Bout length (s)")
        ax3[1].set_xlabel("Bout length (s)")
        ax3[0].set_ylabel("Fraction of movement bouts")
        ax3[1].set_ylabel("Fraction of immobile bouts")
        ax3[0].set_ylim([0, 1])
        ax3[1].set_ylim([0, 1])
        ax3[0].set_xlim([-2, 100])
        ax3[1].set_xlim([-2, 100])
        ax3[0].legend()
        ax3[1].legend()
        fig3.suptitle("Cumulative movement bouts for {}".format(fish), fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(rootdir, "cumsum_movement_bouts_{0}.png".format(species_n.replace(' ', '-'))))
        plt.close()
