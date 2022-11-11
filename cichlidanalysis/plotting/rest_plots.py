import os

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
from matplotlib.ticker import (MultipleLocator)
import numpy as np

from cichlidanalysis.plotting.speed_plots import fill_plot_ts
from cichlidanalysis.analysis.processing import norm_hist
from cichlidanalysis.io.meta import extract_meta


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
        plt.savefig(os.path.join(rootdir, "rest_{0}_individuals_{1}.png".format(ds_unit, species_f.replace(' ', '-'))))
        plt.close()


def plot_rest_sex(rootdir, fish_tracks_ds, change_times_d, fraction_threshold, time_window_s, ds_unit):
    """ Splitting by sex

    :param rootdir:
    :param fish_tracks_ds:
    :param change_times_d:
    :param fraction_threshold:
    :param time_window_s:
    :param ds_unit:
    :return:
    """
    # get each species
    all_species = fish_tracks_ds['species'].unique()
    date_form = DateFormatter("%H")

    for species_f in all_species:
        fig1, ax = plt.subplots(1, 1, figsize=(10, 4))
        date_form = DateFormatter("%H")
        ax = sns.lineplot(data=fish_tracks_ds[fish_tracks_ds.species == species_f], x='ts', y='rest', hue='sex',
                          units="FishID", estimator=None)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_major_formatter(date_form)
        fill_plot_ts(ax, change_times_d, fish_tracks_ds.ts)
        ax.set_ylim([0, 1])
        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., prop={'size': 6})
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Fraction rest")
        ax.set_title("Rest calculated from thresh {} and window {}".format(fraction_threshold, time_window_s))
        plt.tight_layout()
        plt.savefig(os.path.join(rootdir, "rest_{0}_individuals_by_sex_{1}.png".format(ds_unit, species_f.replace(' ',
                                                                                                                '-'))))
        plt.close()


        fig1, ax = plt.subplots(1, 1, figsize=(10, 4))
        date_form = DateFormatter("%H")
        ax = sns.lineplot(data=fish_tracks_ds[fish_tracks_ds.species == species_f], x='ts', y='rest', hue='sex')
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_major_formatter(date_form)
        fill_plot_ts(ax, change_times_d, fish_tracks_ds.ts)
        ax.set_ylim([0, 1])
        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., prop={'size': 6})
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Fraction rest")
        ax.set_title("Rest calculated from thresh {} and window {}".format(fraction_threshold, time_window_s))
        plt.tight_layout()
        plt.savefig(os.path.join(rootdir, "rest_{0}_mean-std_sex_{1}.png".format(ds_unit, species_f.replace(' ', '-'))))
        plt.close()


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
        sns.lineplot(x=rest_piv.index, y=average, color='darkorchid')
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_major_formatter(date_form)
        fill_plot_ts(ax, change_times_d, fish_tracks_ds[fish_tracks_ds.FishID == fish_IDs[0]].ts)
        ax.set_ylim([0, 1])
        plt.xlabel("Time (h)")
        plt.ylabel("Rest fraction")
        plt.title(species_f)
        plt.tight_layout()
        plt.savefig(os.path.join(rootdir, "rest_{0}_m-stdev{1}.png".format(ds_unit, species_f.replace(' ', '-'))))
        plt.close()


def plot_rest_bout_lengths_dn(fish_bouts, rootdir):
    """ Plot rest and nonrest bouts for a species

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
        fish_on_bouts_d = fish_bouts.loc[(fish_bouts['FishID'] == fish) & (fish_bouts['daynight'] == 'd'), "rest_len"]
        fish_on_bouts_n = fish_bouts.loc[(fish_bouts['FishID'] == fish) & (fish_bouts['daynight'] == 'n'), "rest_len"]
        fish_off_bouts_d = fish_bouts.loc[(fish_bouts['FishID'] == fish) & (fish_bouts['daynight'] == 'd'), "nonrest_len"]
        fish_off_bouts_n = fish_bouts.loc[(fish_bouts['FishID'] == fish) & (fish_bouts['daynight'] == 'n'), "nonrest_len"]

        bin_boxes_on = np.arange(0, 1000, 10)
        bin_boxes_off = np.arange(0, 60*60*10, 10)
        counts_on_bout_len_d, _, _ = ax1[0, 0].hist(fish_on_bouts_d.dt.total_seconds(), bins=bin_boxes_on, color='red')
        counts_on_bout_len_n, _, _ = ax1[1, 0].hist(fish_on_bouts_n.dt.total_seconds(), bins=bin_boxes_on, color='blue')
        counts_off_bout_len_d, _, _ = ax1[0, 1].hist(fish_off_bouts_d.dt.total_seconds(), bins=bin_boxes_off, color='red')
        counts_off_bout_len_n, _, _ = ax1[1, 1].hist(fish_off_bouts_n.dt.total_seconds(), bins=bin_boxes_off, color='blue')

        ax1[0, 0].set_ylabel("Day")
        ax1[1, 0].set_ylabel("Night")
        ax1[1, 0].set_xlabel("Rest bout lengths")
        ax1[1, 1].set_xlabel("Active bout lengths")
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
        ax2[0].set_xlabel("Rest bouts lengths (s)")
        ax2[1].set_xlabel("Active bouts lengths (s)")
        ax2[0].set_xlim([-1, 400])
        ax2[1].set_xlim([-1, 1000])
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
        ax3[0].set_ylabel("Fraction of rest bouts")
        ax3[1].set_ylabel("Fraction of active bouts")
        ax3[0].set_ylim([0, 1])
        ax3[1].set_ylim([0, 1])
        ax3[0].set_xlim([-20, 600])
        ax3[1].set_xlim([-20, 10000])
        ax3[0].legend()
        ax3[1].legend()
        fig3.suptitle("Cumulative movement bouts for {}".format(fish), fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(rootdir, "cumsum_rest_nonrest_bouts_{0}.png".format(species_n.replace(' ', '-'))))
        plt.close()
