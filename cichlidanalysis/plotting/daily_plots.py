import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)
import seaborn as sns


def daily_ave_spd(sp_spd_ave, sp_spd_ave_std, rootdir, species_f, change_times_unit):
    """ speed_mm (30m bins daily average) for each fish (individual lines)

    :param sp_spd_ave:
    :param sp_spd_ave_std:
    :param rootdir:
    :param species_f:
    :param change_times_unit:
    :return: daily_speed:
    """
    plt.figure(figsize=(6, 4))
    for cols in np.arange(0, sp_spd_ave.columns.shape[0]):
        ax = sns.lineplot(x=sp_spd_ave.index, y=(sp_spd_ave).iloc[:, cols])
    ax.axvspan(0, change_times_unit[0], color='lightblue', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[0], change_times_unit[1], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[2], change_times_unit[3], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[3], 24 * 2, color='lightblue', alpha=0.5, linewidth=0)
    ax.set_ylim([0, 60])
    ax.set_xlim([0, 24 * 2])
    plt.xlabel("Time (h:m)")
    plt.ylabel("Speed (mm/s)")
    ax.xaxis.set_major_locator(MultipleLocator(6))
    plt.title(species_f)
    plt.savefig(os.path.join(rootdir, "speed_30min_ave_individual{0}.png".format(species_f.replace(' ', '-'))))
    plt.close()

    # speed_mm (30m bins daily average) for each fish (mean  +- std)
    plt.figure(figsize=(6, 4))
    daily_speed = sp_spd_ave.mean(axis=1)
    ax = sns.lineplot(x=sp_spd_ave.index, y=(daily_speed))
    ax = sns.lineplot(x=sp_spd_ave.index, y=(daily_speed + sp_spd_ave_std), color='lightgrey')
    ax = sns.lineplot(x=sp_spd_ave.index, y=(daily_speed - sp_spd_ave_std), color='lightgrey')

    ax.axvspan(0, change_times_unit[0], color='lightblue', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[0], change_times_unit[1], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[2], change_times_unit[3], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[3], 24 * 2, color='lightblue', alpha=0.5, linewidth=0)
    ax.set_ylim([0, 60])
    ax.set_xlim([0, 24 * 2])
    plt.xlabel("Time (h:m)")
    plt.ylabel("Speed (mm/s)")
    ax.xaxis.set_major_locator(MultipleLocator(6))
    plt.title(species_f)
    plt.savefig(os.path.join(rootdir, "speed_30min_ave_ave-stdev{0}.png".format(species_f.replace(' ', '-'))))
    plt.close()


def daily_ave_move(sp_move_ave, sp_move_ave_std, rootdir, species_f, change_times_unit):
    """

    :param sp_move_ave:
    :param sp_move_ave_std:
    :param rootdir:
    :param species_f:
    :param change_times_unit:
    :return:
    """

    plt.figure(figsize=(6, 4))
    for cols in np.arange(0, sp_move_ave.columns.shape[0]):
        ax = sns.lineplot(x=sp_move_ave.index, y=(sp_move_ave).iloc[:, cols])
    ax.axvspan(0, change_times_unit[0], color='lightblue', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[0], change_times_unit[1], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[2], change_times_unit[3], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[3], 24 * 2, color='lightblue', alpha=0.5, linewidth=0)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 24 * 2])
    plt.xlabel("Time (h:m)")
    plt.ylabel("Movement")
    ax.xaxis.set_major_locator(MultipleLocator(6))
    plt.title(species_f)
    plt.savefig(os.path.join(rootdir, "movement_30min_ave_individual{0}.png".format(species_f.replace(' ', '-'))))
    plt.close()

    # movement (30m bins daily average) for each fish (mean  +- std)
    plt.figure(figsize=(6, 4))
    daily_move = sp_move_ave.mean(axis=1)
    ax = sns.lineplot(x=sp_move_ave.index, y=(daily_move + sp_move_ave_std), color='lightgrey')
    ax = sns.lineplot(x=sp_move_ave.index, y=(daily_move - sp_move_ave_std), color='lightgrey')
    ax = sns.lineplot(x=sp_move_ave.index, y=(daily_move))

    ax.axvspan(0, change_times_unit[0], color='lightblue', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[0], change_times_unit[1], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[2], change_times_unit[3], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[3], 24 * 2, color='lightblue', alpha=0.5, linewidth=0)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 24 * 2])
    plt.xlabel("Time (h:m)")
    plt.ylabel("Movement")
    ax.xaxis.set_major_locator(MultipleLocator(6))
    plt.title(species_f)
    plt.savefig(os.path.join(rootdir, "movement_30min_ave_ave-stdev{0}.png".format(species_f.replace(' ', '-'))))
    plt.close()


def daily_ave_rest(sp_rest_ave, sp_rest_ave_std, rootdir, species_f, change_times_unit):

    plt.figure(figsize=(6, 4))
    for cols in np.arange(0, sp_rest_ave.columns.shape[0]):
        ax = sns.lineplot(x=sp_rest_ave.index, y=(sp_rest_ave).iloc[:, cols])
    ax.axvspan(0, change_times_unit[0], color='lightblue', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[0], change_times_unit[1], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[2], change_times_unit[3], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[3], 24 * 2, color='lightblue', alpha=0.5, linewidth=0)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 24 * 2])
    plt.xlabel("Time (h:m)")
    plt.ylabel("Rest")
    ax.xaxis.set_major_locator(MultipleLocator(6))
    plt.title(species_f)
    plt.savefig(os.path.join(rootdir, "Rest_30min_ave_individual{0}.png".format(species_f.replace(' ', '-'))))
    plt.close()

    # rest (30m bins daily average) for each fish (mean  +- std)
    plt.figure(figsize=(6, 4))
    daily_rest = sp_rest_ave.mean(axis=1)
    ax = sns.lineplot(x=sp_rest_ave.index, y=(daily_rest + sp_rest_ave_std), color='lightgrey')
    ax = sns.lineplot(x=sp_rest_ave.index, y=(daily_rest - sp_rest_ave_std), color='lightgrey')
    ax = sns.lineplot(x=sp_rest_ave.index, y=(daily_rest))

    ax.axvspan(0, change_times_unit[0], color='lightblue', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[0], change_times_unit[1], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[2], change_times_unit[3], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[3], 24 * 2, color='lightblue', alpha=0.5, linewidth=0)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 24 * 2])
    plt.xlabel("Time (h:m)")
    plt.ylabel("Rest")
    ax.xaxis.set_major_locator(MultipleLocator(6))
    plt.title(species_f)
    plt.savefig(os.path.join(rootdir, "rest_30min_ave_ave-stdev{0}.png".format(species_f.replace(' ', '-'))))
    plt.close()
    return daily_rest


# make a new col where the daily timestamp is (no year/ month/ day)
def plot_daily(fish_tracks_30m_i, change_times_unit, rootdir):

    all_species = fish_tracks_30m_i['species'].unique()

    for species_f in all_species:
        # ### speed ###
        spd = fish_tracks_30m_i[fish_tracks_30m_i.species == species_f][['speed_mm', 'FishID', 'ts']]
        sp_spd = spd.pivot(columns='FishID', values='speed_mm', index='ts')

        # get time of day so that the same tod for each fish can be averaged
        sp_spd['time_of_day'] = sp_spd.apply(lambda row: str(row.name)[11:16], axis=1)
        sp_spd_ave = sp_spd.groupby('time_of_day').mean()
        sp_spd_ave_std = sp_spd_ave.std(axis=1)

        # make the plots
        daily_ave_spd(sp_spd_ave, sp_spd_ave_std, rootdir, species_f, change_times_unit)

        # ### movement ###
        move = fish_tracks_30m_i[fish_tracks_30m_i.species == species_f][['movement', 'FishID', 'ts']]
        sp_move = move.pivot(columns='FishID', values='movement', index='ts')

        # get time of day so that the same tod for each fish can be averaged
        sp_move['time_of_day'] = sp_move.apply(lambda row: str(row.name)[11:16], axis=1)
        sp_move_ave = sp_move.groupby('time_of_day').mean()
        sp_move_ave_std = sp_move_ave.std(axis=1)

        # make the plots
        daily_ave_move(sp_move_ave, sp_move_ave_std, rootdir, species_f, change_times_unit)

        # ### rest ###
        rest = fish_tracks_30m_i[fish_tracks_30m_i.species == species_f][['rest', 'FishID', 'ts']]
        sp_rest = rest.pivot(columns='FishID', values='rest', index='ts')

        # get time of day so that the same tod for each fish can be averaged
        sp_rest['time_of_day'] = sp_rest.apply(lambda row: str(row.name)[11:16], axis=1)
        sp_rest_ave = sp_rest.groupby('time_of_day').mean()
        sp_rest_ave_std = sp_rest_ave.std(axis=1)

        # make the plots
        daily_ave_rest(sp_rest_ave, sp_rest_ave_std, rootdir, species_f, change_times_unit)
