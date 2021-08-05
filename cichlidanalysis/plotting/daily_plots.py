import os

from matplotlib.dates import DateFormatter
import matplotlib.cm as cm
import pandas as pd
import matplotlib.gridspec as grid_spec
import datetime as dt
from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)
import seaborn as sns

from cichlidanalysis.utils.species_names import six_letter_sp_name


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


def daily_ave_vp(rootdir, sp_vp_ave, sp_vp_ave_std, species_f, change_times_unit):

    plt.figure(figsize=(6, 4))
    for cols in np.arange(0, sp_vp_ave.columns.shape[0]):
        ax = sns.lineplot(x=sp_vp_ave.index, y=(sp_vp_ave).iloc[:, cols])
    ax.axvspan(0, change_times_unit[0], color='lightblue', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[0], change_times_unit[1], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[2], change_times_unit[3], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[3], 24 * 2, color='lightblue', alpha=0.5, linewidth=0)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 24 * 2])
    plt.xlabel("Time (h:m)")
    plt.ylabel("Vertical position")
    ax.xaxis.set_major_locator(MultipleLocator(6))
    plt.title(species_f)
    plt.savefig(os.path.join(rootdir, "vp_30min_ave_individual{0}.png".format(species_f.replace(' ', '-'))))
    plt.close()

    # rest (30m bins daily average) for each fish (mean  +- std)
    plt.figure(figsize=(6, 4))
    daily_rest = sp_vp_ave.mean(axis=1)
    ax = sns.lineplot(x=sp_vp_ave.index, y=(daily_rest + sp_vp_ave_std), color='lightgrey')
    ax = sns.lineplot(x=sp_vp_ave.index, y=(daily_rest - sp_vp_ave_std), color='lightgrey')
    ax = sns.lineplot(x=sp_vp_ave.index, y=(daily_rest))

    ax.axvspan(0, change_times_unit[0], color='lightblue', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[0], change_times_unit[1], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[2], change_times_unit[3], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[3], 24 * 2, color='lightblue', alpha=0.5, linewidth=0)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 24 * 2])
    plt.xlabel("Time (h:m)")
    plt.ylabel("Vertical position")
    ax.xaxis.set_major_locator(MultipleLocator(6))
    plt.title(species_f)
    plt.savefig(os.path.join(rootdir, "vp_30min_ave_ave-stdev{0}.png".format(species_f.replace(' ', '-'))))
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

        # ### vertical position ###
        vertical_pos = fish_tracks_30m_i[fish_tracks_30m_i.species == species_f][['vertical_pos', 'FishID', 'ts']]
        sp_vp = vertical_pos.pivot(columns='FishID', values='vertical_pos', index='ts')

        # get time of day so that the same tod for each fish can be averaged
        sp_vp['time_of_day'] = sp_vp.apply(lambda row: str(row.name)[11:16], axis=1)
        sp_vp_ave = sp_vp.groupby('time_of_day').mean()
        sp_vp_ave_std = sp_vp_ave.std(axis=1)

        # make the plots
        daily_ave_vp(rootdir, sp_vp_ave, sp_vp_ave_std, species_f, change_times_unit)

#
# def plot_daily_ridge_plot(rootdir, aves_ave_feature, feature, ymax, span_max, ylabeling, change_times_datetime_i):
#     """ ridgeplot but of individuals, want to order by clustering.
#
#     :param rootdir:
#     :param aves_ave_feature:
#     :param feature:
#     :param ymax:
#     :param span_max:
#     :param ylabeling:
#     :param change_times_datetime_i:
#     :return:
#     """
#
#     ymax =  20
#     span_max = 20
#     species = aves_ave_feature.columns
#     sorted_index = aves_ave_feature.groupby(by='species_six').mean().sort_values(by='peak_amplitude').index
#
#     cmap = cm.get_cmap('turbo')
#     colour_array = np.arange(0, 1, 1 / len(species))
#
#     date_form = DateFormatter('%H:%M:%S')
#
#     gs = grid_spec.GridSpec(len(species), 1)
#     fig = plt.figure(figsize=(4, 9))
#     ax_objs = []
#     averages = np.zeros([len(species), 303])
#
#     first = 1
#     for species_n, species_name in enumerate(species):
#         date_time_obj = []
#         for i in aves_ave_feature.index:
#             date_time_obj.append(dt.datetime.strptime(i, '%H:%M') + timedelta(days=365*70+18))
#
#         # creating new axes object
#         ax_objs.append(fig.add_subplot(gs[species_n:species_n + 1, 0:]))
#         days_to_plot = 1
#
#         ax_objs[-1].fill_between([dt.datetime.strptime("1970-1-2 00:00:00", '%Y-%m-%d %H:%M:%S'),
#                                   change_times_datetime_i[0]], [span_max, span_max], 0,
#                                  color='lightblue', alpha=0.5, linewidth=0, zorder=1)
#         ax_objs[-1].fill_between([change_times_datetime_i[0], change_times_datetime_i[1]], [span_max, span_max], 0,
#                                  color='wheat', alpha=0.5, linewidth=0)
#         ax_objs[-1].fill_between([change_times_datetime_i[2], change_times_datetime_i[3]], [span_max, span_max], 0,
#                                  color='wheat', alpha=0.5, linewidth=0)
#         ax_objs[-1].fill_between([change_times_datetime_i[3], change_times_datetime_i[4]], [span_max, span_max], 0,
#                                  color='lightblue', alpha=0.5, linewidth=0)
#
#         # plotting the distribution
#         ax_objs[-1].plot(date_time_obj, aves_ave_feature.loc[:, species_name], lw=1, color='w')
#         ax_objs[-1].fill_between(date_time_obj, aves_ave_feature.loc[:, species_name], 0, color=cmap(colour_array[species_n]), zorder=2)
#
#         # setting uniform x and y lims
#         ax_objs[-1].set_xlim(min(date_time_obj), dt.datetime.strptime("1970-1-2 23:59:59", '%Y-%m-%d %H:%M:%S'))
#         ax_objs[-1].set_ylim(0, ymax)
#
#         # make background transparent
#         rect = ax_objs[-1].patch
#         rect.set_alpha(0)
#
#         if species_n == len(species) - 1:
#             ax_objs[-1].set_xlabel("Time", fontsize=10, fontweight="bold")
#             ax_objs[-1].xaxis.set_major_locator(MultipleLocator(20))
#             ax_objs[-1].xaxis.set_major_formatter(date_form)
#             ax_objs[-1].yaxis.tick_right()
#             ax_objs[-1].yaxis.set_label_position("right")
#             ax_objs[-1].set_ylabel(ylabeling)
#
#         else:
#             # remove borders, axis ticks, and labels
#             ax_objs[-1].set_xticklabels([])
#             ax_objs[-1].set_xticks([])
#             ax_objs[-1].set_yticks([])
#             ax_objs[-1].set_yticklabels([])
#             ax_objs[-1].set_ylabel('')
#
#         spines = ["top", "right", "left", "bottom"]
#         for s in spines:
#             ax_objs[-1].spines[s].set_visible(False)
#
#         ax_objs[-1].text(0.9, 0, species_name, fontweight="bold", fontsize=10, ha="right", rotation=-45)
#         gs.update(hspace=-0.1)
#     plt.show()
#
#     plt.savefig(os.path.join(rootdir, "{0}_30min_combined_species_{1}.png".format(feature, dt.date.today())))
#     plt.close()
#     aves_feature = pd.DataFrame(averages.T, columns=species, index=date_time_obj[0:averages.shape[1]])