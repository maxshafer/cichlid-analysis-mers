import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec
import matplotlib.cm as cm
import scipy.signal as signal
import seaborn as sns
from matplotlib.dates import DateFormatter
from matplotlib.ticker import (MultipleLocator)

from cichlidanalysis.analysis.bouts import find_bouts
from cichlidanalysis.analysis.processing import smooth_speed, standardise_cols
from cichlidanalysis.analysis.bs_clustering import kmeans_cluster
from cichlidanalysis.plotting.speed_plots import fill_plot_ts


def norm_hist(input_d):
    """ Normalise input by total number e.g. fraction"""
    input_d_norm = input_d / sum(input_d)
    return input_d_norm


def define_long_states(fish_tracks_i, change_times_d, time_window_s=[5, 15, 30, 60, 120, 300]):
    """ Defines behavioural state by thresholding on a window
    Sleep as defined by not moving for > X seconds

    :param fish_tracks_i:
    :param change_times_d:
    :param time_window_s:
    :return:
    """
    fps = 10

    fig1, ax = plt.subplots(1, 1)
    # fig2, ax2 = plt.subplots(1, 1)
    date_form = DateFormatter("%H")
    first = True

    for win in time_window_s:
        win_f = fps * win
        fish_tracks_i['rest'] = ((fish_tracks_i.groupby('FishID').movement.apply(lambda x: x.rolling(win_f).sum()) == 0) * 1)
        # if first:
        #     fish_tracks_c = copy.copy(fish_tracks_i)
        #     fish_tracks_c["win" + str(win)] = ((fish_tracks_i.groupby('FishID').movement.apply(lambda x: x.rolling(win_f).sum()) == 0) * 1)
        # else:
        #     fish_tracks_c["win" + str(win)] = ((fish_tracks_i.groupby('FishID').movement.apply(lambda x: x.rolling(win_f).sum()) == 0) * 1)


        # print(fish_tracks_i.rest[309:310])
        # print(((fish_tracks_i.groupby('FishID').movement.apply(lambda x: x.rolling(win_f).sum()) == 0) * 1)[309:310])
        # ax2.plot(fish_tracks_i['rest'])
        # fish_tracks_i = fish_tracks_i.drop(columns=['rest'])

        # fish_tracks_i['rest'] = copy.copy(test)
        # fig1 = plt.subplots()
        # plt.fill_between(np.arange(0, len(fish_tracks_i.rest * 200)), fish_tracks_i.rest * 45, alpha=0.5, color='green')
        # plt.plot(fish_tracks_i.speed_mm)
        # plt.plot(fish_tracks_i.movement * 40)
        # plt.plot(fish_tracks_i.rest * 45)

        fish_tracks_30m = copy.copy(fish_tracks_i.groupby('FishID').resample('30T', on='ts').mean())
        fish_tracks_30m = fish_tracks_30m.reset_index()

        fish_tracks_i = fish_tracks_i.drop(columns=['rest'])

        if first:
            fish_tracks_30m_c = copy.copy(fish_tracks_30m)
            first = False
        else:
            fish_tracks_30m_c["win" + str(win)] = fish_tracks_30m["rest"]
        ax.plot(fish_tracks_30m.ts, fish_tracks_30m.rest, label=win)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_major_formatter(date_form)
    fill_plot_ts(ax, change_times_d, fish_tracks_30m.ts)
    ax.set_ylim([0, 1])
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Fraction rest")
    ax.set_title("Rest calculated from different lengths")
    ax.legend()

    print("defining rest")
    return fish_tracks_i


def define_bs(fish_tracks_i, change_times_d, time_window_s=15, fraction_threshold=0.10):
    """ Defines behavioural state by thresholding on a window
    Sleep  as defined by movement below Y% in X seconds

    :param fish_tracks_i:
    :param rootdir:
    :param time_window_s:
    :param fraction_threshold:
    :return:
    """

    fish_tracks_i["behave"] = ((fish_tracks_i.groupby("FishID").movement.rolling(10 * time_window_s).mean()) >
                               fraction_threshold)*1
    # fish_tracks_i["behave"] = ((fish_tracks_i.movement.rolling(10 * time_window_s).mean()) > fraction_threshold) * 1

    # fig1 = plt.subplots()
    # plt.fill_between(np.arange(0, len(fish_tracks_i.behave * 200)), fish_tracks_i.behave * 45, alpha=0.5, color='green')
    # plt.plot(fish_tracks_i.speed_mm)
    # plt.plot(fish_tracks_i.movement * 40)

    fish_tracks_30m = fish_tracks_i.groupby('FishID').resample('30T', on='ts').mean()
    fish_tracks_30m = fish_tracks_30m.reset_index()

    fig1, ax = plt.subplots(1, 1, figsize=(10, 4))
    date_form = DateFormatter("%H")
    # plt.plot(fish_tracks_30m.speed_mm)
    plt.plot(fish_tracks_30m.ts, abs(fish_tracks_30m.behave-1))
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_major_formatter(date_form)
    fill_plot_ts(ax, change_times_d, fish_tracks_30m.ts)
    ax.set_ylim([0, 1])
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Fraction rest")
    ax.set_title("Rest calculated from thresh {} and window {}".format(fraction_threshold, time_window_s))

    print("defining behavioural state")
    return fish_tracks_i


def plt_move_bs(fish_tracks_c, metat):
    fishes = fish_tracks_c.FishID.unique()[:]
    for fish in fishes:
        fig1 = plt.subplot()
        plt.plot(np.arange(0, 100000, 1), fish_tracks_c.loc[fish_tracks_c.FishID == fish, 'speed_mm'][100000:200000])
        plt.fill_between(np.arange(0, len(fish_tracks_c.loc[fish_tracks_c.FishID == fish, 'movement'][100000:200000] *
                                          200)), fish_tracks_c.loc[fish_tracks_c.FishID == fish, 'movement']
        [100000:200000] * 45, alpha=0.5, color='green')
        plt.plot([0, 100000], [metat.loc[fish, 'fish_length_mm']*0.25, metat.loc[fish, 'fish_length_mm']*0.25], 'k')
        plt.fill_between(np.arange(0, len(fish_tracks_c.loc[fish_tracks_c.FishID == fish, 'cluster'][100000:200000]
                                          * 55)), fish_tracks_c.loc[fish_tracks_c.FishID == fish, 'cluster']
        [100000:200000] * 200, alpha=0.5, color='orange')
        plt.xlabel("frame #")
        plt.ylabel("Speed mm/s")


def plotting_clustering_states(fish_tracks_i, resample_units=['1S', '2S', '3S', '4S', '5S', '10S', '15S', '20S', '30S',
                                                              '45S', '1T', '2T', '5T', '10T', '15T', '20T']):
    """ For plotting the different resampling of fish clustering

    :param fish_tracks_i:
    :param resample_units:
    :return:
    """
    bin_boxes = np.arange(0, 150, 1)
    fishes = fish_tracks_i.FishID.unique()[:]
    all_counts_combined_norm = np.zeros([len(bin_boxes)-1, len(resample_units), len(fishes)])

    for fish_n, fish in enumerate(fishes):
        fish_tracks_s = fish_tracks_i.loc[fish_tracks_i.FishID == fish, ['speed_mm', 'ts']]
        counts_combined = np.zeros([len(resample_units), len(bin_boxes)-1])
        counts_combined_std = np.zeros([len(resample_units), len(bin_boxes) - 1])

        fig3, ax3 = plt.subplots()
        for resample_n, resample_unit in enumerate(resample_units):
            # resample data to get mean and std for speed
            fish_tracks_b = fish_tracks_s.resample(resample_unit, on='ts').mean().rename(columns={'speed_mm': 'spd_mean'})
            fish_tracks_std = fish_tracks_s.resample(resample_unit, on='ts').std().rename(columns={'speed_mm': 'spd_std'})

            fish_tracks_ds = pd.concat([fish_tracks_b, fish_tracks_std], axis=1)
            kl, _, _, kmeans_list = kmeans_cluster(fish_tracks_ds, resample_unit, cluster_number=10)

            fig1, ax1 = plt.subplots(2, 1)
            counts, _, _ = plt.hist(fish_tracks_b["spd_mean"], bins=bin_boxes)
            counts_combined[resample_n, :] = counts
            fish_tracks_b.plot.line(y="spd_mean", ax=ax1[0])
            plt.close()

            fig1, ax1 = plt.subplots(2, 1)
            counts_std, _, _ = plt.hist(fish_tracks_std["spd_std"], bins=bin_boxes)
            counts_combined_std[resample_n, :] = counts_std
            fish_tracks_std.plot.line(y="spd_std", ax=ax1[0])
            plt.close()

        fig2, ax2 = plt.subplots(len(resample_units), 1)
        for i, resample_unit in enumerate(resample_units):
            ax2[i].bar(bin_boxes[0:-1], counts_combined[i], color='red', width=1, label=resample_unit)
            ax2[i].get_xaxis().set_ticks([])
            ax2[i].legend()
        ax2[i].get_xaxis().set_ticks(np.arange(0, max(bin_boxes), step=20))
        fig2.suptitle("{}".format(fish), fontsize=8)

        # HEATMAP
        # normalise each row
        counts_combined_norm = pd.DataFrame(data=counts_combined.T, index=bin_boxes[0:-1], columns=resample_units)
        counts_combined_norm = counts_combined_norm.div(counts_combined_norm.sum(axis=0), axis=1)

        counts_combined_std_norm = pd.DataFrame(data=counts_combined_std.T, index=bin_boxes[0:-1], columns=resample_units)
        counts_combined_std_norm = counts_combined_std_norm.div(counts_combined_std_norm.sum(axis=0), axis=1)

        fig3 = plt.figure(figsize=(8, 4))
        plt.imshow(counts_combined_norm.T, cmap='hot', aspect='auto')
        plt.title("{}".format(fish))
        cbar = plt.colorbar(label="% occupancy")
        plt.gca().invert_yaxis()
        plt.gca().set_xticks(np.arange(0, bin_boxes[-1], 10))
        plt.gca().set_yticks(np.arange(0, len(resample_units)))
        plt.gca().set_yticklabels(resample_units)
        plt.gca().set_xlim([0, 120])
        plt.gca().set_xlabel("Speed mm/s")
        plt.gca().set_ylabel("Time bin")
        fig2.suptitle("{}".format(fish), fontsize=8)
        # plt.savefig(os.path.join(rootdir, "xy_ave_Day_{0}.png".format(species_n.replace(' ', '-'))))

        fig3 = plt.figure(figsize=(8, 4))
        plt.imshow(counts_combined_std_norm.T, cmap='hot', aspect='auto')
        plt.title("{}".format(fish))
        cbar = plt.colorbar(label="% occupancy")
        plt.gca().invert_yaxis()
        plt.gca().set_xticks(np.arange(0, bin_boxes[-1], 10))
        plt.gca().set_yticks(np.arange(0, len(resample_units)))
        plt.gca().set_yticklabels(resample_units)
        plt.gca().set_xlim([0, 120])
        plt.gca().set_xlabel("Speed mm/s std")
        plt.gca().set_ylabel("Time bin")
        fig2.suptitle("{}".format(fish), fontsize=8)

        # https://matplotlib.org/matplotblog/posts/create-ridgeplots-in-matplotlib/
        cmap = cm.get_cmap('turbo')
        colour_array = np.arange(0, 1, 1/len(resample_units))

        gs = grid_spec.GridSpec(len(resample_units), 1)
        fig = plt.figure(figsize=(16, 9))

        ax_objs = []
        for resample_n, resample_unit in enumerate(resample_units):
            x = counts_combined_norm.loc[:, resample_unit]

            # creating new axes object
            ax_objs.append(fig.add_subplot(gs[resample_n:resample_n + 1, 0:]))

            # plotting the distribution
            ax_objs[-1].plot(x, lw=1, color='k') #cmap(colour_array[resample_n]))
            ax_objs[-1].fill_between(bin_boxes[0:-1], x, 0, color=cmap(colour_array[resample_n]))

            # remove borders, axis ticks, and labels
            ax_objs[-1].set_yticks([])
            ax_objs[-1].set_yticklabels([])
            ax_objs[-1].set_ylabel('')

            # setting uniform x and y lims
            ax_objs[-1].set_xlim(0, max(bin_boxes))
            ax_objs[-1].set_ylim(0, 0.3)

            # make background transparent
            rect = ax_objs[-1].patch
            rect.set_alpha(0)

            if resample_n == len(resample_units) - 1:
                ax_objs[-1].set_xlabel("Speed mm/s", fontsize=16, fontweight="bold")
            else:
                ax_objs[-1].set_xticklabels([])

            spines = ["top", "right", "left", "bottom"]
            for s in spines:
                ax_objs[-1].spines[s].set_visible(False)
            ax_objs[-1].text(-0.02, 0, resample_unit, fontweight="bold", fontsize=14, ha="right")

            gs.update(hspace=-0.8)
            plt.show()

        all_counts_combined_norm[:, :, fish_n] = counts_combined_norm
        all_counts_combined_norm_mean = np.mean(all_counts_combined_norm, axis=2)

        fig3 = plt.figure(figsize=(8, 4))
        plt.imshow(all_counts_combined_norm_mean.T, cmap='hot', aspect='auto')
        plt.title("Average")
        cbar = plt.colorbar(label="% occupancy")
        plt.gca().invert_yaxis()
        plt.gca().set_xticks(np.arange(0, bin_boxes[-1], 10))
        plt.gca().set_yticks(np.arange(0, len(resample_units)))
        plt.gca().set_yticklabels(resample_units)
        plt.gca().set_xlim([0, 120])
        plt.gca().set_xlabel("Speed mm/s")
        plt.gca().set_ylabel("Time bin")


def set_bs_thresh(fish_tracks_i_fish, fish_tracks_unit, thresholds):
    fishes = fish_tracks_i_fish.FishID.unique()[:]
    # playing around:
    thresholds = [5, 20]
    fish_tracks_i_fish = fish_tracks_i_fish.loc[fish_tracks_i_fish.FishID == fishes[3], ['speed_mm', 'ts']]
    # resample data
    fish_tracks_b = fish_tracks_i_fish.resample('30S', on='ts').mean()
    fish_tracks_b.reset_index(inplace=True)

    first = 1
    for thresh in thresholds:
        if first == 1:
            fish_tracks_b["behav_state"] = (fish_tracks_b.speed_mm > thresh) * 1
            first = 0
        else:
            fish_tracks_b["behav_state"] = fish_tracks_b["behav_state"] + (fish_tracks_b.speed_mm > thresh) * 1

    # fig3 = plt.figure(figsize=(8, 4))
    # plt.fill_between(np.arange(0, len(fish_tracks_unit.behav_state)), fish_tracks_unit.behav_state * 45, alpha=0.5,
    #                  color='green')
    # plt.plot(np.arange(0, len(fish_tracks_unit.behav_state)), fish_tracks_unit.speed_mm)

    fig3 = plt.figure(figsize=(8, 4))
    plt.plot(fish_tracks_i_fish.ts, fish_tracks_i_fish.speed_mm, color='b')
    plt.fill_between(fish_tracks_b.ts, fish_tracks_b.behav_state * 45, alpha=0.5, color='green')
    plt.plot(fish_tracks_b.ts, fish_tracks_b.speed_mm, color='k')


def finding_thresholds(spd_hist):
    # filtering shifts everything to the right
    _, row_n = spd_hist.shape

    for row in np.arange(0, row_n):
        smoothed_row = smooth_speed(spd_hist[:, row], win_size=5)
        peaks, _ = signal.find_peaks(smoothed_row.flatten(), distance=5, height=0.01)
        troughs, _ = signal.find_peaks(-smoothed_row.flatten(), distance=5)
        fig3 = plt.figure(figsize=(8, 4))
        plt.plot(spd_hist[:, row])
        plt.plot(smoothed_row, 'r')
        plt.plot(peaks, smoothed_row[peaks], 'x', color='r')
        plt.plot(troughs, smoothed_row[troughs], 'x', color='k')


def bout_play(fish_tracks_i, metat, fish_tracks_30m):
    fishes = fish_tracks_i.FishID.unique()[:]
    for fish in fishes:
        fig1 = plt.subplot()
        plt.plot(np.arange(0, 100000, 1), fish_tracks_i.loc[fish_tracks_i.FishID == fish, 'speed_mm'][100000:200000])
        plt.fill_between(np.arange(0, len(fish_tracks_i.loc[fish_tracks_i.FishID == fish, 'movement'][100000:200000] *
                                          45)), fish_tracks_i.loc[fish_tracks_i.FishID == fish, 'movement']
        [100000:200000] * 45, alpha=0.5, color='green')
        plt.plot([0, 100000], [metat.loc[fish, 'fish_length_mm']*0.25, metat.loc[fish, 'fish_length_mm']*0.25], 'k')
        plt.fill_between(np.arange(0, len(fish_tracks_i.loc[fish_tracks_i.FishID == fish, 'behav_state'][100000:200000]
                                          * 55)), fish_tracks_i.loc[fish_tracks_i.FishID == fish, 'behav_state']
        [100000:200000] * 45, alpha=0.5, color='orange')
        plt.xlabel("frame #")
        plt.ylabel("Speed mm/s")


        # fish_speed = fish_tracks_i.loc[fish_tracks_i.FishID == fish, 'speed_mm']

        fish_speed_d = fish_tracks_i[(fish_tracks_i.FishID == fish) & (fish_tracks_i.daynight == 'd')].speed_mm
        fish_speed_n = fish_tracks_i[(fish_tracks_i.FishID == fish) & (fish_tracks_i.daynight == 'n')].speed_mm

        # active_bout_lengths, active_bout_end, active_bout_start, inactive_bout_lengths, inactive_bout_end, \
        # inactive_bout_start, active_speed, active_bout_max, active_indices, inactive_speed, inactive_bout_max, \
        # inactive_indices = find_bouts(fish_speed, metat.loc[fish, 'fish_length_mm'] * 0.25)

        # NEED TO DEAL WITH EDGE CASES! IGNORING FOR THE MOMENT SINCE THEY ARE SUCH A SMALL FRACTION
        active_bout_lengths_d, active_bout_end_d, active_bout_start_d, inactive_bout_lengths_d, inactive_bout_end_d, \
        inactive_bout_start_d, active_speed_d, active_bout_max_d, active_indices_d, inactive_speed_d, \
        inactive_bout_max_d, inactive_indices_d = find_bouts(fish_speed_d, metat.loc[fish, 'fish_length_mm'] * 0.25)

        active_bout_lengths_n, active_bout_end_n, active_bout_start_n, inactive_bout_lengths_n, inactive_bout_end_n, \
        inactive_bout_start_n, active_speed_n, active_bout_max_n, active_indices_n, inactive_speed_n, \
        inactive_bout_max_n, inactive_indices_n = find_bouts(fish_speed_n, metat.loc[fish, 'fish_length_mm'] * 0.25)


        # binning bout counts
        bin_boxes_active = np.arange(0, 20000, 10)
        bin_boxes_inactive = np.arange(0, 20000, 10)

        bin_boxes_active_log = np.logspace(np.log10(0.1), np.log10(50000), 50)
        bin_boxes_inactive_log = np.logspace(np.log10(0.1), np.log10(50000), 50)

        fig2, ax2 = plt.subplots(2, 2)
        counts_act_bout_len_d, _, _ = ax2[0, 0].hist(active_bout_lengths_d, bins=bin_boxes_active, color='red')
        # ax2[0, 0].set_yscale('log')
        # ax2[0, 0].set_ylim([0, 10 ** 4.5])
        counts_act_bout_len_n, _, _ = ax2[1, 0].hist(active_bout_lengths_n, bins=bin_boxes_active)
        counts_inact_bout_len_d, _, _ = ax2[0, 1].hist(inactive_bout_lengths_d, bins=bin_boxes_inactive, color='red')
        counts_inact_bout_len_n, _, _ = ax2[1, 1].hist(inactive_bout_lengths_n, bins=bin_boxes_inactive)

        ax2[0, 0].set_ylabel("Day")
        ax2[1, 0].set_ylabel("Night")
        ax2[1, 0].set_xlabel("Active bout lengths")
        ax2[1, 1].set_xlabel("Inactive bout lengths")

        fps=10
        counts_act_bout_len_d_norm = norm_hist(counts_act_bout_len_d)
        counts_act_bout_len_n_norm = norm_hist(counts_act_bout_len_n)
        counts_inact_bout_len_d_norm = norm_hist(counts_inact_bout_len_d)
        counts_inact_bout_len_n_norm = norm_hist(counts_inact_bout_len_n)

        fig2, ax2 = plt.subplots(1, 2)
        ax2[0].plot(bin_boxes_active[0:-1]/fps, counts_act_bout_len_d_norm, 'r', label='Day')
        ax2[0].plot(bin_boxes_active[0:-1]/fps, counts_act_bout_len_n_norm, color='blueviolet', label='Night')
        ax2[1].plot(bin_boxes_inactive[0:-1]/fps, counts_inact_bout_len_d_norm, color='indianred', label='Day')
        ax2[1].plot(bin_boxes_inactive[0:-1]/fps, counts_inact_bout_len_n_norm, color='darkblue', label='Night')
        ax2[0].set_ylabel("Fraction")
        ax2[0].set_xlabel("Mobile bouts lengths (s)")
        ax2[1].set_xlabel("Immobile bouts lengths (s)")
        ax2[0].set_xlim([-1, 50])
        ax2[1].set_xlim([-1, 50])
        ax2[0].set_ylim([0, 1])
        ax2[1].set_ylim([0, 1])
        # ax2[0].set_yscale('log')
        # ax2[1].set_yscale('log')
        ax2[1].legend()

        fig2, ax2 = plt.subplots(1, 1)
        ax2.plot(bin_boxes_active[0:-1]/fps, counts_act_bout_len_d_norm, 'r', label='Day', marker='o', markersize=3)
        ax2.plot(bin_boxes_active[0:-1]/fps, counts_act_bout_len_n_norm, color='blueviolet', label='Night', marker='o', markersize=3)
        ax2.set_ylabel("Fraction")
        ax2.set_xlabel("Mobile bouts lengths (s)")
        ax2.set_xlim([-1, 20])
        ax2.set_ylim([0, 1])
        ax2.legend()

        fig2, ax2 = plt.subplots(1, 1)
        ax2.plot(bin_boxes_inactive[0:-1]/fps, counts_inact_bout_len_d_norm, color='indianred', label='Day', marker='o', markersize=3)
        ax2.plot(bin_boxes_inactive[0:-1]/fps, counts_inact_bout_len_n_norm, color='darkblue', label='Night', marker='o', markersize=3)
        ax2.set_ylabel("Fraction")
        ax2.set_xlabel("Immobile bouts lengths (s)")
        ax2.set_xlim([-1, 20])
        ax2.set_ylim([0, 0.4])
        ax2.legend()


        # bout count distributions on a log scale
        bin_boxes_active_log = np.logspace(np.log10(0.1), np.log10(50000), 50)
        bin_boxes_inactive_log = np.logspace(np.log10(0.1), np.log10(50000), 50)

        fig2, ax2 = plt.subplots(2, 2)
        counts_act_bout_len_d_log, _, _ = ax2[0, 0].hist(active_bout_lengths_d, bins=bin_boxes_active_log, color='red')
        counts_act_bout_len_n_log, _, _ = ax2[1, 0].hist(active_bout_lengths_n, bins=bin_boxes_active_log)
        counts_inact_bout_len_d_log, _, _ = ax2[0, 1].hist(inactive_bout_lengths_d, bins=bin_boxes_inactive_log, color='red')
        counts_inact_bout_len_n_log, _, _ = ax2[1, 1].hist(inactive_bout_lengths_n, bins=bin_boxes_inactive_log)

        counts_act_bout_len_d_log_norm = norm_hist(counts_act_bout_len_d_log)
        counts_act_bout_len_n_log_norm = norm_hist(counts_act_bout_len_n_log)
        counts_inact_bout_len_d_log_norm = norm_hist(counts_inact_bout_len_d_log)
        counts_inact_bout_len_n_log_norm = norm_hist(counts_inact_bout_len_n_log)

        fig2, ax2 = plt.subplots(1, 1)
        ax2.plot(bin_boxes_active_log[0:-1]/fps, counts_act_bout_len_d_log_norm, 'r', label='Day', marker='o', markersize=3)
        ax2.plot(bin_boxes_active_log[0:-1]/fps, counts_act_bout_len_n_log_norm, color='blueviolet', label='Night', marker='o', markersize=3)
        ax2.set_ylabel("Fraction")
        ax2.set_xlabel("Mobile bouts lengths (s)")
        ax2.set_xlim([-1, 200])
        ax2.set_ylim([0, 0.15])
        ax2.legend()

        fig2, ax2 = plt.subplots(1, 1)
        ax2.plot(bin_boxes_inactive_log[0:-1]/fps, counts_inact_bout_len_d_log_norm, color='indianred', label='Day', marker='o', markersize=3)
        ax2.plot(bin_boxes_inactive_log[0:-1]/fps, counts_inact_bout_len_n_log_norm, color='darkblue', label='Night', marker='o', markersize=3)
        ax2.set_ylabel("Fraction")
        ax2.set_xlabel("Immobile bouts lengths (s)")
        ax2.set_xlim([-20, 300])
        ax2.set_ylim([0, 0.15])
        ax2.legend()


        fig2, ax2 = plt.subplots(1, 2)
        ax2[0].plot(bin_boxes_active[0:-1] / 10, np.cumsum(counts_act_bout_len_d_norm), color='red', label='Day')
        ax2[0].plot(bin_boxes_active[0:-1] / 10, np.cumsum(counts_act_bout_len_n_norm), color='blueviolet', label='Night')
        ax2[1].plot(bin_boxes_inactive[0:-1] / 10, np.cumsum(counts_inact_bout_len_d_norm), color='indianred', label='Day')
        ax2[1].plot(bin_boxes_inactive[0:-1] / 10, np.cumsum(counts_inact_bout_len_n_norm), color='darkblue', label='Night')
        ax2[0].set_xlabel("Bout length (s)")
        ax2[1].set_xlabel("Bout length (s)")
        ax2[0].set_ylabel("Fraction of immobile bouts")
        ax2[1].set_ylabel("Fraction of immobile bouts")
        ax2[0].set_ylim([0, 1])
        ax2[1].set_ylim([0, 1])
        ax2[0].set_xlim([-20, 400])
        ax2[1].set_xlim([-20, 400])
        ax2[0].legend()
        ax2[1].legend()
        fig2.suptitle("Cumulative movement bouts for {}".format(fish), fontsize=8)


        # Speed during bouts
        bin_boxes_active = np.arange(0, 300, 3)

        fig2, ax2 = plt.subplots(2, 1)
        counts_act_spd_d, _, _ = ax2[0].hist(active_speed_d, bins=bin_boxes_active, color='red')
        ax2[0].set_yscale('log')
        ax2[0].set_ylim([0, 10 ** 6])

        counts_act_spd_n, _, _ = ax2[1].hist(active_speed_n, bins=bin_boxes_active, color='blueviolet',)
        ax2[1].set_yscale('log')
        ax2[1].set_ylim([0, 10 ** 6])

        ax2[0].set_ylabel("Day")
        ax2[1].set_ylabel("Night")
        ax2[1].set_xlabel("Speed during active bouts")

        counts_act_spd_d_norm = counts_act_spd_d / sum(counts_act_spd_d)
        counts_act_spd_n_norm = counts_act_spd_n / sum(counts_act_spd_n)
        fig2, ax2 = plt.subplots(1, 1)
        ax2.plot(counts_act_spd_d_norm, color='red', label='Day')
        ax2.plot(counts_act_spd_n_norm, color='blueviolet', label='Night')
        ax2.set_ylabel("Fraction")
        ax2.set_xlabel("Max speed during mobile bouts mm/s")
        # ax2.set_yscale('log')
        ax2.legend()


        # # are bout speed and length correlated?
        # fig2, ax2 = plt.subplots(1, 1)
        # ax2.plot(active_bout_lengths_d, active_bout_max_d, linestyle='', marker='o', markersize=1, color='r', alpha=0.1, label='Day')
        #
        # bout_length_vs_spd, xedges_night, yedges_night, _ = plt.hist2d(active_bout_lengths_d, active_bout_max_d,
        #                                                               bins=[300, 300], cmap='inferno')
        # bout_length_vs_spd = bout_length_vs_spd / sum(sum(bout_length_vs_spd))
        #
        #
        # fig2, ax2 = plt.subplots(1, 1)
        # ax2.imshow(bout_length_vs_spd.T, vmin=0, vmax=0.05)
        # ax2.invert_yaxis()
        # ax2.get_xaxis().set_ticks([])
        # ax2.get_yaxis().set_ticks([])


        bins = np.arange(0, 20000, 10)
        # bins = np.append(bins, [72000])

        # bins = np.logspace(np.log10(0.1), np.log10(50000), 50)

        total_time_active_f_d = np.empty(len(bins)-1, dtype=object)
        total_bouts_active_d = np.empty(len(bins)-1, dtype=object)
        total_time_active_f_n = np.empty(len(bins)-1, dtype=object)
        total_bouts_active_n = np.empty(len(bins)-1, dtype=object)
        total_time_inactive_f_d = np.empty(len(bins)-1, dtype=object)
        total_bouts_inactive_d = np.empty(len(bins)-1, dtype=object)
        total_time_inactive_f_n = np.empty(len(bins)-1, dtype=object)
        total_bouts_inactive_n = np.empty(len(bins)-1, dtype=object)
        for bin_idx, bin in enumerate(bins[0:-1]):

            subset_active_d = active_bout_lengths_d[(bins[bin_idx] < active_bout_lengths_d) & (active_bout_lengths_d
                                                                                               < bins[bin_idx + 1])]
            subset_active_n = active_bout_lengths_n[(bins[bin_idx] < active_bout_lengths_n) & (active_bout_lengths_n
                                                                                               < bins[bin_idx + 1])]

            subset_inactive_d = inactive_bout_lengths_d[(bins[bin_idx] < inactive_bout_lengths_d) &
                                                        (inactive_bout_lengths_d < bins[bin_idx + 1])]
            subset_inactive_n = inactive_bout_lengths_n[(bins[bin_idx] < inactive_bout_lengths_n) &
                                                        (inactive_bout_lengths_n < bins[bin_idx + 1])]
            total_time_active_f_d[bin_idx] = sum(subset_active_d)
            total_bouts_active_d[bin_idx] = len(subset_active_d)

            total_time_active_f_n[bin_idx] = sum(subset_active_n)
            total_bouts_active_n[bin_idx] = len(subset_active_n)

            total_time_inactive_f_d[bin_idx] = sum(subset_inactive_d)
            total_bouts_inactive_d[bin_idx] = len(subset_inactive_d)

            total_time_inactive_f_n[bin_idx] = sum(subset_inactive_n)
            total_bouts_inactive_n[bin_idx] = len(subset_inactive_n)

        total_time_active_f_d_norm = norm_hist(total_time_active_f_d)
        total_time_active_f_n_norm = norm_hist(total_time_active_f_n)
        total_time_inactive_f_d_norm = norm_hist(total_time_inactive_f_d)
        total_time_inactive_f_n_norm = norm_hist(total_time_inactive_f_n)

        fig2, ax2 = plt.subplots(2, 1)
        ax2[0].bar(bins[0:-1] / 10, total_time_active_f_d_norm, color='red', alpha=0.5, label='Day', width=2)
        ax2[0].bar(bins[0:-1] / 10, total_time_active_f_n_norm, color='blueviolet', alpha=0.5, label='Night', width=2)
        ax2[1].bar(bins[0:-1] / 10, total_time_inactive_f_d_norm, color='indianred', alpha=0.5, label='Day', width=2)
        ax2[1].bar(bins[0:-1] / 10, total_time_inactive_f_n_norm, color='darkblue', alpha=0.5, label='Night', width=2)
        ax2[1].set_xlabel("Total time in each time bin (s)")
        ax2[0].set_ylabel("Fraction time mobile bouts > length(x)")
        ax2[1].set_ylabel("Fraction time immobile bouts > length(x)")
        ax2[0].set_ylim([0, 0.4])
        ax2[1].set_ylim([0, 0.15])
        ax2[0].set_xlim([0, 200])
        ax2[1].set_xlim([0, 200])
        ax2[0].set_xlim([-20, 500])
        ax2[1].set_xlim([-20, 500])
        ax2[0].legend()
        ax2[1].legend()
        # plt.title("Movement bouts for {}".format(fish))

        fig2, ax2 = plt.subplots(2, 1)
        ax2[0].plot(bins[0:-1] / 10, np.cumsum(total_time_active_f_d_norm), color='red', label='Day')
        ax2[0].plot(bins[0:-1] / 10, np.cumsum(total_time_active_f_n_norm), color='blueviolet', label='Night')
        ax2[1].plot(bins[0:-1] / 10, np.cumsum(total_time_inactive_f_d_norm), color='indianred', label='Day')
        ax2[1].plot(bins[0:-1] / 10, np.cumsum(total_time_inactive_f_n_norm), color='darkblue', label='Night')
        ax2[1].set_xlabel("Time of bout length (s)")
        ax2[0].set_ylabel("Mobile bouts")
        ax2[1].set_ylabel("Immobile bouts")
        ax2[0].set_ylim([0, 1])
        ax2[1].set_ylim([0, 1])
        ax2[0].set_xlim([-20, 4000])
        ax2[1].set_xlim([-20, 4000])
        ax2[0].legend()
        ax2[1].legend()
        fig2.suptitle("Cumulative movement bouts for {}".format(fish), fontsize=8)

        fig2, ax2 = plt.subplots(1, 2)
        ax2[0].plot(bins[0:-1] / 10, np.cumsum(total_time_active_f_d_norm), color='red', label='Day')
        ax2[0].plot(bins[0:-1] / 10, np.cumsum(total_time_active_f_n_norm), color='blueviolet', label='Night')
        ax2[1].plot(bins[0:-1] / 10, np.cumsum(total_time_inactive_f_d_norm), color='indianred', label='Day')
        ax2[1].plot(bins[0:-1] / 10, np.cumsum(total_time_inactive_f_n_norm), color='darkblue', label='Night')
        ax2[0].set_xlabel("Bout length (s)")
        ax2[1].set_xlabel("Bout length (s)")
        ax2[0].set_ylabel("Fraction time mobile bouts > length(x)")
        ax2[1].set_ylabel("Fraction time mobile bouts > length(x)")
        ax2[0].set_ylim([0, 1])
        ax2[1].set_ylim([0, 1])
        ax2[0].set_xlim([-2, 300])
        ax2[1].set_xlim([-20, 4000])
        ax2[0].legend()
        ax2[1].legend()
        fig2.suptitle("Cumulative movement bouts for {}".format(fish), fontsize=8)
