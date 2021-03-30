import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec
import matplotlib.cm as cm
import scipy.signal as signal
from sklearn.cluster import KMeans
from kneed import KneeLocator

from cichlidanalysis.analysis.bouts import find_bouts
from cichlidanalysis.analysis.processing import smooth_speed


def norm_hist(input_d):
    """ Normalise input by total number e.g. fraction"""
    input_d_norm = input_d / sum(input_d)
    return input_d_norm


def standardise_cols(input_pd_df):
    """ Calculate z-scores for every column"""

    first = 1
    cols = input_pd_df.columns
    for col in cols:
        col_zscore = col + '_zscore'
        if first:
            output_pd_df = ((input_pd_df[col] - input_pd_df[col].mean()) / input_pd_df[col].std()).to_frame().\
                rename(columns={'spd_mean': col_zscore})
            first = 0
        else:
            output_pd_df[col_zscore] = (input_pd_df[col] - input_pd_df[col].mean()) / input_pd_df[col].std()

    return output_pd_df


def define_bs(fish_tracks_i, rootdir, time_window_s, fraction_threshold=0.10):
    """ Defines behavioural state by thresholding on a window

    :param fish_tracks_i:
    :param rootdir:
    :param time_window_s:
    :param fraction_threshold:
    :return:
    """
    time_window_s = 60
    fraction_threshold

    fish_tracks_i["behav_state"] = ((fish_tracks_i.movement.rolling(
        10 * time_window_s).mean()) > fraction_threshold) * 1

    fig1 = plt.subplot()
    plt.fill_between(np.arange(0, len(fish_tracks_i.behav_state * 45)), fish_tracks_i.behav_state * 45, alpha=0.5,
                     color='green')
    plt.plot(fish_tracks_i.speed_mm)
    plt.plot(fish_tracks_i.movement * 40)

    plt.plot(fish_tracks_i.behav_state * 45)

    fish_tracks_30m = fish_tracks_i.groupby('FishID').resample('30T', on='ts').mean()
    fig2 = plt.subplot()
    plt.plot(fish_tracks_30m.speed_mm.to_numpy())
    plt.plot(fish_tracks_30m.movement.to_numpy() * 40)
    plt.plot(fish_tracks_30m.behav_state.to_numpy() * 45)

    print("defining bs")


def kmeans_cluster(input_pd_df, resample_unit_i, cluster_number=6):
    """ Clusters behaviour by z-scored values of all input columns and uses kmeans cluster on up to cluster_number
    clussters. Uses inertia and KneeLocator to find the knee to deeterminee how many clusters to use.

    :param input_pd_series: a pd.series of data points to cluster - note will drop any NaNs
    :param resample_unit_i: time unit of resampling
    :param cluster_number: number of clusters to try kmeans with
    :return:
    """
    input_df_zscore = standardise_cols(input_pd_df)
    kmeans_list = []
    clusters = np.arange(1, cluster_number+1)
    cluster_interia = np.zeros(max(clusters))
    for iteration in clusters:
        kmeans_list.append(KMeans(n_clusters=iteration, random_state=0, algorithm='full').fit(input_df_zscore.dropna().to_numpy().reshape
                                                                            (-1, input_df_zscore.shape[1])))
        cluster_interia[iteration - 1] = kmeans_list[iteration - 1].inertia_

    kl = KneeLocator(clusters, max(cluster_interia) - cluster_interia, curve="concave", direction="increasing")
    # kl.plot_knee()
    print(kl.elbow)

    # if one of the clusters is less than 1% of the total state space, assume it is spurious data and use one
    # less cluster
    occurrences, spurious = check_cluster_size(kmeans_list[kl.elbow - 1])
    smaller_cluster = 1
    while spurious:
        print("Found spurious  cluster, reducing cluster number by 1")
        smaller_cluster = smaller_cluster + 1
        occurrences, spurious = check_cluster_size(kmeans_list[kl.elbow - smaller_cluster])

    print((kl.elbow + 1) - smaller_cluster)

    colors = [cm.Blues, cm.Reds, cm.Greens, cm.Oranges, cm.Purples]
    fig3, ax3 = plt.subplots(1, 2)
    for ii in [0, 1]:
        if ii < 2:
            bin_n = np.arange(0, 150, 1)
        else:
            bin_n = np.arange(0, 1.1, 0.1)

        for i in np.arange(0, (kl.elbow + 1) - smaller_cluster):
            ax3[ii].hist(input_pd_df.dropna().iloc[kmeans_list[kl.elbow - smaller_cluster].labels_ == i, ii], bins=bin_n,
                     alpha=0.5, color=colors[i](0.5), label=i)
        ax3[ii].set_xlabel(input_pd_df.columns[ii])
        ax3[ii].legend()
        ax3[ii].set_title(resample_unit_i)

    fig4, ax4 = plt.subplots()
    hist_bins = np.arange(0, 150, 1)
    D_mean_std = np.zeros((hist_bins.shape[0]-1, hist_bins.shape[0]-1, (kl.elbow + 1) - smaller_cluster))
    for i in np.arange(0, (kl.elbow + 1) - smaller_cluster):
        output, _, _, _ = plt.hist2d(input_pd_df.dropna().iloc[kmeans_list[kl.elbow -
                                                smaller_cluster].labels_ == i, 0],input_pd_df.dropna().iloc[kmeans_list
                                                [kl.elbow - smaller_cluster].labels_ == i, 1],
                                                bins=[hist_bins, hist_bins], cmap=colors[i])
        D_mean_std[:, :, i] = output
    plt.close(fig4)

    D_mean_std_nan = copy.copy(D_mean_std)
    D_mean_std_nan[D_mean_std == 0] = np.nan

    fig5, ax5 = plt.subplots()
    for i in np.arange(0, (kl.elbow + 1) - smaller_cluster):
        my_cmap = copy.copy(colors[i])
        ax5.pcolormesh(D_mean_std_nan[:, :, i], cmap=my_cmap, vmin=0.1, vmax=np.percentile(D_mean_std[D_mean_std > 0], 90),
                       alpha=1, edgecolors='none')
    ax5.set_xlabel(input_pd_df.columns[0])
    ax5.set_ylabel(input_pd_df.columns[1])

    return kl, kmeans_list[kl.elbow - smaller_cluster], kl.elbow + 1 - smaller_cluster, kmeans_list


def check_cluster_size(kmeans_knee_i):
    """
    Check if there is a spurious state  by finding fractions of each state
    :param kmeans_knee_i:
    :return:
    """
    occurrences = np.zeros(kmeans_knee_i.n_clusters)
    for cluster in np.arange(0, kmeans_knee_i.n_clusters):
        occurrences[cluster] = np.count_nonzero(kmeans_knee_i.labels_ == cluster)

    # find fractions
    occurrences = occurrences/sum(occurrences)

    return occurrences, max(occurrences < 0.01)


def testing_clustering_states(fish_tracks_i, resample_units=['1S', '2S', '3S', '4S', '5S', '10S', '15S', '20S', '30S',
                                                             '45S', '1T', '2T', '5T', '10T', '15T', '20T']):
    """Resamples and then does k-means clustering on the speed_mm mean and std. loops for each resampling unit and  fish

    :param fish_tracks_i: fish tracks
    :param resample_units: resmapling units to test
    :return:
    """
    fishes = fish_tracks_i.FishID.unique()[:]

    for fish_n, fish in enumerate(fishes):
        fish_tracks_s = fish_tracks_i.loc[fish_tracks_i.FishID == fish, ['speed_mm', 'ts']]
        cluster_list = np.zeros([len(resample_units)])

        fig2, ax2 = plt.subplots()
        for resample_n, resample_unit in enumerate(resample_units):
            # resample data to get mean and std for speed and vertical position
            fish_tracks_spd_mean = fish_tracks_s.resample(resample_unit,
                                                          on='ts').mean().rename(columns={'speed_mm': 'spd_mean'})
            fish_tracks_std = fish_tracks_s.resample(resample_unit,
                                                     on='ts').std().rename(columns={'speed_mm': 'spd_std'})

            fish_tracks_ds = pd.concat([fish_tracks_spd_mean, fish_tracks_std], axis=1)

            print(resample_unit)
            kl, kmeans, best_cluster, _ = kmeans_cluster(fish_tracks_ds, resample_unit, cluster_number=6)
            cluster_list[resample_n] = best_cluster

        ax2.legend()


def clustering_states(fish_tracks_i, resample_unit=['15S']):
    """ Resamples and then does k-means clustering on the speed_mm mean and std.

    :param fish_tracks_i: fish tracks
    :param resample_unit: resampling unit
    :return: fish_tracks_15s: fish_tracks_i resampled with clustering
    """

    fishes = fish_tracks_i.FishID.unique()[:]
    first = True

    for fish_n, fish in enumerate(fishes):
        # get resampled data of fish
        fish_tracks_rs = fish_tracks_i.loc[fish_tracks_i.FishID == fish, :].resample(resample_unit, on='ts').mean()
        fish_tracks_rs["FishID"] = fish

        # get spd mean and std for cluster calculation
        fish_tracks_spd_mean = fish_tracks_i.loc[fish_tracks_i.FishID == fish, ['speed_mm', 'ts']].\
            resample(resample_unit, on='ts').mean().rename(columns={'speed_mm': 'spd_mean'})
        fish_tracks_std = fish_tracks_i.loc[fish_tracks_i.FishID == fish, ['speed_mm', 'ts']].resample(resample_unit,
                                                            on='ts').std().rename(columns={'speed_mm': 'spd_std'})

        # concatenate
        fish_tracks_ds = pd.concat([fish_tracks_spd_mean, fish_tracks_std], axis=1)

        # cluster data
        _, kmeans_clustering, _, _ = kmeans_cluster(fish_tracks_ds, resample_unit, cluster_number=6)

        fish_tracks_rs['cluster'] = np.NaN

        # need to put NaNs back in
        # get position of NaNs in original data
        fish_tracks_rs.reset_index(level=0, inplace=True)
        nan_position = fish_tracks_rs.index[np.isnan(fish_tracks_rs.speed_mm)].tolist()
        # fish_tracks_rs.set_index('ts')

        labels = copy.copy(kmeans_clustering.labels_).astype('float64')

        # insert NaNs (adds before so pushes things back as it should)
        for insertion_n, insertion_index in enumerate(nan_position):
            labels = np.insert(labels, insertion_index, np.NaN)

        # add cluster to fish_data (renames clusters so that cluster 0 has the lowest speed)
        spd_centre = kmeans_clustering.cluster_centers_[:, 0]
        ordering = np.argsort(spd_centre)
        for cluster in np.arange(0, kmeans_clustering.n_clusters):
            print("adding in clusters (reordering so cluster 0 has lowest speed")
            fish_tracks_rs.loc[labels == cluster, 'cluster'] = ordering[cluster]

        if first:
            fish_tracks_15s = copy.copy(fish_tracks_rs)
            first = False
        else:
            fish_tracks_15s = pd.concat([fish_tracks_15s, copy.copy(fish_tracks_rs)], ignore_index=True)

    fish_tracks_15s.reset_index(level=0, inplace=True)
    return fish_tracks_15s


def bin_seperate(fish_tracks_i, resample_units=['1S', '2S', '3S', '4S', '5S', '10S', '15S', '20S', '30S', '45S', '1T',
                                                '2T', '5T', '10T', '15T', '20T']):
    """

    :param fish_tracks_i:
    :param resample_units:
    :return:
    """
    bin_boxes = np.arange(0, 150, 1)
    fishes = fish_tracks_i.FishID.unique()[:]
    all_counts_combined_norm = np.zeros([len(bin_boxes)-1, len(resample_units), len(fishes)])

    for fish_n, fish in enumerate(fishes):
        fish_tracks_s = fish_tracks_i.loc[fish_tracks_i.FishID == fish, ['speed_mm', 'ts']]
        fish_tracks_v = fish_tracks_i.loc[fish_tracks_i.FishID == fish, ['vertical_position', 'ts']]
        counts_combined = np.zeros([len(resample_units), len(bin_boxes)-1])
        counts_combined_std = np.zeros([len(resample_units), len(bin_boxes) - 1])

        fig3, ax3 = plt.subplots()
        for resample_n, resample_unit in enumerate(resample_units):
            # resample data
            fish_tracks_b = fish_tracks_s.resample(resample_unit, on='ts').mean().rename(columns={'speed_mm': 'spd_mean'})
            # fish_tracks_b.reset_index(inplace=True)

            fish_tracks_std = fish_tracks_s.resample(resample_unit, on='ts').std().rename(columns={'speed_mm': 'spd_std'})
            # fish_tracks_std.reset_index(inplace=True)

            fish_tracks_v_mean = fish_tracks_i.loc[fish_tracks_i.FishID == fish, ['vertical_pos', 'ts']].resample(
                resample_unit, on='ts').mean().rename(columns={'vertical_pos': 'vp_mean'})
            fish_tracks_v_std = fish_tracks_i.loc[fish_tracks_i.FishID == fish, ['vertical_pos', 'ts']].resample(
                resample_unit, on='ts').std().rename(columns={'vertical_pos': 'vp_std'})

            fish_tracks_ds = pd.concat([fish_tracks_b, fish_tracks_std, fish_tracks_v_mean, fish_tracks_v_std], axis=1)

            print(resample_unit)
            kl, kmeans_list[kl.elbow], kmeans_list = kmeans_cluster(fish_tracks_ds, cluster_number=15)

            ax3.scatter(fish_tracks_b["speed_mm"], fish_tracks_std["speed_mm"], alpha=0.5)

            # for fish in fishes:
            fig1, ax1 = plt.subplots(2, 1)
            # fish_tracks_b.loc[fish_tracks_b.FishID == fish].plot.hist(y="speed_mm", bins=100, ax=ax1[0])
            counts, _, _ = plt.hist(fish_tracks_b["speed_mm"], bins=bin_boxes)
            counts_combined[resample_n, :] = counts
            fish_tracks_b.plot.line(y="speed_mm", ax=ax1[0])
            plt.close()

            fig1, ax1 = plt.subplots(2, 1)
            # fish_tracks_b.loc[fish_tracks_b.FishID == fish].plot.hist(y="speed_mm", bins=100, ax=ax1[0])
            counts_std, _, _ = plt.hist(fish_tracks_std["speed_mm"], bins=bin_boxes)
            counts_combined_std[resample_n, :] = counts_std
            fish_tracks_std.plot.line(y="speed_mm", ax=ax1[0])
            plt.close()

        fig2, ax2 = plt.subplots(len(resample_units), 1)
        for i, resample_unit in enumerate(resample_units):
            ax2[i].bar(bin_boxes[0:-1], counts_combined[i], color='red', width=1, label=resample_unit)
            ax2[i].get_xaxis().set_ticks([])
            ax2[i].legend()
        ax2[i].get_xaxis().set_ticks(np.arange(0, max(bin_boxes), step=20))
        fig2.suptitle("{}".format(fish), fontsize=8)

        # heatmap
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

    ## playing around:
    thresholds = [5, 20]
    fish_tracks_i_fish = fish_tracks_i.loc[fish_tracks_i.FishID == fishes[3], ['speed_mm', 'ts']]
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


def bout_play(fish_tracks_i, metat):
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

# l, kmeans_list[kl.elbow], kmeans_list = kmeans_cluster(fish_tracks_ds, cluster_number=15)



def bin_seperate(fish_tracks_i, resample_units=['1S', '2S', '3S', '4S', '5S', '10S', '15S', '20S', '30S', '45S', '1T',
                                                '2T', '5T', '10T', '15T', '20T']):
    # resample_units = ['1S', '2S', '3S', '4S', '5S', '10S', '15S', '20S', '30S', '45S', '1T', '2T', '5T', '10T', '15T', '20T']
    bin_boxes = np.arange(0, 150, 1)
    fishes = fish_tracks_i.FishID.unique()[:]
    all_counts_combined_norm = np.zeros([len(bin_boxes)-1, len(resample_units), len(fishes)])

    for fish_n, fish in enumerate(fishes):
        fish_tracks_s = fish_tracks_i.loc[fish_tracks_i.FishID == fish, ['speed_mm', 'ts']]
        fish_tracks_v = fish_tracks_i.loc[fish_tracks_i.FishID == fish, ['vertical_position', 'ts']]
        counts_combined = np.zeros([len(resample_units), len(bin_boxes)-1])
        counts_combined_std = np.zeros([len(resample_units), len(bin_boxes) - 1])

        fig3, ax3 = plt.subplots()
        for resample_n, resample_unit in enumerate(resample_units):
            # resample data
            fish_tracks_b = fish_tracks_s.resample(resample_unit, on='ts').mean().rename(columns={'speed_mm': 'spd_mean'})
            # fish_tracks_b.reset_index(inplace=True)

            fish_tracks_std = fish_tracks_s.resample(resample_unit, on='ts').std().rename(columns={'speed_mm': 'spd_std'})
            # fish_tracks_std.reset_index(inplace=True)

            fish_tracks_v_mean = fish_tracks_i.loc[fish_tracks_i.FishID == fish, ['vertical_pos', 'ts']].resample(
                resample_unit, on='ts').mean().rename(columns={'vertical_pos': 'vp_mean'})
            fish_tracks_v_std = fish_tracks_i.loc[fish_tracks_i.FishID == fish, ['vertical_pos', 'ts']].resample(
                resample_unit, on='ts').std().rename(columns={'vertical_pos': 'vp_std'})

            fish_tracks_ds = pd.concat([fish_tracks_b, fish_tracks_std, fish_tracks_v_mean, fish_tracks_v_std], axis=1)

            print(resample_unit)
            kl, kmeans_list[kl.elbow], kmeans_list = kmeans_cluster(fish_tracks_ds, cluster_number=15)

            ax3.scatter(fish_tracks_b["speed_mm"], fish_tracks_std["speed_mm"], alpha=0.5)

            # for fish in fishes:
            fig1, ax1 = plt.subplots(2, 1)
            # fish_tracks_b.loc[fish_tracks_b.FishID == fish].plot.hist(y="speed_mm", bins=100, ax=ax1[0])
            counts, _, _ = plt.hist(fish_tracks_b["speed_mm"], bins=bin_boxes)
            counts_combined[resample_n, :] = counts
            fish_tracks_b.plot.line(y="speed_mm", ax=ax1[0])
            plt.close()

            fig1, ax1 = plt.subplots(2, 1)
            # fish_tracks_b.loc[fish_tracks_b.FishID == fish].plot.hist(y="speed_mm", bins=100, ax=ax1[0])
            counts_std, _, _ = plt.hist(fish_tracks_std["speed_mm"], bins=bin_boxes)
            counts_combined_std[resample_n, :] = counts_std
            fish_tracks_std.plot.line(y="speed_mm", ax=ax1[0])
            plt.close()

        fig2, ax2 = plt.subplots(len(resample_units), 1)
        for i, resample_unit in enumerate(resample_units):
            ax2[i].bar(bin_boxes[0:-1], counts_combined[i], color='red', width=1, label=resample_unit)
            ax2[i].get_xaxis().set_ticks([])
            ax2[i].legend()
        ax2[i].get_xaxis().set_ticks(np.arange(0, max(bin_boxes), step=20))
        fig2.suptitle("{}".format(fish), fontsize=8)

        # heatmap
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

    ## playing around:
    thresholds = [5, 20]
    fish_tracks_i_fish = fish_tracks_i.loc[fish_tracks_i.FishID == fishes[3], ['speed_mm', 'ts']]
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


def bout_play(fish_tracks_i, metat):
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

