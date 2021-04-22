import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from kneed import KneeLocator
from matplotlib.dates import DateFormatter
from matplotlib.ticker import (MultipleLocator)
import seaborn as sns

from cichlidanalysis.analysis.processing import add_col, standardise_cols
from cichlidanalysis.plotting.speed_plots import fill_plot_ts

# functions for clustering speed mean + std data into different behavioural states


def kmeans_cluster(input_pd_df, resample_unit_i, cluster_number=15):
    """ Clusters behaviour by z-scored values of all input columns and uses kmeans cluster on up to cluster_number
    clusters. Uses inertia and KneeLocator to find the knee to determine how many clusters to use. Note that number
    of starting clusters affects the Kneelocator.

    :param input_pd_series: a pd.series of data points to cluster - note will drop any NaNs
    :param resample_unit_i: time unit of resampling
    :param cluster_number: number of clusters to try kmeans with
    :return: kl, kmeans_list[kl.elbow - smaller_cluster], kl.elbow + 1 - smaller_cluster, kmeans_list
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
    ax5.set_title(resample_unit_i)

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
            # resample data to get mean and std for speed
            fish_tracks_spd_mean = fish_tracks_s.resample(resample_unit,
                                                          on='ts').mean().rename(columns={'speed_mm': 'spd_mean'})
            fish_tracks_std = fish_tracks_s.resample(resample_unit,
                                                     on='ts').std().rename(columns={'speed_mm': 'spd_std'})

            fish_tracks_ds = pd.concat([fish_tracks_spd_mean, fish_tracks_std], axis=1)

            print(resample_unit)
            kl, kmeans, best_cluster, _ = kmeans_cluster(fish_tracks_ds, resample_unit, cluster_number=6)
            cluster_list[resample_n] = best_cluster

        ax2.legend()


def clustering_states(fish_tracks_i, meta, resample_unit='15S'):
    """ Resamples and then does k-means clustering on the speed_mm mean and std.

    :param fish_tracks_i: fish tracks
    :param meta: meta data
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
        labels = copy.copy(kmeans_clustering.labels_).astype('float64')

        # insert NaNs (adds before so pushes things back as it should)
        for insertion_n, insertion_index in enumerate(nan_position):
            labels = np.insert(labels, insertion_index, np.NaN)

        # add cluster to fish_data (renames clusters so that cluster 0 has the lowest speed)
        spd_centre = kmeans_clustering.cluster_centers_[:, 0]
        ordering = np.argsort(spd_centre)
        print("adding in clusters (reordering so cluster 0 has lowest speed)")
        for cluster in np.arange(0, kmeans_clustering.n_clusters):
            fish_tracks_rs.loc[labels == cluster, 'cluster'] = ordering[cluster]

        if first:
            fish_tracks_15s = copy.copy(fish_tracks_rs)
            first = False
        else:
            fish_tracks_15s = pd.concat([fish_tracks_15s, copy.copy(fish_tracks_rs)], ignore_index=True)

    # add back species column
    for col_name in ['species']:
        add_col(fish_tracks_15s, col_name, fishes, meta)

    return fish_tracks_15s


def add_clustering(fish_tracks_rs, fish_tracks):
    """ Adds clustering data from fish_tracks_resampled to fish_tracks by forward filling

    :param fish_tracks_rs: fish_tracks_resampled with cluster column, FishID and ts
    :param fish_tracks: fish_tracks
    :return: merged_fish_tracks: fish_tracks with ffill cluster column and all of it's previous columns
    """
    # make subset of the clustered data
    fish_tracks_i = copy.copy(fish_tracks)
    cols = fish_tracks_rs.columns
    fish_tracks_rs_clust = fish_tracks_rs.drop(columns=cols.drop(['ts', 'FishID', 'cluster']))

    # merge clustered data with fishtracks using "outer" to include everything
    merged_fish_tracks = pd.merge(fish_tracks_i, fish_tracks_rs_clust, how='outer', on=["ts", "FishID"], indicator=
    True)
    # sort the data by first by fish and then by ts
    merged_fish_tracks = merged_fish_tracks.sort_values(["FishID", "ts"])
    # forward fill the cluster data
    merged_fish_tracks["cluster"] = merged_fish_tracks.groupby("FishID")["cluster"].fillna(method="ffill")

    # drop "right only" == resampled data, on indices
    merged_fish_tracks = merged_fish_tracks.drop(merged_fish_tracks.index[merged_fish_tracks._merge == "right_only"],
                                                 axis=0)
    # drop _merge column
    merged_fish_tracks = merged_fish_tracks.drop(columns="_merge")

    return merged_fish_tracks


def add_clustering_to_30m(fish_tracks_in, fish_tracks_30m):
    """ Adds clustering data from fish_tracks_in to fish_tracks_30m

    :param fish_tracks_in:
    :param fish_tracks_30m:
    :return:
    """

    # get each fish ID
    fish_IDs = fish_tracks_in['FishID'].unique()

    # for each fish, group by cluster and resample, this allows us to get the counts for each cluster type
    first = True
    for fish in fish_IDs:
        df = fish_tracks_in.loc[fish_tracks_in.FishID == fish, :].set_index('ts').groupby('cluster').resample('30T').size()

        # weirdly sometimes get pd.series and sometimes pd.df this deals with it
        if isinstance(df, pd.Series):
              df = df.unstack(0, fill_value=0)
        elif isinstance(df, pd.DataFrame):
            df = df.T

        df['FishID'] = fish

        # add the fishes back together
        if first:
            fish_tracks_30m_c = copy.copy(df)
            first = False
        else:
            fish_tracks_30m_c = pd.concat([fish_tracks_30m_c, copy.copy(df)], ignore_index=False)

    fish_tracks_30m_c = fish_tracks_30m_c.reset_index().rename(columns={'index': 'ts'})

    # for col_name in ['species']:
    #     add_col(fish_tracks_30m_c, col_name, fish_IDs, meta)

    cluster_names = ['zero', 'one', 'two', 'three', 'four']
    clusters = fish_tracks_30m_c.columns.drop(['ts', 'FishID'])
    for i in clusters:
        fish_tracks_30m_c = fish_tracks_30m_c.rename(columns={i: cluster_names[int(i)]})
    clusters = fish_tracks_30m_c.columns.drop(['ts', 'FishID'])

    # find total time points for each resampled bin
    fish_tracks_30m_c["total"] = 0
    next_cluster = 0
    while next_cluster < clusters.shape[0]:
        fish_tracks_30m_c["total"] = fish_tracks_30m_c["total"] + fish_tracks_30m_c.fillna(0)[clusters[next_cluster]]
        next_cluster = next_cluster + 1

    # make cluster counts normalised to total number of timepoints
    for clust in clusters:
        fish_tracks_30m_c[clust] = fish_tracks_30m_c[clust]/fish_tracks_30m_c["total"]

    print(fish_tracks_30m.shape)
    print(fish_tracks_30m_c.shape)

    merged_fish_tracks_30m = pd.merge(fish_tracks_30m, fish_tracks_30m_c, how='inner', on=["ts", "FishID"], indicator=True)

    # checks
    if not (merged_fish_tracks_30m._merge.unique() == 'both')[0]:
        print("somehow merging did not work as have exclusive rows!")
        return False
    else:
        merged_fish_tracks_30m = merged_fish_tracks_30m.drop(columns="_merge")
        return merged_fish_tracks_30m


def plt_clusters(merged_fish_tracks_30m, change_times_d, rootdir, meta):
    """ Takes fish_tracks with clusters, resamples to 30min to plot

    :param merged_fish_tracks_30m: clustered data
    :param change_times_d:
    :param rootdir: path for saving
    :param meta: file with meta data
    :return:
    """

    # get each species
    all_species = merged_fish_tracks_30m['species'].unique()
    fishes = merged_fish_tracks_30m.FishID.unique()[:]

    date_form = DateFormatter("%H")
    for species_f in all_species:
        plt.figure(figsize=(10, 4))
        ax = sns.lineplot(data=merged_fish_tracks_30m[merged_fish_tracks_30m.species == species_f], x='ts', y='one',
                          hue='FishID')
        ax.get_legend().remove()
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_major_formatter(date_form)
        fill_plot_ts(ax, change_times_d, merged_fish_tracks_30m[merged_fish_tracks_30m.FishID == fishes[0]].ts)
        ax.set_ylim([0, 1])
        plt.xlabel("Time (h)")
        plt.ylabel("Fraction cluster one")
        plt.title(species_f)
        # plt.savefig(os.path.join(rootdir, "speed_30min_individual{0}.png".format(species_f.replace(' ', '-'))))
