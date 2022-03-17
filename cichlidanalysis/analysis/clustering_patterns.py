import os

import datetime as dt
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
import pandas as pd
from scipy.spatial.distance import pdist
from matplotlib.dates import DateFormatter
from matplotlib.ticker import (MultipleLocator)

from cichlidanalysis.plotting.single_plots import fill_plot_ts
from cichlidanalysis.plotting.daily_plots import daily_ave_spd, daily_ave_move, daily_ave_rest
from cichlidanalysis.utils.timings import output_timings

# Inspiration from:
# https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/


def plot_bin_cluster(rootdir, subset, cluster_n, change_times_d, feature):

    date_form = DateFormatter("%H")
    plt.figure(figsize=(10, 4))
    plt.plot(subset.index, subset)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_major_formatter(date_form)
    fill_plot_ts(ax, change_times_d, subset.reset_index().rename(columns={'index': 'ts'}).ts)
    if feature == 'Speed':
        ax.set_ylim([0, 90])
    plt.xlabel("Time (h)")
    plt.ylabel(feature)
    plt.title("cluster: {}".format(cluster_n))
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "{0}_bin_individual_cluster-{1}.png".format(feature, cluster_n)))
    plt.close()

    date_form = DateFormatter("%H")
    plt.figure(figsize=(10, 4))
    plt.plot(subset.index, subset.mean(axis=1)+subset.std(axis=1), color='silver')
    plt.plot(subset.index, subset.mean(axis=1)-subset.std(axis=1), color='silver')
    plt.plot(subset.index, subset.mean(axis=1))
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_major_formatter(date_form)
    fill_plot_ts(ax, change_times_d, subset.reset_index().rename(columns={'index': 'ts'}).ts)
    if feature == 'Speed':
        ax.set_ylim([0, 60])
    plt.xlabel("Time (h)")
    plt.ylabel(feature)
    plt.title("cluster: {}".format(cluster_n))
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "{0}_30min_mean-std_cluser-{1}.png".format(feature, cluster_n)))
    plt.close()
    return


def cluster_patterns(data_feature, rootdir, feature, max_d=1.3, label='feature'):
    individ_corr = data_feature.corr(method='pearson')
    z = linkage(individ_corr, 'single')

    plt.figure(figsize=[8, 5])
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8,  # font size for the x axis labels
        labels=individ_corr.index,
        color_threshold=max_d
    )
    plt.axhline(max_d, color='k')
    plt.savefig(os.path.join(rootdir, "species_corr_dendrogram_max_d-{0}_{1}_{2}_{3}.png".format(label, max_d,
                                                                                             feature, dt.date.today())))
    plt.close()

    c, coph_dists = cophenet(z, pdist(individ_corr))
    print("The cophenetic correlation coefficient is {}, The closer the value is to 1, the better the clustering "
          "preserves the original distances".format(round(c, 2)))

    clusters = fcluster(z, max_d, criterion='distance')
    d = {'species_six': individ_corr.index, "cluster": clusters}
    species_cluster = pd.DataFrame(d)
    species_cluster = species_cluster.sort_values(by="cluster")

    return species_cluster


def run_species_pattern_cluster_daily(aves_ave_spd, aves_ave_move, aves_ave_rest, rootdir):

    change_times_unit = [7*2, 7.5*2, 18.5*2, 19*2]

    species_cluster_spd = cluster_patterns(aves_ave_spd, rootdir, feature="speed", max_d=1.3, label='daily')
    species_cluster_move = cluster_patterns(aves_ave_move, rootdir, feature="movement", max_d=1.2,  label='daily')
    species_cluster_rest = cluster_patterns(aves_ave_rest, rootdir, feature="rest", max_d=1.05,  label='daily')

    for i in species_cluster_spd.cluster.unique():
        species_subset = species_cluster_spd.loc[species_cluster_spd.cluster == i, 'species_six'].tolist()
        subset = aves_ave_spd[species_subset]
        daily_ave_spd(subset, subset.std(axis=1), rootdir, 'cluster_{}'.format(i), change_times_unit)

    for i in species_cluster_move.cluster.unique():
        species_subset = species_cluster_move.loc[species_cluster_move.cluster == i, 'species_six'].tolist()
        subset = aves_ave_move[species_subset]
        daily_ave_move(subset, subset.std(axis=1), rootdir, 'cluster_{}'.format(i), change_times_unit)

    for i in species_cluster_rest.cluster.unique():
        species_subset = species_cluster_rest.loc[species_cluster_rest.cluster == i, 'species_six'].tolist()
        subset = aves_ave_rest[species_subset]
        daily_ave_rest(subset, subset.std(axis=1), rootdir, 'cluster_{}'.format(i), change_times_unit)

    return species_cluster_spd, species_cluster_move, species_cluster_rest


def run_species_pattern_cluster_weekly(averages_spd, averages_move, averages_rest, rootdir):

    # set sunrise, day, sunset, night times (ns, s, m, h) and set day length in ns
    change_times_s, change_times_ns, change_times_m, change_times_h, day_ns, day_s, change_times_d = output_timings()

    species_cluster_spd = cluster_patterns(averages_spd, rootdir, feature="speed", max_d=1.25, label='weekly')
    species_cluster_move = cluster_patterns(averages_move, rootdir, feature="movement", max_d=1.2, label='weekly')
    species_cluster_rest = cluster_patterns(averages_rest, rootdir, feature="rest", max_d=1.25, label='weekly')

    for i in species_cluster_spd.cluster.unique():
        species_subset = species_cluster_spd.loc[species_cluster_spd.cluster == i, 'species_six'].tolist()
        subset = averages_spd[species_subset]
        plot_bin_cluster(rootdir, subset, i, change_times_d, feature='Speed')

    for i in species_cluster_move.cluster.unique():
        species_subset = species_cluster_move.loc[species_cluster_move.cluster == i, 'species_six'].tolist()
        subset = averages_move[species_subset]
        plot_bin_cluster(rootdir, subset, i, change_times_d, feature='Movement')

    for i in species_cluster_rest.cluster.unique():
        species_subset = species_cluster_rest.loc[species_cluster_rest.cluster == i, 'species_six'].tolist()
        subset = averages_rest[species_subset]
        plot_bin_cluster(rootdir, subset, i, change_times_d, feature='Rest')

    return species_cluster_spd, species_cluster_move, species_cluster_rest
