import os

import datetime as dt
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
import pandas as pd
from scipy.spatial.distance import pdist

from cichlidanalysis.plotting.daily_plots import daily_ave_spd, daily_ave_move, daily_ave_rest

# Inspiration from:
# https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/


def cluster_patterns(aves_ave_feature, rootdir, feature, max_d=1.3):
    individ_corr = aves_ave_feature.corr(method='pearson')
    Z = linkage(individ_corr, 'single')

    plt.figure(figsize=[8, 5])
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
        labels=individ_corr.index,
        color_threshold=max_d
    )
    plt.axhline(max_d, color='k')
    plt.show()
    plt.savefig(os.path.join(rootdir, "species_corr_dendrogram_max_d-{0}_{1}_{2}.png".format(max_d,
                                                                                             feature, dt.date.today())))
    plt.close()

    c, coph_dists = cophenet(Z, pdist(individ_corr))
    print("The cophenetic correlation coefficient is {}, The closer the value is to 1, the better the clustering "
          "preserves the original distances".format(round(c, 2)))

    clusters = fcluster(Z, max_d, criterion='distance')
    d = {'species_six': individ_corr.index, "cluster": clusters}
    species_cluster = pd.DataFrame(d)
    species_cluster = species_cluster.sort_values(by="cluster")

    return species_cluster


def run_species_pattern_cluster(aves_ave_spd, aves_ave_move, aves_ave_rest, rootdir):

    change_times_unit = [7*2, 7.5*2, 18.5*2, 19*2]

    species_cluster_spd = cluster_patterns(aves_ave_spd, rootdir, feature="speed", max_d=1.3)
    species_cluster_move = cluster_patterns(aves_ave_move, rootdir, feature="movement", max_d=1.2)
    species_cluster_rest = cluster_patterns(aves_ave_rest, rootdir, feature="rest", max_d=1.05)

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
