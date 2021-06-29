import os

import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import hsv
import pandas as pd

from cichlidanalysis.io.meta import add_meta_from_name
from cichlidanalysis.analysis.processing import species_feature_fish_daily_ave
from cichlidanalysis.utils.species_metrics import add_metrics, tribe_cols


def cluster_all_fish(rootdir, fish_tracks_ds):
    """ clustering of daily average of individuals, massive clustermap!

    :param rootdir:
    :param fish_tracks_ds:
    :return:
    """

    species = fish_tracks_ds['species'].unique()
    features = ['speed_mm', 'rest', 'movement']
    for feature in features:
        first = True
        for species_name in species:
            fish_daily_ave_feature = species_feature_fish_daily_ave(fish_tracks_ds, species_name, feature)

            if first:
                all_fish_daily_ave_feature = fish_daily_ave_feature
                first = False
            else:
                all_fish_daily_ave_feature = pd.concat([all_fish_daily_ave_feature, fish_daily_ave_feature], axis=1)

        col_species = add_meta_from_name(all_fish_daily_ave_feature.columns, 'species').T
        species_code = col_species.species.astype('category').cat.codes

        num_categories = len(set(col_species.species))
        colors = [hsv(float(i) / num_categories) for i in species_code]

        sns.clustermap(all_fish_daily_ave_feature, row_cluster=False, col_colors=colors, xticklabels=col_species.species)
        plt.savefig(os.path.join(rootdir, "all_fish_daily_clustered_30min_{0}_{1}.png".format(feature, dt.date.today())))


def cluster_species_daily(rootdir, aves_ave_spd, aves_ave_vp, aves_ave_rest, aves_ave_move, species_sixes):
    """ Daily clustered heatmap of species features

    :return:
    """
    tribe_col = tribe_cols()

    metrics_path = '/Users/annikanichols/Desktop/cichlid_species_database.xlsx'
    sp_metrics = add_metrics(species_sixes, metrics_path)

    row_cols = []
    for i in sp_metrics.tribe:
        row_cols.append(tribe_col[i])

    # row_cols_species = pd.DataFrame(row_cols, index=[aves_ave_spd.columns.tolist()]).apply(tuple, axis=1)
    row_cols_unnamed = pd.DataFrame(row_cols).apply(tuple, axis=1)

    # with species names
    ax = sns.clustermap(aves_ave_spd.T, figsize=(7, 5), col_cluster=False, method='single', metric='correlation',
                        yticklabels=True)
    ax.fig.suptitle("Speed mm/s")
    plt.savefig(os.path.join(rootdir, "all_species_daily_clustered_30min_{0}_{1}.png".format("speed",
                                                                                             dt.date.today())))
    plt.close()

    # with tribe colours
    ax = sns.clustermap(aves_ave_spd.T.reset_index(drop=True), figsize=(7, 5), col_cluster=False, method='single',
                        metric='correlation', row_colors=row_cols_unnamed)
    ax.fig.suptitle("Speed mm/s")
    plt.close()

    ax = sns.clustermap(aves_ave_vp.T.reset_index(drop=True), figsize=(7, 5), col_cluster=False, method='single',
                        metric='correlation', row_colors=row_cols_unnamed)
    ax.fig.suptitle("Vertical position")
    plt.savefig(os.path.join(rootdir, "all_species_daily_clustered_30min_{0}_{1}.png".format("vertical-position",
                                                                                             dt.date.today())))
    plt.close()

    ax = sns.clustermap(aves_ave_rest.T, figsize=(7, 5), col_cluster=False, method='single', metric='correlation',
                        yticklabels=True)
    ax.fig.suptitle("Rest")
    plt.savefig(os.path.join(rootdir, "all_species_daily_clustered_30min_{0}_{1}.png".format("rest", dt.date.today())))
    plt.close()

    ax = sns.clustermap(aves_ave_rest.T.reset_index(drop=True), figsize=(7, 5), col_cluster=False, method='single',
                        metric='correlation', row_colors=row_cols_unnamed, yticklabels=True)
    ax.fig.suptitle("Rest")
    plt.close()


    ax = sns.clustermap(aves_ave_move.T, figsize=(7, 5), col_cluster=False, method='single', metric='correlation',
                        yticklabels=True)
    ax.fig.suptitle("Movement")
    plt.savefig(os.path.join(rootdir, "all_species_daily_clustered_30min_{0}_{1}.png".format("movement", dt.date.today())))
    plt.close()

    ax = sns.clustermap(aves_ave_move.T.reset_index(drop=True), figsize=(7, 5), col_cluster=False, method='single',
                        metric='correlation', row_colors=row_cols_unnamed)
    ax.fig.suptitle("Movement")
    plt.close()
