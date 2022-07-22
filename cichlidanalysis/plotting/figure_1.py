import os

import numpy as np
import datetime as dt
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)
from matplotlib.dates import DateFormatter
from datetime import timedelta

from cichlidanalysis.utils.timings import output_timings


def getKeysByValue(dict, value_to_find):
    listOfKeys = list()
    listOfItems = dict.items()
    for item in listOfItems:
        if item[1].count(value_to_find) > 0:
            listOfKeys.append(item[0])
    return listOfKeys

def cluster_dics():
    # dic_complex = {'diurnal': [6], 'nocturnal': [1], 'crepuscular1': [2], 'crepuscular2': [4], 'crepuscular3': [5],
    #        'undefined': [3, 7, 8, 9, 10, 11]}
    # dic_simple = {'diurnal': [6], 'nocturnal': [1], 'crepuscular': [2, 4, 5], 'undefined': [3, 7, 8, 9, 10, 11]}

    # final
    dic_complex = {'diurnal': [7], 'nocturnal': [1], 'crepuscular1': [2], 'crepuscular2': [4], 'crepuscular3': [6],
           'crepuscular4': [5], 'undefined': [3, 8, 9, 10, 11, 12]}
    dic_simple = {'diurnal': [7], 'nocturnal': [1], 'crepuscular': [2, 4, 5, 6], 'undefined': [3, 8, 9, 10, 11, 12]}

    col_dic_complex = {'diurnal': 'orange', 'nocturnal': 'royalblue', 'crepuscular1': 'orchid', 'crepuscular2':
        'mediumorchid', 'crepuscular3': 'darkorchid',  'crepuscular4': 'mediumpurple', 'undefined': 'dimgrey'}
    col_dic_simple = {'diurnal': 'orange', 'nocturnal': 'royalblue', 'crepuscular': 'orchid', 'undefined': 'dimgrey'}

    # cluster_dic = {'1': 'nocturnal', '2': 'crepuscular1', '3': 'undefined', '4': 'crepuscular2', '5': 'crepuscular3',
    #                '6': 'diurnal', '7': 'undefined', '8': 'undefined', '9': 'undefined', '10': 'undefined',
    #                '11': 'undefined'}
    cluster_order = [12, 11, 10, 9, 3, 1, 2, 4, 6, 5, 8, 7]
    return dic_complex, dic_simple, col_dic_simple, col_dic_complex, col_dic_simple, cluster_order


def dendrogram_sp_clustering(aves_ave_spd, link_method='single', max_d=1.35):
    """ Dendrogram of the clustering as done in clustermap. This allows me to get out the clusters

    :param aves_ave_spd:
    :param link_method:
    :param max_d:
    :return:
    """
    dic_complex, dic_simple, col_dic_simple, col_dic_complex, col_dic_simple, cluster_order = cluster_dics()
    aves_ave_spd = aves_ave_spd.reindex(sorted(aves_ave_spd.columns), axis=1)
    individ_corr = aves_ave_spd.corr(method='pearson')
    z = linkage(individ_corr, link_method)

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
    plt.close()

    clusters = fcluster(z, max_d, criterion='distance')
    d = {'species_six': individ_corr.index, "cluster": clusters}
    species_cluster = pd.DataFrame(d)
    species_cluster = species_cluster.sort_values(by="cluster")
    species_cluster['colour'] = 'grey'
    for col in col_dic_complex:
        cluster_n = dic_complex[col]
        species_cluster.loc[species_cluster.cluster.isin(cluster_n), 'colour'] = col_dic_complex[col]

    return individ_corr, species_cluster


def clustered_spd_map(rootdir, aves_ave_spd, link_method='single'):

    individ_corr, species_cluster = dendrogram_sp_clustering(aves_ave_spd, link_method=link_method, max_d=1.35)

    # Plot cluster map with one dendrogram, main clusters (hardcoded) as row/col colours
    cg = sns.clustermap(individ_corr, figsize=(12, 12), method=link_method, metric='euclidean', vmin=-1, vmax=1,
                        cmap='viridis', yticklabels=True, xticklabels=True,
                        row_colors=species_cluster.sort_values(by='species_six').colour.to_list(),
                        col_colors=species_cluster.sort_values(by='species_six').colour.to_list(),
                        cbar_kws={'label': 'Correlation coefficient'})
    cg.ax_col_dendrogram.set_visible(False)
    plt.savefig(os.path.join(rootdir, "figure_panel_1_clustermap_{}.png".format(dt.date.today())))
    plt.close()
    return


def cluster_daily_ave(rootdir, aves_ave_spd, link_method='single'):
    dic_complex, dic_simple, col_dic_simple, col_dic_complex, col_dic_simple, cluster_order = cluster_dics()
    change_times_s, change_times_ns, change_times_m, change_times_h, day_ns, day_s, change_times_d, \
    change_times_datetime, change_times_unit = output_timings()
    individ_corr, species_cluster = dendrogram_sp_clustering(aves_ave_spd, link_method=link_method, max_d=1.35)

    date_form = DateFormatter('%H')
    feature, ymax, span_max, ylabeling = 'speed_mm', 95, len(cluster_order)*25, 'Speed mm/s'
    fig = plt.figure(figsize=(2, 9))

    # create time vector in datetime format
    date_time_obj = []
    for i in aves_ave_spd.index:
        date_time_obj.append(dt.datetime.strptime(i, '%H:%M') + timedelta(days=(365.25 * 70), hours=12))

    day_n = 0
    plt.fill_between(
        [dt.datetime.strptime("1970-1-2 00:00:00", '%Y-%m-%d %H:%M:%S') + timedelta(days=day_n),
         change_times_datetime[0] + timedelta(days=day_n)], [span_max, span_max], 0,
        color='lightblue', alpha=0.5, linewidth=0, zorder=1)
    plt.fill_between([change_times_datetime[0] + timedelta(days=day_n),
                      change_times_datetime[1] + timedelta(days=day_n)], [span_max, span_max], 0,
                     color='wheat',
                     alpha=0.5, linewidth=0)
    plt.fill_between([change_times_datetime[2] + timedelta(days=day_n), change_times_datetime[3] + timedelta
    (days=day_n)], [span_max, span_max], 0, color='wheat', alpha=0.5, linewidth=0)
    plt.fill_between([change_times_datetime[3] + timedelta(days=day_n), change_times_datetime[4] + timedelta
    (days=day_n)], [span_max, span_max], 0, color='lightblue', alpha=0.5, linewidth=0)

    top = len(cluster_order) * 25
    for cluster_count, cluster_n in enumerate(cluster_order):
        subset_spe = species_cluster.loc[species_cluster.cluster == cluster_n, 'species_six']
        subset_spd = aves_ave_spd.loc[:, aves_ave_spd.columns.isin(subset_spe)]
        # subset_spd_stdev = subset_spd.std(axis=1)
        daily_speed = subset_spd.mean(axis=1) + top - 25 - cluster_count * 25

        colour = col_dic_complex[getKeysByValue(dic_complex, cluster_n)[0]]
        # plotting
        # ax = sns.lineplot(x=date_time_obj, y=(daily_speed + subset_spd_stdev), color='lightgrey')
        # ax = sns.lineplot(x=date_time_obj, y=(daily_speed - subset_spd_stdev), color='lightgrey')
        for species in subset_spe:
            ax = sns.lineplot(x=date_time_obj, y=subset_spd.loc[:, species] + top - 25 - cluster_count * 25,
                              color=colour, alpha=0.3)
        ax = sns.lineplot(x=date_time_obj, y=daily_speed, color=colour, linewidth=3)

    # setting uniform x and y lims
    ax = plt.gca()
    ax.set_xlim(min(date_time_obj), dt.datetime.strptime("1970-1-3 00:00:00", '%Y-%m-%d %H:%M:%S'))
    ax.set_ylim(0, span_max)

    ax.set_xlabel("Time", fontsize=10) #, fontweight="bold")
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.xaxis.set_major_formatter(date_form)

    ax.yaxis.set_ticks(np.arange(0, 30, step=10))
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel(ylabeling, loc='bottom')

    spines = ["top", "right", "left", "bottom"]
    for s in spines:
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "figure_panel_1_daily_traces_{}.png".format(dt.date.today())))
    plt.close()
    return

