from tkinter.filedialog import askdirectory
from tkinter import *
import warnings
import os

import datetime as dt
import pandas as pd

from cichlidanalysis.io.als_files import load_ds_als_files
from cichlidanalysis.utils.timings import load_timings
from cichlidanalysis.utils.species_names import shorten_sp_name, six_letter_sp_name
from cichlidanalysis.utils.species_metrics import add_metrics, tribe_cols
from cichlidanalysis.analysis.processing import feature_daily, species_feature_fish_daily_ave, \
    fish_tracks_add_day_twilight_night, add_day_number_fish_tracks
from cichlidanalysis.analysis.diel_pattern import diel_pattern_ttest_individ_ds
from cichlidanalysis.analysis.self_correlations import species_daily_corr, fish_daily_corr, fish_weekly_corr
from cichlidanalysis.analysis.crepuscular_pattern import crepuscular_peaks
from cichlidanalysis.plotting.cluster_plots import cluster_all_fish, cluster_species_daily
from cichlidanalysis.plotting.plot_diel_patterns import plot_day_night_species, plot_cre_dawn_dusk_strip_box
from cichlidanalysis.plotting.speed_plots import plot_spd_30min_combined, plot_spd_30min_combined_daily

# debug pycharm problem
warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == '__main__':
    # Allows user to select top directory and load all als files here
    root = Tk()
    root.withdraw()
    root.update()
    rootdir = askdirectory(parent=root)
    root.destroy()

    fish_tracks_bin = load_ds_als_files(rootdir, "*als_30m.csv")
    fish_tracks_bin = fish_tracks_bin.reset_index(drop=True)
    fish_tracks_bin['time_of_day_dt'] = fish_tracks_bin.ts.apply(lambda row: int(str(row)[11:16][:-3]) * 60 + int(str(row)[11:16][-2:]))
    fish_tracks_bin.loc[fish_tracks_bin.species == 'Aaltolamprologus calvus', 'species'] = 'Altolamprologus calvus'
    fish_tracks_bin.FishID = fish_tracks_bin.FishID.str.replace('Aaltolamprologus', 'Altolamprologus')

    # get each fish ID and all species
    fish_IDs = fish_tracks_bin['FishID'].unique()
    species = fish_tracks_bin['species'].unique()

    # reorganising
    # species_short = shorten_sp_name(species)
    species_sixes = six_letter_sp_name(species)

    tribe_col = tribe_cols()

    metrics_path = '/Users/annikanichols/Desktop/cichlid_species_database.xlsx'
    sp_metrics = add_metrics(species_sixes, metrics_path)

    # add species six name and tribe
    fish_tracks_bin['species_six'] = fish_tracks_bin.apply(lambda row: six_letter_sp_name(row.species)[0], axis=1)
    fish_tracks_bin = pd.merge(fish_tracks_bin, sp_metrics.loc[:, ['species_six', 'tribe']], how='left')

    # get timings
    fps, tv_ns, tv_sec, tv_24h_sec, num_days, tv_s_type, change_times_s, change_times_ns, change_times_h, day_ns, day_s,\
        change_times_d, change_times_m = load_timings(fish_tracks_bin[fish_tracks_bin.FishID == fish_IDs[0]].shape[0])
    change_times_unit = [7*2, 7.5*2, 18.5*2, 19*2]
    change_times_datetime = [dt.datetime.strptime("1970-1-2 07:00:00", '%Y-%m-%d %H:%M:%S'),
                             dt.datetime.strptime("1970-1-2 07:30:00", '%Y-%m-%d %H:%M:%S'),
                             dt.datetime.strptime("1970-1-2 18:30:00", '%Y-%m-%d %H:%M:%S'),
                             dt.datetime.strptime("1970-1-2 19:00:00", '%Y-%m-%d %H:%M:%S'),
                             dt.datetime.strptime("1970-1-3 00:00:00", '%Y-%m-%d %H:%M:%S')]
    day_unit = dt.datetime.strptime("1970:1", "%Y:%d")

    # ###########################
    # ## ridge plots for each feature ###
    # daily
    feature, ymax, span_max, ylabeling = 'speed_mm', 95, 80, 'Speed mm/s'
    plot_spd_30min_combined_daily(fish_tracks_bin, feature, ymax, span_max, ylabeling, change_times_datetime, rootdir,
                                  sp_metrics, tribe_col)
    feature, ymax, span_max, ylabeling = 'movement', 1, 0.8, 'Movement'
    plot_spd_30min_combined_daily(fish_tracks_bin, feature, ymax, span_max, ylabeling, change_times_datetime, rootdir,
                                  sp_metrics, tribe_col)
    feature, ymax, span_max, ylabeling = 'rest', 1, 0.8, 'Rest'
    plot_spd_30min_combined_daily(fish_tracks_bin, feature, ymax, span_max, ylabeling, change_times_datetime, rootdir,
                                  sp_metrics, tribe_col)

    # weekly
    feature, ymax, span_max, ylabeling = 'vertical_pos', 1, 0.8, 'Vertical position'
    averages_vp, date_time_obj_vp, sp_vp_combined = plot_spd_30min_combined(fish_tracks_bin, feature, ymax, span_max,
                                                                            ylabeling, change_times_datetime, rootdir)
    feature, ymax, span_max, ylabeling = 'speed_mm', 95, 80, 'Speed mm/s'
    averages_spd, _, sp_spd_combined = plot_spd_30min_combined(fish_tracks_bin, feature, ymax, span_max,
                                                               ylabeling, change_times_datetime, rootdir)
    feature, ymax, span_max, ylabeling = 'rest', 1, 0.8, 'Rest'
    averages_rest, _, sp_rest_combined = plot_spd_30min_combined(fish_tracks_bin, feature, ymax, span_max,
                                                                 ylabeling, change_times_datetime, rootdir)
    feature, ymax, span_max, ylabeling = 'movement', 1, 0.8, 'Movement'
    averages_move, _, sp_move_combined = plot_spd_30min_combined(fish_tracks_bin, feature, ymax, span_max,
                                                                 ylabeling, change_times_datetime, rootdir)

    # ### generate averages of the the averages ###
    aves_ave_spd = feature_daily(averages_spd)
    aves_ave_vp = feature_daily(averages_vp)
    aves_ave_rest = feature_daily(averages_rest)
    aves_ave_move = feature_daily(averages_move)

    aves_ave_spd.columns = species_sixes
    aves_ave_vp.columns = species_sixes
    aves_ave_rest.columns = species_sixes
    aves_ave_move.columns = species_sixes

    # ###########################
    # ### clustered heatmaps ###
    # cluster_species_daily(rootdir, aves_ave_spd, aves_ave_vp, aves_ave_rest, aves_ave_move, species_sixes)
    # cluster_all_fish(rootdir, fish_tracks_bin)

    # ###########################
    # ## correlations ##
    fish_tracks_bin = add_day_number_fish_tracks(fish_tracks_bin)

    # correlations for days across week for an individual
    # week_corr(rootdir, fish_tracks_bin, 'rest')

    features = ['speed_mm', 'rest']
    for feature in features:
        for species_name in species:
            # correlations for individuals across daily average
            fish_daily_ave_feature = species_feature_fish_daily_ave(fish_tracks_bin, species_name, feature)
            # correlations for individuals average days
            # fish_daily_corr(fish_daily_ave_feature, feature, species_name, rootdir)

        # correlations for individuals across week
        # _ = fish_weekly_corr(rootdir, fish_tracks_bin, feature, 'single')

    # correlations for species
    species_daily_corr(rootdir, aves_ave_spd, 'speed_mm', 'single')
    species_daily_corr(rootdir, aves_ave_rest, 'rest', 'single')
    species_daily_corr(rootdir, aves_ave_move, 'movement', 'single')

    # ###########################
    # ### Define diel pattern ###
    fish_tracks_bin = fish_tracks_add_day_twilight_night(fish_tracks_bin)
    fish_tracks_bin = add_day_number_fish_tracks(fish_tracks_bin)
    fish_diel_patterns = diel_pattern_ttest_individ_ds(fish_tracks_bin, feature='rest')

    # define species diel pattern
    states = ['nocturnal', 'diurnal']
    fish_diel_patterns['species_diel_pattern'] = 'undefined'
    for species_name in species_sixes:
        for state in states:
            if ((fish_diel_patterns.loc[fish_diel_patterns.species_six == species_name, 'diel_pattern'] == state)*1).mean() > 0.5:
                fish_diel_patterns.loc[fish_diel_patterns.species_six == species_name, 'species_diel_pattern'] = state
        print("{} is {}".format(species_name, fish_diel_patterns.loc[fish_diel_patterns.species_six == species_name, 'species_diel_pattern'].unique()))

    plot_day_night_species(rootdir, fish_diel_patterns, 'rest', 'day_night_dif')

    fish_diel_patterns_move = diel_pattern_ttest_individ_ds(fish_tracks_bin, feature='movement')
    fish_diel_patterns_move['species_diel_pattern'] = 'undefined'
    plot_day_night_species(rootdir, fish_diel_patterns_move, 'movement')
    plot_day_night_species(rootdir, fish_diel_patterns_move, 'movement', 'day_night_dif')

    fish_diel_patterns_spd = diel_pattern_ttest_individ_ds(fish_tracks_bin, feature='speed_mm')
    fish_diel_patterns_spd['species_diel_pattern'] = 'undefined'
    plot_day_night_species(rootdir, fish_diel_patterns_spd, 'speed_mm')
    plot_day_night_species(rootdir, fish_diel_patterns_spd, 'speed_mm', 'day_night_dif')

    # better crepuscular
    # feature = 'rest'
    # crespuscular_daily_ave_fish(rootdir, feature, fish_tracks_bin, species)  # for plotting daily average for each species
    # crespuscular_weekly_fish(rootdir, feature, fish_tracks_bin, species)     # for plotting weekly data for each species

    feature = 'speed_mm'
    cres_peaks = crepuscular_peaks(feature, fish_tracks_bin, species, fish_diel_patterns)
    plot_cre_dawn_dusk_strip_box(rootdir, cres_peaks)

    # make and save diel patterns csv
    cresp_sp = cres_peaks.groupby(['species_six', 'species']).mean().reset_index(level=[1])
    diel_sp = fish_diel_patterns.groupby('species_six').mean()
    diel_patterns_df = pd.concat([cresp_sp, diel_sp.day_night_dif, ], axis=1).reset_index()
    diel_patterns_df.to_csv(os.path.join(rootdir, "combined_diel_patterns_{}_dp.csv".format(dt.date.today())))
    print("Finished saving out diel pattern data")

    ## feature vs time of day density plot
    import seaborn as sns
    ax = sns.displot(pd.melt(aves_ave_move.reset_index(), id_vars='time_of_day'), x='time_of_day', y='value')
    for axes in ax.axes.flat:
        _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=90)


# clusters from daily averages
# https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
import numpy as np
from scipy.spatial.distance import pdist

individ_corr = aves_ave_spd.corr(method='pearson')
Z = linkage(individ_corr, 'single')

plt.figure()
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

# check copheneticcorrelation coefficient, The closer the value is to 1, the better the clustering preserves the
# original distances
c, coph_dists = cophenet(Z, pdist(aves_ave_spd))
print(c)

max_d = 1.3
clusters = fcluster(Z, max_d, criterion='distance')
d = {'species_six': individ_corr.index, "cluster": clusters}
df = pd.DataFrame(d)
df = df.sort_values(by="cluster")

# make average of each cluster
from cichlidanalysis.plotting.daily_plots import daily_ave_spd

change_times_unit = [7*2, 7.5*2, 18.5*2, 19*2]
for i in df.cluster.unique():
    species_subset = df.loc[df.cluster == i, 'species_six'].tolist()
    subset = aves_ave_spd[species_subset]
    cluster_mean = subset.mean(axis=1)
    cluster_std = subset.std(axis=1)
    daily_ave_spd(subset, cluster_std, rootdir, 'cluster_{}'.format(i), change_times_unit)