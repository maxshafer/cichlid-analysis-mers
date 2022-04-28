from tkinter.filedialog import askdirectory
from tkinter import *
import warnings
import os

import datetime as dt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from cichlidanalysis.io.als_files import load_ds_als_files
from cichlidanalysis.utils.timings import load_timings
from cichlidanalysis.utils.species_names import six_letter_sp_name
from cichlidanalysis.utils.species_metrics import add_metrics, tribe_cols
from cichlidanalysis.analysis.processing import feature_daily, species_feature_fish_daily_ave, \
    fish_tracks_add_day_twilight_night, add_day_number_fish_tracks
from cichlidanalysis.analysis.diel_pattern import diel_pattern_stats_individ_bin, diel_pattern_stats_species_bin
from cichlidanalysis.analysis.self_correlations import species_daily_corr, fish_daily_corr, fish_weekly_corr
from cichlidanalysis.analysis.crepuscular_pattern import crepuscular_peaks
from cichlidanalysis.plotting.cluster_plots import cluster_all_fish, cluster_species_daily
from cichlidanalysis.plotting.plot_diel_patterns import plot_day_night_species, plot_cre_dawn_dusk_strip_box, \
    plot_day_night_species_ave
from cichlidanalysis.plotting.speed_plots import plot_ridge_plots
from cichlidanalysis.analysis.clustering_patterns import run_species_pattern_cluster_daily, run_species_pattern_cluster_weekly

# debug pycharm problem
warnings.simplefilter(action='ignore', category=FutureWarning)


def setup_run_binned(rootdir):
    fish_tracks_bin = load_ds_als_files(rootdir, "*als_30m.csv")
    fish_tracks_bin = fish_tracks_bin.reset_index(drop=True)
    fish_tracks_bin['time_of_day_dt'] = fish_tracks_bin.ts.apply(lambda row: int(str(row)[11:16][:-3]) * 60 + int(str(row)[11:16][-2:]))
    fish_tracks_bin.loc[fish_tracks_bin.species == 'Aaltolamprologus calvus', 'species'] = 'Altolamprologus calvus'
    fish_tracks_bin.FishID = fish_tracks_bin.FishID.str.replace('Aaltolamprologus', 'Altolamprologus')

    # get each fish ID and all species
    fish_IDs = fish_tracks_bin['FishID'].unique()
    species = fish_tracks_bin['species'].unique()

    # getting extra data (colours for plotting, species metrics)
    species_sixes = six_letter_sp_name(species)
    tribe_col = tribe_cols()
    metrics_path = '/Users/annikanichols/Desktop/cichlid_species_database.xlsx'
    sp_metrics = add_metrics(species_sixes, metrics_path)

    # add species six name and tribe
    fish_tracks_bin['species_six'] = fish_tracks_bin.apply(lambda row: six_letter_sp_name(row.species)[0], axis=1)
    fish_tracks_bin = pd.merge(fish_tracks_bin, sp_metrics.loc[:, ['species_six', 'tribe']], how='left')
    fish_tracks_bin = add_day_number_fish_tracks(fish_tracks_bin)

    return fish_tracks_bin, sp_metrics, tribe_col, species_sixes, species, fish_IDs


if __name__ == '__main__':
    # Allows user to select top directory and load all als files here
    root = Tk()
    rootdir = askdirectory(parent=root)
    root.destroy()

    fish_tracks_bin, sp_metrics, tribe_col, species_sixes, species, fish_IDs = setup_run_binned(rootdir)

    # get timings
    fps, tv_ns, tv_sec, tv_24h_sec, num_days, tv_s_type, change_times_s, change_times_ns, change_times_h, \
    day_ns, day_s, change_times_d, change_times_m, change_times_datetime, change_times_unit\
        = load_timings(fish_tracks_bin[fish_tracks_bin.FishID == fish_IDs[0]].shape[0])
    day_unit = dt.datetime.strptime("1970:1", "%Y:%d")

    # ###########################
    # ## ridge plots for each feature ###
    averages_vp, date_time_obj_vp, sp_vp_combined, averages_spd, sp_spd_combined, averages_rest, sp_rest_combined, \
    averages_move, sp_move_combined = plot_ridge_plots(fish_tracks_bin, change_times_datetime,
                                                       rootdir, sp_metrics, tribe_col)

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
    # # correlations for days across week for an individual
    # week_corr(rootdir, fish_tracks_bin, 'rest')

    features = ['speed_mm', 'rest']
    for feature in features:
        first = True
        for species_name in species:
            # correlations for individuals across daily average
            fish_daily_ave_feature = species_feature_fish_daily_ave(fish_tracks_bin, species_name, feature)
            # correlations for individuals average days
            corr_vals_f = fish_daily_corr(fish_daily_ave_feature, feature, species_name, rootdir)

            if first:
                corr_vals = pd.DataFrame(corr_vals_f, columns=[six_letter_sp_name(species_name)[0]])
                first = False
            else:
                corr_vals = pd.concat([corr_vals, pd.DataFrame(corr_vals_f, columns=[six_letter_sp_name(species_name)
                                                                                     [0]])], axis=1)

        corr_vals_long = pd.melt(corr_vals, var_name='species_six', value_name='corr_coef')

        f, ax = plt.subplots(figsize=(4, 10))
        sns.boxplot(data=corr_vals_long, x='corr_coef', y='species_six', ax=ax, fliersize=0,
                    order=corr_vals_long.groupby('species_six').mean().sort_values("corr_coef").index.to_list())
        sns.stripplot(data=corr_vals_long, x='corr_coef', y='species_six', color=".2", ax=ax, size=3,
                      order=corr_vals_long.groupby('species_six').mean().sort_values("corr_coef").index.to_list())
        ax.set(xlabel='Correlation', ylabel='Species')
        ax.set(xlim=(-1, 1))
        ax = plt.axvline(0, ls='--', color='k')
        plt.tight_layout()
        plt.savefig(os.path.join(rootdir, "daily_fish_corr_coefs_{0}_{1}.png".format(feature, dt.date.today())))
        plt.close()

        # correlations for individuals across week
        # _ = fish_weekly_corr(rootdir, fish_tracks_bin, feature, 'single')

    # correlations for species and clusters
    species_daily_corr(rootdir, aves_ave_spd, 'ave','speed_mm', 'single')
    species_daily_corr(rootdir, aves_ave_rest, 'ave', 'rest', 'single')
    species_daily_corr(rootdir, aves_ave_move, 'ave', 'movement', 'single')

    species_cluster_spd, species_cluster_move, species_cluster_rest = run_species_pattern_cluster_daily(aves_ave_spd,
                                                                                                  aves_ave_move,
                                                                                                  aves_ave_rest,
                                                                                                  rootdir)
    # species_cluster_spd_wk, species_cluster_move_wk, species_cluster_rest_wk = run_species_pattern_cluster_weekly(
    #     averages_spd, averages_move, averages_rest, rootdir)

    # ###########################
    # ### Define diel pattern ###
    fish_tracks_bin = fish_tracks_add_day_twilight_night(fish_tracks_bin)
    fish_diel_patterns = diel_pattern_stats_individ_bin(fish_tracks_bin, feature='rest')
    fish_diel_patterns_sp = diel_pattern_stats_species_bin(fish_tracks_bin, feature='rest')
    plot_day_night_species_ave(rootdir, fish_diel_patterns, fish_diel_patterns_sp, feature='rest')

    fish_diel_patterns_move = diel_pattern_stats_individ_bin(fish_tracks_bin, feature='movement')
    fish_diel_patterns_move_sp = diel_pattern_stats_species_bin(fish_tracks_bin, feature='movement')
    plot_day_night_species_ave(rootdir, fish_diel_patterns_move, fish_diel_patterns_move_sp, feature="movement")

    fish_diel_patterns_spd = diel_pattern_stats_individ_bin(fish_tracks_bin, feature='speed_mm')
    fish_diel_patterns_spd_sp = diel_pattern_stats_species_bin(fish_tracks_bin, feature='speed_mm')
    plot_day_night_species_ave(rootdir, fish_diel_patterns_spd, fish_diel_patterns_spd_sp, feature='speed_mm')

    # better crepuscular
    # feature = 'rest'
    # crespuscular_daily_ave_fish(rootdir, feature, fish_tracks_bin, species)  # for plotting daily average for each species
    # crespuscular_weekly_fish(rootdir, feature, fish_tracks_bin, species)     # for plotting weekly data for each species

    feature = 'speed_mm'
    cres_peaks = crepuscular_peaks(feature, fish_tracks_bin, species, fish_diel_patterns_sp)
    plot_cre_dawn_dusk_strip_box(rootdir, cres_peaks, feature)

    # make and save diel patterns csv
    cresp_sp = cres_peaks.groupby(['species_six', 'species']).mean().reset_index(level=[1])
    diel_sp = fish_diel_patterns.groupby('species_six').mean()
    diel_patterns_df = pd.concat([cresp_sp, diel_sp.day_night_dif], axis=1).reset_index()
    diel_patterns_df = diel_patterns_df.merge(species_cluster_spd, on="species_six")

    diel_patterns_df.to_csv(os.path.join(rootdir, "combined_diel_patterns_{}_dp.csv".format(dt.date.today())))
    print("Finished saving out diel pattern data")

    # ## feature vs time of day density plot
    # ax = sns.displot(pd.melt(aves_ave_move.reset_index(), id_vars='time_of_day'), x='time_of_day', y='value')
    # for axes in ax.axes.flat:
    #     _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=90)
