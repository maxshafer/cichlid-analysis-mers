from tkinter.filedialog import askdirectory
from tkinter import *
import warnings

import datetime as dt

from cichlidanalysis.io.tracks import load_ds_als_files
from cichlidanalysis.utils.timings import load_timings
from cichlidanalysis.utils.species_names import shorten_sp_name, six_letter_sp_name
from cichlidanalysis.utils.species_metrics import add_metrics, tribe_cols
from cichlidanalysis.analysis.processing import feature_daily, species_feature_fish_daily_ave, \
    fish_tracks_add_day_twilight_night, add_day_number_fish_tracks
from cichlidanalysis.analysis.diel_pattern import diel_pattern_ttest_individ_ds
from cichlidanalysis.analysis.self_correlations import species_daily_corr, fish_daily_corr, fish_weekly_corr
from cichlidanalysis.analysis.crepuscular_pattern import crepuscular_peaks, crespuscular_daily_ave_fish, \
    crespuscular_weekly_fish
from cichlidanalysis.plotting.cluster_plots import cluster_all_fish, cluster_species_daily
from cichlidanalysis.plotting.plot_diel_patterns import plot_day_night_species, plot_cre_dawn_dusk_strip_box
from cichlidanalysis.plotting.speed_plots import plot_spd_30min_combined

# debug pycharm problem
warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == '__main__':
    # Allows user to select top directory and load all als files here
    root = Tk()
    root.withdraw()
    root.update()
    rootdir = askdirectory(parent=root)
    root.destroy()

    fish_tracks_ds = load_ds_als_files(rootdir, "*als_30m.csv")
    fish_tracks_ds = fish_tracks_ds.reset_index(drop=True)
    fish_tracks_ds['time_of_day_dt'] = fish_tracks_ds.ts.apply(lambda row: int(str(row)[11:16][:-3]) * 60 + int(str(row)[11:16][-2:]))
    fish_tracks_ds.loc[fish_tracks_ds.species == 'Aaltolamprologus calvus', 'species'] = 'Altolamprologus calvus'

    # get each fish ID and all species
    fish_IDs = fish_tracks_ds['FishID'].unique()
    species = fish_tracks_ds['species'].unique()

    # reorganising
    species_short = shorten_sp_name(species)
    species_sixes = six_letter_sp_name(species)

    tribe_col = tribe_cols()

    metrics_path = '/Users/annikanichols/Desktop/cichlid_species_database.xlsx'
    sp_metrics = add_metrics(species_sixes, metrics_path)

    # get timings
    fps, tv_ns, tv_sec, tv_24h_sec, num_days, tv_s_type, change_times_s, change_times_ns, change_times_h, day_ns, day_s,\
        change_times_d, change_times_m = load_timings(fish_tracks_ds[fish_tracks_ds.FishID == fish_IDs[0]].shape[0])
    change_times_unit = [7*2, 7.5*2, 18.5*2, 19*2]
    change_times_datetime = [dt.datetime.strptime("1970-1-2 07:00:00", '%Y-%m-%d %H:%M:%S'),
                             dt.datetime.strptime("1970-1-2 07:30:00", '%Y-%m-%d %H:%M:%S'),
                             dt.datetime.strptime("1970-1-2 18:30:00", '%Y-%m-%d %H:%M:%S'),
                             dt.datetime.strptime("1970-1-2 19:00:00", '%Y-%m-%d %H:%M:%S'),
                             dt.datetime.strptime("1970-1-3 00:00:00", '%Y-%m-%d %H:%M:%S')]
    day_unit = dt.datetime.strptime("1970:1", "%Y:%d")

    # ###########################
    # ## ridge plots for each feature ###
    feature, ymax, span_max, ylabeling = 'vertical_pos', 1, 0.8, 'Vertical position'
    averages_vp, date_time_obj_vp, sp_vp_combined = plot_spd_30min_combined(fish_tracks_ds, feature, ymax, span_max,
                                                                            ylabeling, change_times_datetime, rootdir)
    feature, ymax, span_max, ylabeling = 'speed_mm', 95, 80, 'Speed mm/s'
    averages_spd, _, sp_spd_combined = plot_spd_30min_combined(fish_tracks_ds, feature, ymax, span_max,
                                                                              ylabeling, change_times_datetime, rootdir)
    feature, ymax, span_max, ylabeling = 'rest', 1, 0.8, 'Rest'
    averages_rest, _, sp_rest_combined = plot_spd_30min_combined(fish_tracks_ds, feature, ymax, span_max,
                                                                              ylabeling, change_times_datetime, rootdir)
    feature, ymax, span_max, ylabeling = 'movement', 1, 0.8, 'Movement'
    averages_move, _, sp_move_combined = plot_spd_30min_combined(fish_tracks_ds, feature, ymax, span_max,
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
    cluster_species_daily(rootdir, aves_ave_spd, aves_ave_vp, aves_ave_rest, aves_ave_move, species_sixes)
    cluster_all_fish(rootdir, fish_tracks_ds)

    # ###########################
    # ## correlations ##
    fish_tracks_ds = add_day_number_fish_tracks(fish_tracks_ds)

    # correlations for days across week for an individual
    # week_corr(rootdir, fish_tracks_ds, 'rest')

    features = ['speed_mm', 'rest']
    for feature in features:
        for species_name in species:
            # correlations for individuals across daily average
            fish_daily_ave_feature = species_feature_fish_daily_ave(fish_tracks_ds, species_name, feature)
            fish_daily_corr(fish_daily_ave_feature, feature, species_name, rootdir)

        # correlations for individuals across week
        _ = fish_weekly_corr(rootdir, fish_tracks_ds, feature, 'single')

    # correlations for species
    species_daily_corr(rootdir, aves_ave_spd, 'speed_mm', 'single')
    species_daily_corr(rootdir, aves_ave_rest, 'rest', 'single')

    # ###########################
    # ### Define diel pattern ###
    fish_tracks_ds = fish_tracks_add_day_twilight_night(fish_tracks_ds)
    fish_tracks_ds = add_day_number_fish_tracks(fish_tracks_ds)
    fish_diel_patterns = diel_pattern_ttest_individ_ds(fish_tracks_ds, feature='rest')

    # define species diel pattern
    states = ['nocturnal', 'diurnal']
    fish_diel_patterns['species_diel_pattern'] = 'undefined'
    for species_name in species_sixes:
        for state in states:
            if ((fish_diel_patterns.loc[fish_diel_patterns.species_six == species_name, 'diel_pattern'] == state)*1).mean() > 0.5:
                fish_diel_patterns.loc[fish_diel_patterns.species_six == species_name, 'species_diel_pattern'] = state
        print("{} is {}".format(species_name, fish_diel_patterns.loc[fish_diel_patterns.species_six == species_name, 'species_diel_pattern'].unique()))

    plot_day_night_species(rootdir, fish_diel_patterns)

# better crepuscular
crespuscular_daily_ave_fish(rootdir, feature, fish_tracks_ds, species)  # for plotting daily average for each species
crespuscular_weekly_fish(rootdir, feature, fish_tracks_ds, species)     # for plotting weekly data for each species

cres_peaks = crepuscular_peaks(feature, fish_tracks_ds, species, fish_diel_patterns)
plot_cre_dawn_dusk_strip_box(rootdir, cres_peaks)



# # calculate ave and stdv
# average = sp_feature.mean(axis=1)
# averages[species_n, :] = average[0:303]
#
#
#
#     fig = plt.figure(figsize=(10, 5))
#     plt.hist(species_peaks_df.peak_amplitude)
#
#     fig = plt.figure(figsize=(10, 5))
#     plt.hist(species_peaks_df_dusk.loc[species_peaks_df_dusk.peak_loc == 0, 'peak_amplitude'])
#
#         x = fish_feature.iloc[:, i]
#         fig = plt.figure(figsize=(10, 5))
#         plt.plot(x)
#         plt.plot(x.reset_index().index[fish_peaks[0, :].astype(int)].values, fish_peaks[1, :],   "o", color="r")
#         plt.title(species_name)
#         plt.show()

# 	1. Find peaks in daily average of Individuals and  species
# 	2. Find peaks across week
# 	3. Find amplitude of peaks
# For non-peaks -  take the most common peak bin
#
# feature_i.loc[feature_i.peak_loc > 0].groupby('FishID').mean().peak_loc
# feature_i.loc[feature_i.peak_loc > 0].groupby('FishID').peak_loc.agg(pd.Series.mode)
#
# x = fish_feature.iloc[:, i]
# plt.plot(fish_peaks[0, :], x[(fish_peaks[0, :]).astype(int)],  "x", color="k")

