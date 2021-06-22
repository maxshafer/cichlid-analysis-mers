from tkinter.filedialog import askdirectory
from tkinter import *
import warnings
import os
import copy

import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import hsv
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from cichlidanalysis.io.meta import add_meta_from_name, extract_meta
from cichlidanalysis.io.tracks import load_ds_als_files
from cichlidanalysis.utils.timings import load_timings
from cichlidanalysis.utils.species_names import shorten_sp_name, six_letter_sp_name
from cichlidanalysis.utils.species_metrics import add_metrics, tribe_cols
from cichlidanalysis.plotting.speed_plots import plot_spd_30min_combined
from cichlidanalysis.analysis.processing import feature_daily, species_feature_fish_daily_ave, \
    fish_tracks_add_day_twilight_night, add_day_number_fish_tracks
from cichlidanalysis.analysis.diel_pattern import replace_crep_peaks, make_fish_peaks_df, diel_pattern_ttest_individ_ds
from cichlidanalysis.analysis.self_correlations import species_daily_corr, fish_daily_corr, fish_weekly_corr

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

    # generate
    aves_ave_spd = feature_daily(averages_spd)
    aves_ave_vp = feature_daily(averages_vp)
    aves_ave_rest = feature_daily(averages_rest)
    aves_ave_move = feature_daily(averages_move)

    aves_ave_spd.columns = species_sixes
    aves_ave_vp.columns = species_sixes
    aves_ave_rest.columns = species_sixes
    aves_ave_move.columns = species_sixes

    row_cols = []
    for i in sp_metrics.tribe:
        row_cols.append(tribe_col[i])

    row_cols_2 = pd.DataFrame(row_cols, index=[aves_ave_spd.columns.tolist()]).apply(tuple, axis=1)
    row_cols_1 = pd.DataFrame(row_cols).apply(tuple, axis=1)
    # ax = sns.clustermap(aves_ave_spd.T.reset_index(drop=True), figsize=(7, 5), col_cluster=False, method='single',
    #                     metric='correlation',
    #                     row_colors=row_cols_2)

    ax = sns.clustermap(aves_ave_spd.T, figsize=(7, 5), col_cluster=False, method='single', metric='correlation',
                        yticklabels=True)
    ax.fig.suptitle("Speed mm/s")
    ax = sns.clustermap(aves_ave_spd.T.reset_index(drop=True), figsize=(7, 5), col_cluster=False, method='single',
                        metric='correlation', row_colors=row_cols_1)
    ax.fig.suptitle("Speed mm/s")
    plt.close()
    ax = sns.clustermap(aves_ave_vp.T.reset_index(drop=True), figsize=(7, 5), col_cluster=False, method='single',
                        metric='correlation', row_colors=row_cols_1)
    ax.fig.suptitle("Vertical position")
    plt.close()
    ax = sns.clustermap(aves_ave_rest.T, figsize=(7, 5), col_cluster=False, method='single', metric='correlation',
                        yticklabels=True)
    ax.fig.suptitle("Rest")
    ax = sns.clustermap(aves_ave_rest.T.reset_index(drop=True), figsize=(7, 5), col_cluster=False, method='single',
                        metric='correlation', row_colors=row_cols_1, yticklabels=True)
    ax.fig.suptitle("Rest")
    plt.close()
    ax = sns.clustermap(aves_ave_move.T, figsize=(7, 5), col_cluster=False, method='single', metric='correlation',
                        yticklabels=True)
    ax.fig.suptitle("Movement")
    ax = sns.clustermap(aves_ave_move.T.reset_index(drop=True), figsize=(7, 5), col_cluster=False, method='single',
                        metric='correlation', row_colors=row_cols_1)
    ax.fig.suptitle("Movement")
    plt.close()

    # clustering of daily average of individuals, massive clustermap!
    features = ['speed_mm', 'rest',  'movement']
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

    # ## correlations ##
    # fish_tracks_ds = add_day(fish_tracks_ds)
    fish_tracks_ds = add_day_number_fish_tracks(fish_tracks_ds)
    # correlations for days across week for an individual
    # week_corr(fish_tracks_ds, 'rest')

    features = ['speed_mm', 'rest']
    for feature in features:
        for species_name in species:
            # correlations for individuals across daily average
            fish_daily_ave_feature = species_feature_fish_daily_ave(fish_tracks_ds, species_name, feature)
            fish_daily_corr(fish_daily_ave_feature, feature, species_name)

        # correlations for individuals across week
        _ = fish_weekly_corr(fish_tracks_ds, feature, 'single')

    # correlations for species
    species_daily_corr(aves_ave_spd, 'speed_mm', 'single')
    species_daily_corr(aves_ave_rest, 'rest', 'single')

    # ### Define diel pattern ###
    fish_tracks_ds = fish_tracks_add_day_twilight_night(fish_tracks_ds)
    fish_tracks_ds = add_day_number_fish_tracks(fish_tracks_ds)
    fish_diel_patterns = diel_pattern_ttest_individ_ds(fish_tracks_ds, feature='rest')

    # add back 'species'
    fishes = fish_tracks_ds.FishID.unique()
    fish_diel_patterns['species_six'] = 'blank'
    for fish in fishes:
        fish_diel_patterns.loc[fish_diel_patterns['FishID'] == fish, 'species_six'] = six_letter_sp_name(extract_meta(fish)['species'])

    # define species diel pattern
    states = ['nocturnal', 'diurnal', ]
    fish_diel_patterns['species_diel_pattern'] = 'undefined'
    for species_name in species_sixes:
        for state in states:
            if ((fish_diel_patterns.loc[fish_diel_patterns.species_six == species_name, 'diel_pattern'] == state)*1).mean() > 0.5:
                fish_diel_patterns.loc[fish_diel_patterns.species_six == species_name, 'species_diel_pattern'] = state
        print("{} is {}".format(species_name, fish_diel_patterns.loc[fish_diel_patterns.species_six == species_name, 'species_diel_pattern'].unique()))

    # row colours
    row_cols = []
    subset = fish_diel_patterns.loc[:, ['species_six', 'species_diel_pattern']].drop_duplicates(subset = ["species_six"])
    for index, row in subset.iterrows():
        if row['species_diel_pattern'] == 'diurnal':
            # row_cols.append(sns.color_palette()[1])
            row_cols.append((255/255, 224/255, 179/255))
        elif row['species_diel_pattern'] == 'nocturnal':
            # row_cols.append(sns.color_palette()[0])
            row_cols.append((153/255, 204/255, 255/255))
        elif row['species_diel_pattern'] == 'undefined':
            # row_cols.append(sns.color_palette()[2])
            row_cols.append((179/255, 230/255, 179/255))

    clrs = [(sns.color_palette()[2]), sns.color_palette()[0], sns.color_palette()[1]]
    # 2 = green, 1 =  orange, 0 =  blue

    f, ax = plt.subplots(figsize=(5, 10))
    sns.boxplot(data=fish_diel_patterns, x='day_night_dif', y='species_six', palette=row_cols,  ax=ax, fliersize=0) #, color="gainsboro"
    sns.stripplot(data=fish_diel_patterns, x='day_night_dif', y='species_six', hue='diel_pattern', ax=ax, size=4, palette=clrs)
    ax.set(xlabel='Day mean - night mean', ylabel='Species')
    # ax.set(xlim=(-1, 1))
    ax = plt.axvline(0, ls='--', color='k')
    plt.tight_layout()

# better crepuscular
feature= 'speed_mm'
border_top = np.ones(48)
border_bottom = np.ones(48)*1.05
dawn_s, dawn_e, dusk_s, dusk_e = [6*2, 8*2, 18*2, 20*2]
border_bottom[6*2:8*2] = 0
border_bottom[18*2:20*2] = 0

peak_prom = 0.15
if feature == 'speed_mm':
    border_top_i = border_top * 200
    border_bottom_i = border_bottom * 200
    peak_prom = 7

# features = ['speed_mm', 'rest', 'movement']
# for feature in features:
#     first = True
for species_name in species:
    fish_daily_ave_feature = species_feature_fish_daily_ave(fish_tracks_ds, species_name, feature)

    if feature == 'rest':
        fish_daily_ave_feature = np.abs(fish_daily_ave_feature-1)

    fig1, ax1 = plt.subplots(figsize=(5, 5))
    sns.heatmap(fish_daily_ave_feature.T, cmap="Greys")
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.axvspan(dawn_s, dawn_e, color='wheat', alpha=0.5, linewidth=0)
    ax2.axvspan(dusk_s, dusk_e, color='wheat', alpha=0.5, linewidth=0)
    for i in np.arange(0, len(fish_daily_ave_feature.columns)):
        x = fish_daily_ave_feature.iloc[:, i]
        peaks, _ = find_peaks(x, distance=4, prominence=peak_prom, height=(border_bottom_i, border_top_i))
        ax2.plot(x)
        ax2.plot(peaks, x[peaks],  "o", color="r")
        ax1.plot(x.reset_index().index[peaks].values, (np.ones(len(peaks))*i)+0.5,   "o", color="r")
        plt.title(species_name)

border_bottom_week = np.concatenate((border_bottom, border_bottom, border_bottom, border_bottom, border_bottom,
                                border_bottom, border_bottom))
border_top_week = np.ones(48*7)

peak_prom = 0.15
if feature == 'speed_mm':
    border_bottom_week = border_bottom_week * 200
    border_top_week = border_top_week * 200
    peak_prom = 7

for species_name in species:
    fish_feature = fish_tracks_ds.loc[fish_tracks_ds.species == species_name, ['ts', 'FishID', feature]].pivot(
        columns='FishID', values=feature, index='ts')

    if feature == 'rest':
        fish_feature = np.abs(fish_daily_ave_feature-1)

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.heatmap(fish_feature.T, cmap="Greys")

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.set_ylabel(feature)
    for day in range(7):
        ax2.axvspan(dawn_s+day*48, dawn_e+day*48, color='wheat', alpha=0.5, linewidth=0)
        ax2.axvspan(dusk_s+day*48, dusk_e+day*48, color='wheat', alpha=0.5, linewidth=0)

    for i in np.arange(0, len(fish_feature.columns)):
        x = fish_feature.iloc[:, i]
        peaks, peak_prop = find_peaks(x, distance=4, prominence=peak_prom, height=(border_bottom_week[0:x.shape[0]],
                                                                              border_top_week[0:x.shape[0]]))

        np.around(peak_prop['peak_heights'], 2)

        ax2.plot(x)
        ax2.plot(peaks, x[peaks], "o", color="r")
        plt.title(species_name)

        ax1.plot(x.reset_index().index[peaks].values, (np.ones(len(peaks))*i)+0.5,   "o", color="r")


# need: peak height, peak location, dawn/dusk, max day/night for that day, if  peak missing, find most common peak,
# if all peaks missing use average of the whole  period
# location and use  the  value of that  bin. Find amplitude of peaks
border_top = np.ones(48)
border_bottom = np.ones(48)*1.05
dawn_border_bottom = copy.copy(border_bottom)
dawn_border_bottom[6*2:(8*2)+1] = 0
dusk_border_bottom = copy.copy(border_bottom)
dusk_border_bottom[18*2:(20*2)+1] = 0

border = np.zeros(48)
day_border = copy.copy(border)
day_border[8*2:18*2] = 1
night_border = copy.copy(border)
night_border[0:6*2] = 1
night_border[20*2:24*2] = 1

peak_prom = 0.15
if feature == 'speed_mm':
    border_top_i = border_top * 200
    dawn_border_bottom = dawn_border_bottom * 200
    dusk_border_bottom = dusk_border_bottom * 200
    peak_prom = 7


# check what happens when that crepuscaulr period is missing!!!!!
first_all = True
for species_name in species:
    fish_feature = fish_tracks_ds.loc[fish_tracks_ds.species == species_name, ['ts', 'FishID', feature]].pivot(
        columns='FishID', values=feature, index='ts')
    first = True
    for i in np.arange(0, len(fish_feature.columns)):
        epoques = np.arange(0, 48*7.5, 48).astype(int)
        fish_peaks_dawn = np.zeros([4, int(np.floor(fish_feature.iloc[:, i].reset_index().shape[0]/48))])
        fish_peaks_dusk = np.zeros([4, int(np.floor(fish_feature.iloc[:, i].reset_index().shape[0] / 48))])
        for j in np.arange(0, int(np.ceil(fish_feature.shape[0]/48))):
            x = fish_feature.iloc[epoques[j]:epoques[j+1], i]
            if x.size == 48:
                #dawn peak
                dawn_peak, dawn_peak_prop = find_peaks(x, distance=4, prominence=peak_prom, height=
                (dawn_border_bottom[0:x.shape[0]], border_top_i[0:x.shape[0]]))

                # duskpeak
                dusk_peak, dusk_peak_prop = find_peaks(x, distance=4, prominence=peak_prom, height=
                (dusk_border_bottom[0:x.shape[0]], border_top_i[0:x.shape[0]]))

                # fig = plt.figure(figsize=(10, 5))
                # plt.plot(x)

                if dawn_peak.size != 0:
                    fish_peaks_dawn[0, j] = dawn_peak[0]
                    fish_peaks_dawn[1, j] = dawn_peak[0] + epoques[j]
                    fish_peaks_dawn[2, j] = np.round(dawn_peak_prop['peak_heights'][0], 2)
                    # plt.plot(dawn_peak[0], x[int(dawn_peak[0])], "o", color="r")

                if dusk_peak.size != 0:
                    fish_peaks_dusk[0, j] = dusk_peak[0]
                    fish_peaks_dusk[1, j] = dusk_peak[0] + epoques[j]
                    fish_peaks_dusk[2, j] = np.round(dusk_peak_prop['peak_heights'][0], 2)
                    # plt.plot(dusk_peak[0], x[int(dusk_peak[0])], "o", color="r")

                # plt.plot(dawn_border_bottom)
                # plt.plot(dusk_border_bottom)

                # day mean
                day_mean = np.round(x[(day_border).astype(int) == 1].mean(), 2)
                # night mean
                night_mean = np.round(x[(night_border).astype(int) == 1].mean(), 2)

                fish_peaks_dawn[3, j] = fish_peaks_dawn[2, j] - np.max([day_mean, night_mean])
                fish_peaks_dusk[3, j] = fish_peaks_dusk[2, j] - np.max([day_mean, night_mean])

                # keep track  of if day or night max
        fish_peaks_dawn = replace_crep_peaks(fish_peaks_dawn, fish_feature, i, epoques)
        fish_peaks_dusk = replace_crep_peaks(fish_peaks_dusk, fish_feature, i, epoques)

        fish_peaks_df_dawn = make_fish_peaks_df(fish_peaks_dawn, fish_feature.columns[i])
        fish_peaks_df_dusk = make_fish_peaks_df(fish_peaks_dusk, fish_feature.columns[i])

        fish_peaks_df_dawn['twilight'] = 'dawn'
        fish_peaks_df_dusk['twilight'] = 'dusk'
        fish_peaks_df = pd.concat([fish_peaks_df_dawn, fish_peaks_df_dusk], axis=0)

        if first:
            species_peaks_df = fish_peaks_df
            first = False
        else:
            species_peaks_df = pd.concat([species_peaks_df, fish_peaks_df], axis=0)
    species_peaks_df['species'] = species_name
    if first_all:
        all_peaks_df = species_peaks_df
        first_all = False
    else:
        all_peaks_df = pd.concat([all_peaks_df, species_peaks_df], axis=0)

all_peaks_df = all_peaks_df.reset_index(drop=True)
all_peaks_df['peak'] = (all_peaks_df.peak_loc !=0)*1

fig = plt.figure(figsize=(10, 5))
sns.swarmplot(x='species', y='peak_amplitude', data=all_peaks_df, hue='peak')


# average for each fish for dawn and dusk for 'peak_amplitude', peaks/(peaks+nonpeaks)
periods = ['dawn', 'dusk']
first_all =  True
for species_name in species:
    first  = True
    for period in periods:
        feature_i = all_peaks_df[(all_peaks_df['species']==species_name) & (all_peaks_df['twilight']==period)]
        [['peak_amplitude', 'FishID', 'crep_num', 'peak']]
        sp_average_peak_amp = feature_i.groupby('FishID').mean().peak_amplitude.reset_index()
        sp_average_peak = feature_i.groupby('FishID').mean().peak.reset_index()
        sp_average_peak_data = pd.concat([sp_average_peak_amp, sp_average_peak], axis=1)
        sp_average_peak_data['twilight'] = period
        if first:
            sp_feature_combined = sp_average_peak_data
            first = False
        else:
            sp_feature_combined = pd.concat([sp_feature_combined, sp_average_peak_data], axis=0)
    sp_feature_combined['species'] = species_name

    if first_all:
        all_feature_combined = sp_feature_combined
        first_all = False
    else:
        all_feature_combined = pd.concat([all_feature_combined, sp_feature_combined], axis=0)
all_feature_combined = all_feature_combined.reset_index(drop=True)
all_feature_combined = all_feature_combined.loc[:, ~all_feature_combined.columns.duplicated()]

all_feature_combined['species_six'] = 'blank'
for fish in fishes:
    all_feature_combined.loc[all_feature_combined['FishID'] == fish, 'species_six'] = six_letter_sp_name(extract_meta(fish)['species'])[0]

fig = plt.figure(figsize=(10, 5))
sns.swarmplot(x='species', y='peak_amplitude', data=all_feature_combined, hue='twilight', dodge=True)

g = sns.catplot(x='species_six', y='peak_amplitude', data=all_feature_combined.loc[all_feature_combined.species_six == 'Neosav'],
                hue='peak',  palette='vlag', col="twilight", legend=False)
for axes in g.axes.flat:
    _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=90)
plt.tight_layout()

sorted_index = all_feature_combined.groupby(by='species_six').mean().sort_values(by='peak_amplitude').index
grped_bplot = sns.catplot(y='species_six',
                          x='peak_amplitude',
                          hue="twilight",
                          kind="box",
                          legend=False,
                          height=6,
                          aspect=1.3,
                          data=all_feature_combined,
                          fliersize=0,
                          boxprops=dict(alpha=.3),
                          order=sorted_index)
plt.axvline(0, color='k', linestyle='--')
grped_bplot = sns.stripplot(y='species_six',
                            x='peak_amplitude',
                            hue='twilight',
                            jitter=True,
                            dodge=True,
                            marker='o',
                            data=all_feature_combined,
                            order=sorted_index)


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

