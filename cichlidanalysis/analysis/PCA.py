from tkinter.filedialog import askdirectory
from tkinter import *
import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm

from cichlidanalysis.io.als_files import load_ds_als_files
from cichlidanalysis.io.io_ecological_measures import get_meta_paths
from cichlidanalysis.utils.timings import load_timings
from cichlidanalysis.utils.species_metrics import tribe_cols
from cichlidanalysis.analysis.processing import feature_daily, species_feature_fish_daily_ave, \
    fish_tracks_add_day_twilight_night, add_day_number_fish_tracks
from cichlidanalysis.analysis.diel_pattern import diel_pattern_stats_individ_bin, diel_pattern_stats_species_bin
from cichlidanalysis.analysis.self_correlations import species_daily_corr, fish_daily_corr, fish_weekly_corr
from cichlidanalysis.analysis.crepuscular_pattern import crepuscular_peaks
from cichlidanalysis.plotting.cluster_plots import cluster_all_fish, cluster_species_daily
from cichlidanalysis.plotting.plot_diel_patterns import plot_day_night_species, plot_cre_dawn_dusk_strip_box, \
    plot_day_night_species_ave
from cichlidanalysis.plotting.speed_plots import plot_ridge_plots
from cichlidanalysis.analysis.run_binned_als import setup_run_binned

# insipired by https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

def run_pca(fish_tracks_bin, sp_metrics, tribe_col, species_full, fish_IDs, species_sixes):

    # Standardizing the features
    n_com = 2
    x = StandardScaler().fit_transform(aves_ave_spd.values)
    mu = np.mean(aves_ave_spd, axis=0)

    pca = PCA(n_components=n_com)
    principalComponents = pca.fit_transform(x)
    labels = []
    for i in range(n_com):
        labels.append('pc{}'.format(i+1))
    principalDf = pd.DataFrame(data=principalComponents, columns=labels)
    finalDf = pd.concat([principalDf, aves_ave_spd.index.to_series().reset_index(drop=True)], axis=1)
    day = set(np.where(aves_ave_spd.index.to_series().reset_index(drop=True) <'19:00')[0]) & set(np.where(aves_ave_spd.index.to_series().reset_index(drop=True) >= '07:00')[0])
    six_thirty_am = set(np.where(aves_ave_spd.index.to_series().reset_index(drop=True) == '06:30')[0])
    seven_am = set(np.where(aves_ave_spd.index.to_series().reset_index(drop=True) == '07:00')[0])
    seven_pm = set(np.where(aves_ave_spd.index.to_series().reset_index(drop=True) == '19:00')[0])
    six_thirty_pm = set(np.where(aves_ave_spd.index.to_series().reset_index(drop=True) == '18:30')[0])
    finalDf['daynight'] = 'night'
    finalDf.loc[day, 'daynight'] = 'day'
    finalDf.loc[six_thirty_am, 'daynight'] = 'six_thirty_am'
    finalDf.loc[six_thirty_pm, 'daynight'] = 'six_thirty_pm'
    finalDf.loc[seven_am, 'daynight'] = 'seven_am'
    finalDf.loc[seven_pm, 'daynight'] = 'seven_pm'

    # reconstructing the fish series
    # https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com
    Xhat = np.dot(pca.transform(aves_ave_spd)[:, :n_com], pca.components_[:n_com, :])
    Xhat += mu
    reconstructed = pd.DataFrame(data=Xhat, columns=aves_ave_spd.columns)
    f, ax = plt.subplots(figsize=(10, 5))
    plt.plot(reconstructed)

    # reconstruct with only pc1
    Xhat = np.dot(pca.transform(aves_ave_spd)[:, :1], pca.components_[:1, :])
    Xhat += mu
    reconstructed = pd.DataFrame(data=Xhat, columns=aves_ave_spd.columns)
    night_minus_day = reconstructed.loc[10, :] - reconstructed.loc[25, :]
    f, ax = plt.subplots(figsize=(10, 5))
    plt.scatter(night_minus_day.sort_values().index, night_minus_day.sort_values())
    ax.set_xticklabels(ax.get_xticks(), rotation=90)
    ax.set_xticklabels(night_minus_day.sort_values().index.to_list())

    # reconstruct with only pc2
    Xhat = np.dot(pca.transform(aves_ave_spd)[:, 1:2], pca.components_[1:2, :])
    Xhat += mu
    reconstructed = pd.DataFrame(data=Xhat, columns=aves_ave_spd.columns)
    night_minus_day = reconstructed.loc[14, :] - reconstructed.loc[25, :]
    f, ax = plt.subplots(figsize=(10, 5))
    plt.scatter(night_minus_day.sort_values().index, night_minus_day.sort_values())
    ax.set_xticklabels(ax.get_xticks(), rotation=90)
    ax.set_xticklabels(night_minus_day.sort_values().index.to_list())


    f, ax = plt.subplots(figsize=(10, 5))
    plt.plot(principalDf)
    f, ax = plt.subplots(figsize=(10, 5))
    plt.plot(np.cumsum(pca.explained_variance_))

    f, ax = plt.subplots(figsize=(10, 5))
    plt.scatter(principalDf.loc[:, 'pc1'], principalDf.loc[:, 'pc2'])

    cmap = matplotlib.cm.get_cmap('twilight_shifted')

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    times = finalDf.daynight
    timepoints = times.unique()
    # colors = ['r', 'g', 'b']
    for time_n, time in enumerate(timepoints):
        indicesToKeep = finalDf['daynight'] == time
        ax.scatter(finalDf.loc[indicesToKeep, 'pc1'], finalDf.loc[indicesToKeep, 'pc2'], c=cmap(time_n/len(timepoints)), s=50)
    ax.legend(timepoints)
    ax.grid()


if __name__ == '__main__':
    # Allows user to select top directory and load all als files here
    root = Tk()
    rootdir = askdirectory(parent=root)
    root.destroy()

    fish_tracks_bin, sp_metrics, tribe_col, species_full, fish_IDs, species_sixes = setup_run_binned(rootdir)
    # get timings
    fps, tv_ns, tv_sec, tv_24h_sec, num_days, tv_s_type, change_times_s, change_times_ns, change_times_h, \
    day_ns, day_s, change_times_d, change_times_m, change_times_datetime, change_times_unit\
        = load_timings(fish_tracks_bin[fish_tracks_bin.FishID == fish_IDs[0]].shape[0])

    averages_vp, date_time_obj_vp, sp_vp_combined, averages_spd, sp_spd_combined, averages_rest, sp_rest_combined, \
    averages_move, sp_move_combined = plot_ridge_plots(fish_tracks_bin, change_times_datetime,
                                                       rootdir, sp_metrics, tribe_col)

    # ### generate averages of the averages ###
    aves_ave_spd = feature_daily(averages_spd)
    aves_ave_rest = feature_daily(averages_rest)
    aves_ave_move = feature_daily(averages_move)

    aves_ave_spd.columns = species_sixes

    run_pca(fish_tracks_bin, sp_metrics, tribe_col, species_full, fish_IDs, species_sixes)
