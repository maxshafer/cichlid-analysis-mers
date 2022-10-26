from tkinter.filedialog import askdirectory
from tkinter import *
import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm

from cichlidanalysis.utils.timings import load_timings
from cichlidanalysis.analysis.processing import feature_daily
from cichlidanalysis.plotting.speed_plots import plot_ridge_plots
from cichlidanalysis.analysis.run_binned_als import setup_run_binned
from cichlidanalysis.io.io_feature_vector import load_diel_pattern
from cichlidanalysis.analysis.linear_regression import run_linear_reg, plt_lin_reg

# insipired by https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60


def plot_loadings(rootdir, pca, labels, data_input):
    loadings = pd.DataFrame(pca.components_.T, columns=labels, index=data_input.columns)

    for pc_name in labels:
        loadings_sorted = loadings.sort_values(by=pc_name)
        f, ax = plt.subplots(figsize=(10, 5))
        plt.scatter(loadings_sorted.index, loadings_sorted.loc[:, pc_name])
        ax.set_xticklabels(loadings_sorted.index, rotation=90)
        plt.title(pc_name)
        ax.set_ylabel('loading')
        sns.despine(top=True, right=True)
        plt.tight_layout()
    return loadings


def reconstruct_pc(data_input, pca, mu, pc_n):
    # reconstruct the data with only pc 'n'
    Xhat = np.dot(pca.transform(data_input)[:, pc_n-1:pc_n], pca.components_[:1, :])
    Xhat += mu
    reconstructed = pd.DataFrame(data=Xhat, columns=data_input.columns)
    f, ax = plt.subplots(figsize=(10, 5))
    plt.plot(reconstructed)

    f, ax = plt.subplots(figsize=(10, 5))
    plt.plot(reconstructed.iloc[:, 0:5:])

    # species = 'Astbur'
    # f, ax = plt.subplots(figsize=(10, 5))
    # plt.plot(reconstructed.loc[:, species])
    # plt.plot(data_input.loc[:, species])
    # plt.close('all')


def run_pca(rootdir, data_input):

    # data_input = sp_metrics_sub
    # data_input = aves_ave_spd
    # Standardizing the features
    n_com = 2
    x = StandardScaler().fit_transform(data_input.values)
    mu = np.mean(data_input, axis=0)

    # run PCA
    pca = PCA(n_components=n_com)
    principalComponents = pca.fit_transform(x)
    labels = []
    for i in range(n_com):
        labels.append('pc{}'.format(i+1))
    principalDf = pd.DataFrame(data=principalComponents, columns=labels)
    finalDf = pd.concat([principalDf, data_input.index.to_series().reset_index(drop=True)], axis=1)

    # reconstructing the fish series
    # https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com
    Xhat = np.dot(pca.transform(data_input)[:, :n_com], pca.components_[:n_com, :])
    Xhat += mu
    reconstructed = pd.DataFrame(data=Xhat, columns=data_input.columns)
    f, ax = plt.subplots(figsize=(10, 5))
    plt.plot(reconstructed)
    plt.savefig(os.path.join(rootdir, "reconstructed.png"), dpi=1000)
    plt.close()

    # plot reconstruction of pc 'n'
    reconstruct_pc(data_input, pca, mu, 1)

    # plot loadings of each pc
    loadings = plot_loadings(rootdir, pca, labels, data_input)
    f, ax = plt.subplots(figsize=(10, 5))
    plt.plot(reconstructed.loc[:, 'Astbur'])
    plt.savefig(os.path.join(rootdir, "reconstructed_Astbur.png"), dpi=1000)
    plt.close()

    # plot variance explained
    f, ax = plt.subplots(figsize=(10, 5))
    plt.plot(principalDf)
    plt.savefig(os.path.join(rootdir, "principalDf.png"), dpi=1000)
    plt.close()
    f, ax = plt.subplots(figsize=(10, 5))
    plt.plot(np.cumsum(pca.explained_variance_))
    plt.savefig(os.path.join(rootdir, "explained_variance_.png"), dpi=1000)
    plt.close()

    # plot 2D PC space with labeled points
    day = set(np.where(data_input.index.to_series().reset_index(drop=True) <'19:00')[0]) & set(np.where(aves_ave_spd.index.to_series().reset_index(drop=True) >= '07:00')[0])
    six_thirty_am = set(np.where(data_input.index.to_series().reset_index(drop=True) == '06:30')[0])
    seven_am = set(np.where(data_input.index.to_series().reset_index(drop=True) == '07:00')[0])
    seven_pm = set(np.where(data_input.index.to_series().reset_index(drop=True) == '19:00')[0])
    six_thirty_pm = set(np.where(data_input.index.to_series().reset_index(drop=True) == '18:30')[0])
    finalDf['daynight'] = 'night'
    finalDf.loc[day, 'daynight'] = 'day'
    finalDf.loc[six_thirty_am, 'daynight'] = 'six_thirty_am'
    finalDf.loc[six_thirty_pm, 'daynight'] = 'six_thirty_pm'
    finalDf.loc[seven_am, 'daynight'] = 'seven_am'
    finalDf.loc[seven_pm, 'daynight'] = 'seven_pm'

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
    plt.savefig(os.path.join(rootdir, "PCA.png"), dpi=1000)
    return pca, labels, loadings


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

    diel_patterns = load_diel_pattern(rootdir, suffix="*dp.csv")

    # ### generate averages of the averages ###
    aves_ave_spd = feature_daily(averages_spd)
    aves_ave_vp = feature_daily(averages_vp)
    aves_ave_rest = feature_daily(averages_rest)
    aves_ave_move = feature_daily(averages_move)

    aves_ave_spd.columns = species_sixes
    aves_ave_vp.columns = species_sixes
    aves_ave_rest.columns = species_sixes
    aves_ave_move.columns = species_sixes

    # reshape fish_tracks_bin by making FishID the rows,
    # Columns:
    #   speed_mm for each 30min bin
    #   rest for each 30min bin
    #   vertical position for each 30min bin
    #    'body_PC1', 'body_PC2',
    #        'LPJ_PC1', 'LPJ_PC2',  'oral_PC1', 'oral_PC2', 'd15N', 'd13C',
    #         'size_male', 'size_female', 'habitat', 'diet'
    #   'fish_length_mm'




    spd_df = aves_ave_spd.T.reset_index().rename(columns={'index': 'six_letter_name_Ronco'})

    sp_metrics_sub = sp_metrics.loc[:, ['six_letter_name_Ronco', 'habitat', 'diet']]

    for col in ['habitat', 'diet']:
        sp_metrics_sub[col].replace(sp_metrics_sub.loc[:, col].unique(), np.arange(len(sp_metrics_sub.loc[:, col].unique())), inplace=True)

    sp_metrics_sub = sp_metrics_sub.merge(spd_df, on="six_letter_name_Ronco")
    sp_metrics_sub = sp_metrics_sub.set_index("six_letter_name_Ronco")

    pca, labels, loadings = run_pca(rootdir, aves_ave_spd)

    loadings = loadings.reset_index().rename(columns={'index': 'species'})
    pca_df = loadings.merge(diel_patterns, on='species')
    model, r_sq = run_linear_reg(pca_df.pc1, pca_df.day_night_dif)
    plt_lin_reg(rootdir, pca_df.pc1, pca_df.day_night_dif, model, r_sq)

    model, r_sq = run_linear_reg(pca_df.pc2, pca_df.peak)
    plt_lin_reg(rootdir, pca_df.pc2, pca_df.peak, model, r_sq)
