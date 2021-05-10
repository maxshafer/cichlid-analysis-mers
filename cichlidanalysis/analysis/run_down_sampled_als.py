from tkinter.filedialog import askdirectory, askopenfilename
from tkinter import *
import warnings
import os
import time

import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch

from cichlidanalysis.io.tracks import load_ds_als_files
from cichlidanalysis.utils.timings import load_timings
from cichlidanalysis.utils.species_names import shorten_sp_name, six_letter_sp_name
from cichlidanalysis.utils.species_metrics import add_metrics, tribe_cols
from cichlidanalysis.plotting.speed_plots import plot_spd_30min_combined
from cichlidanalysis.analysis.processing import feature_daily

# debug pycharm problem
warnings.simplefilter(action='ignore', category=FutureWarning)


def fish_corr(fish_tracks_ds, feature, link_method):
    species = fish_tracks_ds['species'].unique()
    first = True

    for species_i in species:
        print(species_i)
        fish_tracks_ds_sp = fish_tracks_ds.loc[fish_tracks_ds.species == species_i, ['FishID', 'ts', feature]]
        fish_tracks_ds_sp = fish_tracks_ds_sp.pivot(columns='FishID', values=feature, index='ts')
        individ_corr = fish_tracks_ds_sp.corr()

        mask = np.ones(individ_corr.shape, dtype='bool')
        mask[np.triu_indices(len(individ_corr))] = False
        corr_val_f = individ_corr.values[mask]

        if first:
            corr_vals = pd.DataFrame(corr_val_f, columns=[six_letter_sp_name(species_i)[0]])
            first = False
        else:
            corr_vals = pd.concat([corr_vals, pd.DataFrame(corr_val_f, columns=[six_letter_sp_name(species_i)[0]])], axis=1)

        X = individ_corr.values
        d = sch.distance.pdist(X)
        L = sch.linkage(d, method=link_method)
        ind = sch.fcluster(L, 0.5*d.max(), 'distance')
        cols = [individ_corr.columns.tolist()[i] for i in list((np.argsort(ind)))]
        individ_corr = individ_corr[cols]
        individ_corr = individ_corr.reindex(cols)

        fish_sex = fish_tracks_ds.loc[fish_tracks_ds.species == species_i, ['FishID', 'sex']].drop_duplicates()
        fish_sex = list(fish_sex.sex)
        fish_sex_clus = [fish_sex[i] for i in list((np.argsort(ind)))]

        f, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(data=individ_corr, vmin=-1, vmax=1, xticklabels=fish_sex_clus, yticklabels=fish_sex_clus,
                         cmap='seismic', ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(rootdir, "{0}_corr_by_30min_{1}_{2}_{3}.png".format(species_i.replace(' ', '-'),
                                                                                     feature, dt.date.today(),
                                                                                     link_method)))
        plt.close()

    corr_vals_long = pd.melt(corr_vals, var_name='species_six', value_name='corr_coef')

    f, ax = plt.subplots(figsize=(3, 5))
    sns.boxplot(data=corr_vals_long, x='corr_coef', y='species_six', ax=ax, fliersize=0)
    sns.stripplot(data=corr_vals_long, x='corr_coef', y='species_six', color=".2", ax=ax, size=3)
    ax.set(xlabel='Correlation', ylabel='Species')
    ax.set(xlim=(-1, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "fish_corr_coefs_{0}_{1}.png".format(feature,  dt.date.today())))
    plt.close()

    return corr_vals


def species_corr(averages_feature, feature, link_method):
    """ Plots corr matrix of clustered species by given feature

    :param averages_feature:
    :param feature:
    :return:
    """

    individ_corr = averages_feature.corr()

    X = individ_corr.values
    d = sch.distance.pdist(X)  # vector of ('55' choose 2) pairwise distances
    L = sch.linkage(d, method=link_method)
    Z = sch.dendrogram(L, orientation='right')
    ind = Z['leaves']

    individ_corr = individ_corr.to_numpy()
    individ_corr = individ_corr[ind, :]
    individ_corr = individ_corr[:, ind]

    # ind = sch.fcluster(L, 0.5 * d.max(), 'distance')
    cols = [individ_corr.columns.tolist()[i] for i in list((np.argsort(ind)))]
    individ_corr = individ_corr[cols]
    individ_corr = individ_corr.reindex(cols)

    #add back col names

    f, ax = plt.subplots(figsize=(7, 5))
    ax = sns.heatmap(individ_corr, vmin=-1, vmax=1, cmap='RdBu_r')
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "species_corr_by_30min_{0}_{1}_{2}.png".format(feature, dt.date.today(), link_method)))
    plt.close()


def week_corr(fish_tracks_ds, feature):
    """ Plots corr matrix of clustered species by given feature

    :param averages_feature:
    :param feature:
    :return:
    """
    species = fish_tracks_ds['species'].unique()

    for species_i in species:

        fishes = fish_tracks_ds.loc[fish_tracks_ds.species == species_i, 'FishID'].unique()
        first = True

        for fish in fishes:
            print(fish)
            fish_tracks_ds_day = fish_tracks_ds.loc[fish_tracks_ds.FishID == fish, ['day', 'time_of_day_dt', feature]]
            fish_tracks_ds_day = fish_tracks_ds_day.pivot(columns='day', values=feature, index='time_of_day_dt')
            individ_corr = fish_tracks_ds_day.corr()

            mask = np.ones(individ_corr.shape, dtype='bool')
            mask[np.triu_indices(len(individ_corr))] = False
            corr_val_f = individ_corr.values[mask]
            if first:
                corr_vals = pd.DataFrame(corr_val_f, columns=[fish])
                first = False
            else:
                corr_vals = pd.concat([corr_vals, pd.DataFrame(corr_val_f, columns=[fish])], axis=1)

            X = individ_corr.values
            d = sch.distance.pdist(X)
            L = sch.linkage(d, method='single')
            ind = sch.fcluster(L, 0.5*d.max(), 'distance')
            cols = [individ_corr.columns.tolist()[i] for i in list((np.argsort(ind)))]
            individ_corr = individ_corr[cols]
            individ_corr = individ_corr.reindex(cols)

            f, ax = plt.subplots(figsize=(7, 5))
            ax = sns.heatmap(individ_corr, vmin=-1, vmax=1, cmap='bwr')
            ax.set_title(fish)
            plt.tight_layout()
            plt.savefig(os.path.join(rootdir, "species_corr_by_30min_{0}_{1}.png".format(feature, dt.date.today())))
            plt.close()

        f, ax = plt.subplots(figsize=(7, 5))
        ax = sns.catplot(data=corr_vals, kind="swarm", vmin=-1, vmax=1)
        ax.set_title(species_i)


def add_day(fish_df):
    """ Adds day number to the fish  dataframe, by using the timestamp (ts) column

    :param fish_df:
    :return:
    """
    # add new column with day number (starting from 1)
    fish_df['day'] = fish_df.ts.apply(lambda row: int(str(row)[8:10]) - 1)
    print("added night and day column")
    return fish_df


if __name__ == '__main__':
    # pick folder
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

    # extra data
    # root = Tk()
    # root.withdraw()
    # root.update()
    # metrics_path = askopenfilename(title="Select metrics file")
    # root.destroy()
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
    plt.close()
    feature, ymax, span_max, ylabeling = 'speed_mm', 95, 80, 'Speed mm/s'
    averages_spd, _, sp_spd_combined = plot_spd_30min_combined(fish_tracks_ds, feature, ymax, span_max,
                                                                              ylabeling, change_times_datetime, rootdir)
    plt.close()
    feature, ymax, span_max, ylabeling = 'rest', 1, 0.8, 'Rest'
    averages_rest, _, sp_rest_combined = plot_spd_30min_combined(fish_tracks_ds, feature, ymax, span_max,
                                                                              ylabeling, change_times_datetime, rootdir)
    plt.close()
    feature, ymax, span_max, ylabeling = 'movement', 1, 0.8, 'Movement'
    averages_move, _, sp_move_combined = plot_spd_30min_combined(fish_tracks_ds, feature, ymax, span_max,
                                                                              ylabeling, change_times_datetime, rootdir)
    plt.close()

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

    ax = sns.clustermap(aves_ave_spd.T, figsize=(7, 5), col_cluster=False, method='single', metric='correlation')
    ax.fig.suptitle("Speed mm/s")
    plt.close()
    ax = sns.clustermap(aves_ave_vp.T.reset_index(drop=True), figsize=(7, 5), col_cluster=False, method='single', metric='correlation', row_colors=row_cols_1)
    ax.fig.suptitle("Vertical position")
    plt.close()
    ax = sns.clustermap(aves_ave_rest.T.reset_index(drop=True), figsize=(7, 5), col_cluster=False, method='single', metric='correlation', row_colors=row_cols_1)
    ax.fig.suptitle("Rest")
    plt.close()
    ax = sns.clustermap(aves_ave_move.T.reset_index(drop=True), figsize=(7, 5), col_cluster=False, method='single', metric='correlation', row_colors=row_cols_1)
    ax.fig.suptitle("Movement")
    plt.close()

# ## correlations ##
    fish_tracks_ds = add_day(fish_tracks_ds)
    # correlations for days across week for an individual
    # week_corr(fish_tracks_ds, 'rest')

    # correlations for individuals
    _ = fish_corr(fish_tracks_ds, 'rest', 'ward')
    _ = fish_corr(fish_tracks_ds, 'speed_mm')

    # correlations for species
    species_corr(aves_ave_spd, 'speed_mm', 'single')
    species_corr(aves_ave_rest, 'rest')

    # # https://matplotlib.org/matplotblog/posts/create-ridgeplots-in-matplotlib/
    # cmap = cm.get_cmap('turbo')
    # colour_array = np.arange(0, 1, 1 / len(species))
    #
    # gs = grid_spec.GridSpec(len(species), 1)
    # fig = plt.figure(figsize=(16, 9))
    # # date_form = DateFormatter('%Y-%m-%d')
    # date_form = DateFormatter('%H:%M:%S')
    # ax_objs = []
    # averages = np.zeros([len(species), 303])
    #
    # first = 1
    # for species_n, species_name in enumerate(species):
    #     # get speeds for each individual for a given species
    #     spd = fish_tracks_ds[fish_tracks_ds.species == species_name][[feature, 'FishID', 'ts']]
    #     sp_spd = spd.pivot(columns='FishID', values=feature, index='ts')
    #     if first:
    #         sp_spd_combined = sp_spd
    #         first = 0
    #     else:
    #         frames = [sp_spd_combined, sp_spd]
    #         sp_spd_combined = pd.concat(frames, axis=1)
    #
    #     # calculate ave and stdv
    #     average = sp_spd.mean(axis=1)
    #     averages[species_n, :] = average[0:303]
    #     # stdv = sp_spd.std(axis=1)
    #
    #     # create time vector in datetime format
    #     # tv = fish_tracks_ds.loc[fish_tracks_ds.FishID == fish_IDs[0], 'ts']
    #     date_time_obj = []
    #     for i in sp_spd.index:
    #         date_time_obj.append(dt.datetime.strptime(i, '%Y-%m-%d %H:%M:%S'))
    #
    #     # creating new axes object
    #     ax_objs.append(fig.add_subplot(gs[species_n:species_n + 1, 0:]))
    #
    #     days_to_plot = (date_time_obj[-1] - date_time_obj[0]).days + 1
    #
    #     for day_n in range(days_to_plot):
    #         # ax_objs[-1].axvspan(dt.datetime.strptime("1970-1-2 00:00:00", '%Y-%m-%d %H:%M:%S')+timedelta(days=day_n),
    #         #                     change_times_datetime[0]+timedelta(days=day_n), color='lightblue',
    #         #                     alpha=0.5, linewidth=0)
    #         # ax_objs[-1].axvspan(change_times_datetime[0]+timedelta(days=day_n),
    #         #                     change_times_datetime[1]+timedelta(days=day_n),  color='wheat',
    #         #                     alpha=0.5, linewidth=0)
    #         # ax_objs[-1].axvspan(change_times_datetime[2]+timedelta(days=day_n), change_times_datetime[3]+timedelta
    #         # (days=day_n), color='wheat', alpha=0.5, linewidth=0)
    #         # ax_objs[-1].axvspan(change_times_datetime[3]+timedelta(days=day_n), change_times_datetime[4]+timedelta
    #         # (days=day_n), color='lightblue', alpha=0.5, linewidth=0)
    #
    #         ax_objs[-1].fill_between([dt.datetime.strptime("1970-1-2 00:00:00", '%Y-%m-%d %H:%M:%S')+timedelta(days=day_n),
    #                                   change_times_datetime[0]+timedelta(days=day_n)], [span_max, span_max], 0,
    #                                  color='lightblue', alpha=0.5, linewidth=0, zorder=1)
    #         ax_objs[-1].fill_between([change_times_datetime[0]+timedelta(days=day_n),
    #                             change_times_datetime[1]+timedelta(days=day_n)], [span_max, span_max], 0,  color='wheat',
    #                             alpha=0.5, linewidth=0)
    #         ax_objs[-1].fill_between([change_times_datetime[2]+timedelta(days=day_n), change_times_datetime[3]+timedelta
    #         (days=day_n)], [span_max, span_max], 0, color='wheat', alpha=0.5, linewidth=0)
    #         ax_objs[-1].fill_between([change_times_datetime[3]+timedelta(days=day_n), change_times_datetime[4]+timedelta
    #         (days=day_n)], [span_max, span_max], 0, color='lightblue', alpha=0.5, linewidth=0)
    #
    #     # plotting the distribution
    #     ax_objs[-1].plot(date_time_obj, average, lw=1, color='w')
    #     ax_objs[-1].fill_between(date_time_obj, average, 0, color=cmap(colour_array[species_n]), zorder=2)
    #
    #     # setting uniform x and y lims
    #     ax_objs[-1].set_xlim(min(date_time_obj), dt.datetime.strptime("1970-1-8 08:30:00", '%Y-%m-%d %H:%M:%S'))
    #     ax_objs[-1].set_ylim(0, ymax)
    #
    #     # make background transparent
    #     rect = ax_objs[-1].patch
    #     rect.set_alpha(0)
    #
    #     if species_n == len(species) - 1:
    #         ax_objs[-1].set_xlabel("Time", fontsize=10, fontweight="bold")
    #         ax_objs[-1].xaxis.set_major_locator(MultipleLocator(20))
    #         ax_objs[-1].xaxis.set_major_formatter(date_form)
    #         ax_objs[-1].yaxis.tick_right()
    #         ax_objs[-1].yaxis.set_label_position("right")
    #         ax_objs[-1].set_ylabel(ylabeling)
    #
    #     else:
    #         # remove borders, axis ticks, and labels
    #         ax_objs[-1].set_xticklabels([])
    #         ax_objs[-1].set_xticks([])
    #         ax_objs[-1].set_yticks([])
    #         ax_objs[-1].set_yticklabels([])
    #         ax_objs[-1].set_ylabel('')
    #
    #     spines = ["top", "right", "left", "bottom"]
    #     for s in spines:
    #         ax_objs[-1].spines[s].set_visible(False)
    #
    #     short_name = shorten_sp_name(species_name)
    #     shortened_sp_name = species_name[0] + ". " + species_name.split(' ')[1]
    #     ax_objs[-1].text(0.9, 0, short_name[0], fontweight="bold", fontsize=10, ha="right", rotation=-45)
    #     gs.update(hspace=-0.1)
    # plt.show()

    # # clustering
    # X = aves_ave.corr().values
    # d = sch.distance.pdist(X)   # vector of ('55' choose 2) pairwise distances
    # L = sch.linkage(d, method='complete')
    # ind = sch.fcluster(L, 0.5*d.max(), 'distance')
    # cols = [aves_ave.columns.tolist()[i] for i in list((np.argsort(ind)))]
    # aves_ave = aves_ave[cols]
    #
    # ## heatmap average
    # changes = [7, 19]
    # changes_ticks = []
    # for i in np.arange(0, 6):
    #     changes_ticks = np.append(changes_ticks, np.add(changes, 24*i))
    # changes_ticks = np.multiply(changes_ticks, 2)
    #
    # fig1, ax1 = plt.subplots()
    # fig1.set_figheight(5)
    # fig1.set_figwidth(15)
    # im_spd = ax1.imshow(averages, aspect='auto', vmin=0, vmax=70)
    # ax1.get_yaxis().set_ticks(np.arange(0, len(species)))
    # ax1.get_yaxis().set_ticklabels(species_short, rotation=45)
    # ax1.get_xaxis().set_ticks(np.arange(0, averages.shape[1], 12))
    # ax1.get_xaxis().set_ticks(changes_ticks)
    # ax1.get_xaxis().set_ticklabels(['7am', '7pm']*6)
    # cbar = fig1.colorbar(im_spd, label="Speed mm/s")
    # fig1.tight_layout(pad=2)
    #
    # ## heatmap daily average
    # fig1, ax1 = plt.subplots()
    # fig1.set_figheight(6)
    # fig1.set_figwidth(6)
    # im_spd = ax1.imshow(aves_ave.T, aspect='auto', vmin=0, vmax=50, cmap='magma')
    # ax1.get_yaxis().set_ticks(np.arange(0, len(species)))
    # sp_names = shorten_sp_name(aves_ave.columns)
    # ax1.get_yaxis().set_ticklabels(sp_names, rotation=45)
    # ax1.get_xaxis().set_ticks(np.arange(0, aves_ave.shape[1], 12))
    # ax1.get_xaxis().set_ticks(changes_ticks[0:2])
    # ax1.get_xaxis().set_ticklabels(['7am', '7pm'])
    # cbar = fig1.colorbar(im_spd, label="Speed mm/s")
    # plt.title('Daily average speed mm/s')
    # fig1.tight_layout(pad=3)
    #
    # ## heatmap individuals
    # changes = [7, 19]
    # changes_ticks = []
    # for i in np.arange(0, 6):
    #     changes_ticks = np.append(changes_ticks, np.add(changes, 24*i))
    # changes_ticks = np.multiply(changes_ticks, 2)
    #
    # fig1, ax1 = plt.subplots()
    # fig1.set_figheight(5)
    # fig1.set_figwidth(15)
    # im_spd = ax1.imshow(sp_spd_combined.T, aspect='auto', vmin=0, vmax=70)
    # ax1.get_yaxis().set_ticks(np.arange(0, len(fish_IDs)))
    # ax1.get_yaxis().set_ticklabels(sp_spd_combined.columns, rotation=45)
    # ax1.get_xaxis().set_ticks(np.arange(0, sp_spd_combined.shape[1], 12))
    # ax1.get_xaxis().set_ticks(changes_ticks)
    # ax1.get_xaxis().set_ticklabels(['7am', '7pm']*6)
    # cbar = fig1.colorbar(im_spd, label="Speed mm/s")
    # fig1.tight_layout(pad=3)
    #
