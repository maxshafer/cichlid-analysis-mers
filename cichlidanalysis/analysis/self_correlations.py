import os

import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch

from cichlidanalysis.utils.species_names import six_letter_sp_name


def fish_weekly_corr(rootdir, fish_tracks_ds, feature, link_method):
    """

    :param fish_tracks_ds:
    :param feature:
    :param link_method:
    :return:
    """
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
    ax = plt.axvline(0, ls='--', color='k')
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "fish_corr_coefs_{0}_{1}.png".format(feature,  dt.date.today())))
    plt.close()

    return corr_vals


def fish_daily_corr(averages_feature, feature, species_name, rootdir, link_method='single'):
    """ Plots corr matrix of clustered species by given feature

    :param averages_feature:
    :param feature:
    :param species_name:
    :param link_method:
    :return:
    """

    individ_corr = averages_feature.corr()

    ax = sns.clustermap(individ_corr, figsize=(7, 5), method=link_method, metric='euclidean', vmin=-1, vmax=1,
                        cmap='RdBu_r', xticklabels=False, yticklabels=False)
    ax.fig.suptitle(feature)
    plt.savefig(os.path.join(rootdir, "fish_of_{0}_corr_by_30min_{1}_{2}_{3}.png".format(species_name, feature, dt.date.today(), link_method)))
    plt.close()



def species_daily_corr(rootdir, averages_feature, feature, link_method='single'):
    """ Plots corr matrix of clustered species by given feature

    :param averages_feature:
    :param feature:
    :return:
    """

    individ_corr = averages_feature.corr()

    ax = sns.clustermap(individ_corr, figsize=(8, 7), method=link_method, metric='euclidean', vmin=-1, vmax=1,
                        cmap='viridis', yticklabels=True, xticklabels=True)
    ax.fig.suptitle(feature)
    plt.savefig(os.path.join(rootdir, "species_corr_by_30min_{0}_{1}_{2}.png".format(feature, dt.date.today(), link_method)))
    plt.close()


def week_corr(rootdir, fish_tracks_ds, feature):
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

