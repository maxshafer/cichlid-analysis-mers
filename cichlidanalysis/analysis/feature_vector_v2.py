from tkinter.filedialog import askdirectory
from tkinter import *
import warnings
import os
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.spatial import ConvexHull

from cichlidanalysis.io.io_feature_vector import load_feature_vectors, load_diel_pattern
from cichlidanalysis.utils.species_names import six_letter_sp_name
from cichlidanalysis.io.meta import extract_meta
from cichlidanalysis.utils.species_metrics import add_metrics, tribe_cols
from cichlidanalysis.analysis.diel_pattern import daily_more_than_pattern_individ, daily_more_than_pattern_species, \
    day_night_ratio_individ, day_night_ratio_species
from cichlidanalysis.io.io_ecological_measures import get_meta_paths
from cichlidanalysis.plotting.figure_1 import cluster_dics

# debug pycharm problem
warnings.simplefilter(action='ignore', category=FutureWarning)


def run_linear_reg(x, y):
    """  Runs linear regression on x and y  varialbes. Will reshape x as needed (so  you  can add in pandas series)

    :param x: behavioural feature, pandas series
    :param y: ecological feature, pandas series
    :return: model and r_sq (R2 value for model fit)
    """
    x = x.to_numpy()
    if x.ndim == 1:
        x = np.reshape(x, (-1, 1))

    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    return model, r_sq


def plt_lin_reg(x, y, model, r_sq):
    """ Plot the scatter of two variables and add in the linear regression model, pearson's correlation and R2

    :param x: behavioural feature, pandas series
    :param y: ecological feature, pandas series
    :param model: run_linear_reg model
    :param r_sq: R2 value for model fit
    :return: saves plot
    """
    fig = plt.figure(figsize=(5, 5))
    ax = sns.scatterplot(x, y)
    # x_intercept = (model.intercept_/model.coef_)[0]*-1
    # plt.plot([0, x_intercept], [model.intercept_, 0], color='k')
    y_pred = model.predict(np.reshape(x.to_numpy(), (-1, 1)))
    plt.plot(x, y_pred, color='k')
    ax.set(xlim=(min(x), max(x)), ylim=(min(y), max(y)))

    r = np.corrcoef(x, y)
    ax.text(0.9, 0.94, s="$R^2$={}".format(np.round(r_sq, 2)), va='top', ha='center', transform=ax.transAxes)
    ax.text(0.9, 0.98, s="$r$={}".format(np.round(r[0, 1], 2)), va='top', ha='center', transform=ax.transAxes)
    fig.tight_layout()
    plt.savefig(os.path.join(rootdir, "feature_correlation_plot_{0}_vs_{1}_{2}.png".format(x.name, y.name,
                                                                                           datetime.date.today())))
    plt.close()

    # # residuals plot
    # fig = plt.figure(figsize=(5, 5))
    # ax = sns.scatterplot(x, y-y_pred)
    # ax.set_xlabel('Predicted')
    # ax.set_ylabel('Residuals')
    # plt.axhline(y=0, color='k')
    # fig.tight_layout()
    return


def subset_feature_plt(averages_i, features, labelling):
    """

    :param averages_i:
    :param features:
    :param labelling:
    :return:
    """
    # fig = plt.figure(figsize=(5, 10))
    fig = sns.clustermap(averages_i.T.loc[:, features], col_cluster=False,
                         yticklabels=True)  # , cbar_kws=dict(use_gridspec=False,location="top")
    plt.tight_layout(pad=2)
    plt.close()


if __name__ == '__main__':
    # pick folder
    # Allows user to select top directory and load all als files here
    root = Tk()
    rootdir = askdirectory(parent=root)
    root.destroy()

    feature_v = load_feature_vectors(rootdir, "*als_fv2.csv")
    diel_patterns = load_diel_pattern(rootdir, suffix="*dp.csv")
    ronco_data_path, cichlid_meta_path = get_meta_paths()
    ronco_data = pd.read_csv(ronco_data_path)
    ronco_data = ronco_data.replace({'Julmrk': 'Julmar', 'Pcynig': 'Parnig', 'Cphfro': 'Cypfro'})
    cichlid_meta = pd.read_csv(cichlid_meta_path)

    # add species and species_six
    feature_v['species'] = 'undefined'
    feature_v['species_six'] = 'undefined'
    for id_n, id in enumerate(feature_v.fish_ID):
        sp = extract_meta(id)['species']
        feature_v.loc[id_n, 'species'] = sp
        feature_v.loc[id_n, 'species_six'] = six_letter_sp_name(sp)

    # renaming misspelled species name
    feature_v = feature_v.replace('Aalcal', 'Altcal')
    feature_v = feature_v.replace('Aaltolamprologus-calvus', 'Altolamprologus-calvus')

    feature_v = feature_v.merge(cichlid_meta.drop(columns=['species', 'tribe']), on="species_six")
    feature_v = feature_v.merge(diel_patterns, on="species_six")
    feature_v = feature_v.drop({'species_x'}, axis=1)
    feature_v = feature_v.rename(columns={'species_y': 'species'})
    species = feature_v['species'].unique()

    tribe_col = tribe_cols()
    metrics_path = '/Users/annikanichols/Desktop/cichlid_species_database.xlsx'
    sp_metrics = add_metrics(feature_v.species_six.unique(), metrics_path)

    feature_v['tribe'] = 'undefined'
    for species_n in feature_v.species_six.unique():
        if sp_metrics.loc[sp_metrics['species_six'] == species_n, 'tribe'].empty:
            feature_v.loc[feature_v['species_six'] == species_n, 'tribe'] = 'undefined'
        else:
            feature_v.loc[feature_v['species_six'] == species_n, 'tribe'] = \
                sp_metrics.loc[sp_metrics['species_six'] == species_n, 'tribe'].values[0]

    # add column for cluster, hardcoded!!!!
    dic_complex, dic_simple, col_dic_simple, col_dic_complex, col_dic_simple, cluster_order = cluster_dics()
    # dic = {'diurnal': [7], 'nocturnal': [1], 'crepuscular': [2, 3, 5, 6, 8, 9], 'undefined': [4, 10, 11, 12]}
    # col_dic = {'diurnal': 'gold', 'nocturnal': 'royalblue', 'crepuscular': 'mediumorchid', 'undefined': 'black'}
    feature_v['cluster_pattern'] = 'placeholder'
    for key in dic_simple:
        # find the species which are in diel cluster group
        sp_diel_group = set(diel_patterns.loc[diel_patterns.cluster.isin(dic_simple[key]), 'species_six'].to_list())
        feature_v.loc[feature_v.species_six.isin(sp_diel_group), 'cluster_pattern'] = key

    # make species average
    for species_n, species_name in enumerate(species):
        # get speeds for each individual for a given species
        sp_subset = feature_v[feature_v.species == species_name]

        # calculate ave and stdv
        average = sp_subset.mean(axis=0)
        average = average.rename(six_letter_sp_name(species_name)[0])
        if species_n == 0:
            averages = average
        else:
            averages = pd.concat([averages, average], axis=1, join='inner')
        stdv = sp_subset.std(axis=0)

    averages_norm = averages.div(averages.sum(axis=1), axis=0)

    # ## heatmap of fv
    # fig1, ax1 = plt.subplots()
    # fig1.set_figheight(6)
    # fig1.set_figwidth(12)
    # im_spd = ax1.imshow(averages_norm.T, aspect='auto', vmin=0, cmap='magma')
    # ax1.get_yaxis().set_ticks(np.arange(0, len(species)))
    # ax1.get_yaxis().set_ticklabels(averages_norm.columns, rotation=0)
    # ax1.get_xaxis().set_ticks(np.arange(0, averages_norm.shape[0]))
    #
    # ax1.get_xaxis().set_ticklabels(averages_norm.index, rotation=90)
    # plt.title('Feature vector (normalised by feature)')
    # fig1.tight_layout(pad=3)

    # # clustered heatmap of  fv
    # fig = sns.clustermap(averages_norm, figsize=(20, 10), col_cluster=False, method='single', yticklabels=True)
    # plt.savefig(os.path.join(rootdir, "cluster_map_fv_{0}.png".format(datetime.date.today())))

    # # # total rest
    # # ax = sns.catplot(data=feature_v, y='species_six', x='total_rest', kind="swarm")
    # fig = plt.figure(figsize=(5, 12))
    # ax = sns.boxplot(data=feature_v, y='species_six', x='total_rest', hue='tribe')
    # ax = sns.swarmplot(data=feature_v, y='species_six', x='total_rest', color=".2")
    # ax.set(xlabel='Average total rest per day', ylabel='Species')
    # ax.set(xlim=(0, 24))
    # plt.tight_layout()
    # ax = plt.axvline(12, ls='--', color='k')
    # plt.savefig(os.path.join(rootdir, "total_rest_{0}.png".format(datetime.date.today())))

    ### summary statistics #####
    # N per species histogram
    fig = plt.figure(figsize=(5, 5))
    ax = feature_v["species"].value_counts(sort=False).plot.hist()
    ax.set_xlabel("Individuals for a species")
    ax.set_xlim([0, 14])
    plt.close()

    # number of species
    # feature_v["species"].value_counts().index

    # total rest ordered by mean, coloured by temporal guild
    colors = ['royalblue', 'mediumorchid', 'gold', 'silver']
    customPalette = sns.set_palette(sns.color_palette(colors))
    fig = plt.figure(figsize=(5, 10))
    ax = sns.boxplot(data=feature_v, y='species_six', x='total_rest', hue="cluster_pattern", dodge=False, showfliers=False,
                     order=feature_v.groupby('species_six').mean().sort_values("total_rest").index.to_list())
    ax = sns.swarmplot(data=feature_v, y='species_six', x='total_rest', color=".2", size=4,
                       order=feature_v.groupby('species_six').mean().sort_values("total_rest").index.to_list())
    ax.set(xlabel='Average total rest per day', ylabel='Species')
    ax.set(xlim=(0, 24))
    plt.tight_layout()
    ax = plt.axvline(12, ls='--', color='k')
    plt.savefig(os.path.join(rootdir, "total_rest_ordered_{0}.png".format(datetime.date.today())))
    plt.close()

    # total rest ordered by mean, coloured by diet guild
    colors = ['tomato', 'steelblue', 'sandybrown', 'mediumseagreen']
    customPalette = sns.set_palette(sns.color_palette(colors))
    fig = plt.figure(figsize=(5, 10))
    ax = sns.boxplot(data=feature_v, y='species_six', x='total_rest', hue="diet", dodge=False, showfliers=False,
                     order=feature_v.groupby('species_six').mean().sort_values("total_rest").index.to_list())
    ax = sns.swarmplot(data=feature_v, y='species_six', x='total_rest', color=".2", size=4,
                       order=feature_v.groupby('species_six').mean().sort_values("total_rest").index.to_list())
    ax.set(xlabel='Average total rest per day', ylabel='Species')
    ax.set(xlim=(0, 24))
    plt.tight_layout()
    ax = plt.axvline(12, ls='--', color='k')
    plt.savefig(os.path.join(rootdir, "total_rest_ordered_diet_{0}.png".format(datetime.date.today())))
    plt.close()

    # histogram of total rest
    feature_v_mean = feature_v.groupby('species_six').mean()
    feature_v_mean = feature_v_mean.reset_index()
    # feature_v_mean['tribe'] = 'undefined'
    # for row in feature_v_mean['species_six']:
    #     if sp_metrics.loc[sp_metrics['species_six'] == row, 'tribe'].empty:
    #         feature_v_mean.loc[feature_v_mean['species_six'] == row, 'tribe'] = 'undefined'
    #     else:
    #         feature_v_mean.loc[feature_v_mean['species_six'] == row, 'tribe'] = \
    #             sp_metrics.loc[sp_metrics['species_six'] == row, 'tribe'].values[0]

    fig = plt.figure(figsize=(10, 5))
    sns.histplot(data=feature_v, x='total_rest', binwidth=1, multiple="stack", color='skyblue')
    sns.histplot(data=feature_v_mean, x='total_rest', binwidth=1, multiple="stack", color='royalblue')
    plt.close()

    # total rest vs day speed
    feature_v['spd_max_mean'] = pd.concat([feature_v['spd_mean_day'], feature_v['spd_mean_night'],
                                           feature_v['spd_mean_predawn'], feature_v['spd_mean_dawn'],
                                           feature_v['spd_mean_dusk'], feature_v['spd_mean_postdusk']], axis=1).max(axis=1)

    fig = plt.figure(figsize=(5, 5))
    sns.scatterplot(data=feature_v, x='total_rest', y='spd_mean_day')
    fig = plt.figure(figsize=(5, 5))
    sns.scatterplot(data=feature_v, x='total_rest', y='spd_mean_night')
    fig = plt.figure(figsize=(5, 5))
    sns.regplot(data=feature_v, x='total_rest', y='spd_max_mean')

    # # bout lengths rest
    fig = plt.figure(figsize=(5, 10))
    ax = sns.boxplot(data=feature_v, y='species_six', x='rest_bout_mean_day', fliersize=1, hue="cluster",
                     order=diel_patterns.sort_values(by="cluster").species_six, dodge=False)
    ax = sns.swarmplot(data=feature_v, y='species_six', x='rest_bout_mean_day', color=".2", size=3,
                       order=diel_patterns.sort_values(by="cluster").species_six)
    ax.set(xlim=(0, 1250))
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "rest_bout_mean_day_{0}.png".format(datetime.date.today())))

    fig = plt.figure(figsize=(5, 10))
    ax = sns.boxplot(data=feature_v, y='species_six', x='rest_bout_mean_night', fliersize=1, hue="cluster",
                     order=diel_patterns.sort_values(by="cluster").species_six, dodge=False)
    ax = sns.swarmplot(data=feature_v, y='species_six', x='rest_bout_mean_night', color=".2", size=3,
                       order=diel_patterns.sort_values(by="cluster").species_six)
    ax.set(xlim=(0, 1250))
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "rest_bout_mean_night_{0}.png".format(datetime.date.today())))

    feature_v['rest_bout_mean_dn_dif'] = feature_v['rest_bout_mean_day'] - feature_v['rest_bout_mean_night']
    fig = plt.figure(figsize=(5, 10))
    ax = sns.boxplot(data=feature_v, y='species_six', x='rest_bout_mean_dn_dif', fliersize=1, hue="cluster",
                     order=diel_patterns.sort_values(by="cluster").species_six, dodge=False)
    ax = sns.swarmplot(data=feature_v, y='species_six', x='rest_bout_mean_dn_dif', color=".2", size=3,
                       order=diel_patterns.sort_values(by="cluster").species_six)
    plt.axvline(0, ls='--', color='k')
    ax.set(xlim=(-750, 600))
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "rest_bout_mean_dn_dif_{0}.png".format(datetime.date.today())))

    # # bout lengths non-rest
    fig = plt.figure(figsize=(5, 10))
    ax = sns.boxplot(data=feature_v, y='species_six', x='nonrest_bout_mean_day', fliersize=1, hue="cluster",
                     order=diel_patterns.sort_values(by="cluster").species_six, dodge=False)
    ax = sns.swarmplot(data=feature_v, y='species_six', x='nonrest_bout_mean_day', color=".2", size=3,
                       order=diel_patterns.sort_values(by="cluster").species_six)
    ax.set(xlim=(0, 1250))
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "nonrest_bout_mean_day_{0}.png".format(datetime.date.today())))

    fig = plt.figure(figsize=(5, 10))
    ax = sns.boxplot(data=feature_v, y='species_six', x='nonrest_bout_mean_night', fliersize=1, hue="cluster",
                     order=diel_patterns.sort_values(by="cluster").species_six, dodge=False)
    ax = sns.swarmplot(data=feature_v, y='species_six', x='nonrest_bout_mean_night', color=".2", size=3,
                       order=diel_patterns.sort_values(by="cluster").species_six)
    ax.set(xlim=(0, 1250))
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "nonrest_bout_mean_night_{0}.png".format(datetime.date.today())))

    feature_v['nonrest_bout_mean_dn_dif'] = feature_v['nonrest_bout_mean_day'] - feature_v['nonrest_bout_mean_night']
    fig = plt.figure(figsize=(5, 10))
    ax = sns.boxplot(data=feature_v, y='species_six', x='nonrest_bout_mean_dn_dif', fliersize=1, hue="cluster",
                     order=diel_patterns.sort_values(by="cluster").species_six, dodge=False)
    ax = sns.swarmplot(data=feature_v, y='species_six', x='nonrest_bout_mean_dn_dif', color=".2", size=3,
                       order=diel_patterns.sort_values(by="cluster").species_six)
    plt.axvline(0, ls='--', color='k')
    ax.set(xlim=(-4000, 3000))
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "nonrest_bout_mean_dn_dif_{0}.png".format(datetime.date.today())))

    feature_v_mean['rest_bout_mean_dn_dif'] = feature_v_mean['rest_bout_mean_day'] - feature_v_mean['rest_bout_mean_night']
    feature_v_mean['nonrest_bout_mean_dn_dif'] = feature_v_mean['nonrest_bout_mean_day'] - feature_v_mean[
        'nonrest_bout_mean_night']
    fig = plt.figure(figsize=(5, 10))
    sns.histplot(data=feature_v_mean, x='rest_bout_mean_dn_dif', hue='cluster', multiple="stack")
    fig = plt.figure(figsize=(5, 10))
    sns.histplot(data=feature_v_mean, x='rest_bout_mean_night', hue='cluster', multiple="stack")
    fig = plt.figure(figsize=(5, 10))
    sns.histplot(data=feature_v_mean, x='rest_bout_mean_day', hue='cluster', multiple="stack")
    plt.close('all')

    data_names = ['spd_mean', 'move_mean', 'rest_mean', 'y_mean', 'spd_std', 'move_std', 'rest_std', 'y_std',
                  'move_bout_mean', 'nonmove_bout_mean', 'rest_bout_mean', 'nonrest_bout_mean', 'move_bout_std',
                  'nonmove_bout_std', 'rest_bout_std', 'nonrest_bout_std']
    time_v2_m_names = ['predawn', 'dawn', 'day', 'dusk', 'postdusk', 'night']

    spd_means = ['spd_mean_predawn', 'spd_mean_dawn', 'spd_mean_day', 'spd_mean_dusk', 'spd_mean_postdusk',
                 'spd_mean_night']
    rest_means = ['rest_mean_predawn', 'rest_mean_dawn', 'rest_mean_day', 'rest_mean_dusk', 'rest_mean_postdusk',
                  'rest_mean_night']
    move_means = ['move_mean_predawn', 'move_mean_dawn', 'move_mean_day', 'move_mean_dusk', 'move_mean_postdusk',
                  'move_mean_night']
    rest_b_means = ['rest_bout_mean_predawn', 'rest_bout_mean_dawn', 'rest_bout_mean_day', 'rest_bout_mean_dusk',
                    'rest_bout_mean_postdusk', 'rest_bout_mean_night']
    nonrest_b_means = ['nonrest_bout_mean_predawn', 'nonrest_bout_mean_dawn', 'nonrest_bout_mean_day',
                       'nonrest_bout_mean_dusk',
                       'nonrest_bout_mean_postdusk', 'nonrest_bout_mean_night']
    move_b_means = ['move_bout_mean_predawn', 'move_bout_mean_dawn', 'move_bout_mean_day', 'move_bout_mean_dusk',
                    'move_bout_mean_postdusk', 'move_bout_mean_night']
    nonmove_b_means = ['nonmove_bout_mean_predawn', 'nonmove_bout_mean_dawn', 'nonmove_bout_mean_day',
                       'nonmove_bout_mean_dusk',
                       'nonmove_bout_mean_postdusk', 'nonmove_bout_mean_night']

    # movement_bouts = ['move_bout_mean', 'nonmove_bout_mean', 'move_bout_std']
    # rest_bouts = ['rest_bout_mean', 'nonrest_bout_mean']

    subset_feature_plt(averages, spd_means, 'Average speed mm/s')
    subset_feature_plt(averages, rest_means, 'Average fraction rest per hour')
    subset_feature_plt(averages, move_b_means, 'Average movement bout length')
    subset_feature_plt(averages, nonmove_b_means, 'Average nonmovement bout length')
    subset_feature_plt(averages, rest_b_means, 'Average rest bout length')
    subset_feature_plt(averages, nonrest_b_means, 'Average nonrest bout length')
    # make clustermaps!

    ### Daily activity pattern ###
    diel_fish = daily_more_than_pattern_individ(feature_v, species, plot=False)
    diel_species = daily_more_than_pattern_species(averages, plot=False)
    day_night_ratio_fish = day_night_ratio_individ(feature_v)
    day_night_ratio_sp = day_night_ratio_species(averages)

    sns.clustermap(pd.concat([day_night_ratio_fish.ratio, diel_fish.crepuscular * 1], axis=1), col_cluster=False,
                   yticklabels=day_night_ratio_fish.species_six, cmap='RdBu_r', vmin=0, vmax=2)
    plt.tight_layout()

    sns.clustermap(pd.concat([day_night_ratio_sp, diel_species.crepuscular * 1], axis=1), col_cluster=False, cmap='RdBu_r',
                   vmin=0, vmax=2)
    plt.tight_layout()

    #### Correlations for average behaviour vs average species ecological measures
    ave_rest = averages.loc[['total_rest', 'rest_mean_night', 'rest_mean_day', 'fish_length_mm'], :].transpose().reset_index().rename(
        columns={'index': 'sp'})
    ave_rest['night-day_dif_rest'] = ave_rest.rest_mean_night - ave_rest.rest_mean_day
    sp_in_both = set(ave_rest.sp) & set(ronco_data.sp)
    missing_in_ronco = set(ave_rest.sp) - set(sp_in_both)
    df = pd.merge(ronco_data, ave_rest, how='left')
    df = df.drop(df.loc[(np.isnan(df.loc[:, 'total_rest']))].index).reset_index(drop=True)
    df = df.rename(columns={'sp': 'species_six'})
    df = pd.merge(df, cichlid_meta, how='left', on='species_six')

    fig = plt.figure(figsize=(5, 5))
    sns.regplot(data=df, x='total_rest', y='d15N')

    fig = plt.figure(figsize=(5, 5))
    sns.regplot(data=df, x='total_rest', y='body_PC1')

    fig = plt.figure(figsize=(5, 5))
    sns.scatterplot(data=df, x='d15N', y='d13C', hue='total_rest')

    sub_df = df.groupby('species_six').mean()
    for behav in ['total_rest', 'night-day_dif_rest', 'size_female']:
        for col in ['body_PC1', 'body_PC2', 'LPJ_PC1', 'LPJ_PC2', 'oral_PC1', 'oral_PC2', 'd15N', 'd13C', 'size_male',
                    'size_female', 'fish_length_mm']:
            non_nan_rows = sub_df[sub_df[behav].isna()==False].index & sub_df[sub_df[col].isna() == False].index
            model, r_sq = run_linear_reg(sub_df.loc[non_nan_rows, behav], sub_df.loc[non_nan_rows, col])
            plt_lin_reg(sub_df.loc[non_nan_rows, behav], sub_df.loc[non_nan_rows, col], model, r_sq)

    #### draw convex hull for each temporal guild ####
    # speed_mm guilds 24.02.2022
    # dic = {'diurnal': [3], 'nocturnal': [1], 'crepuscular': [7, 8, 9, 10, 11], 'undefined': [2, 4, 5, 6, 12, 13]}
    # dic = {'diurnal': [1], 'nocturnal': [8], 'crepuscular': [5, 6, 7], 'undefined': [2, 3, 4, 9, 10, 11, 12]}
    # col_dic = {'diurnal': 'gold', 'nocturnal': 'royalblue', 'crepuscular': 'mediumorchid', 'undefined': 'black'}


    # pelagic and trophic levels
    fig = plt.figure(figsize=(3, 3))
    ax = sns.scatterplot(df.loc[:, 'd13C'], df.loc[:, 'd15N'], color='silver', s=12)
    for key in dic_simple:
        # find the species which are in diel group
        overlap_species = list(set(diel_patterns.loc[diel_patterns.cluster.isin(dic_simple[key]), 'species_six'].to_list()) &
             set(sub_df.index.to_list()))
        points = sub_df.loc[overlap_species, ['d13C', 'd15N']]
        points = points.to_numpy()
        plt.scatter(points[:, 0], points[:, 1],  color=col_dic_simple[key], s=12)
        if key in ['diurnal', 'nocturnal', 'crepuscular']:
            hull = ConvexHull(points)
            for simplex in hull.simplices:
                ax.plot(points[simplex, 0], points[simplex, 1], color=col_dic_simple[key])
    ax.set_xlabel('$\delta^{13} C$')
    ax.set_ylabel('$\delta^{15} N$')
    sns.despine(top=True, right=True)
    fig.tight_layout()
    plt.savefig(os.path.join(rootdir, "d15N_d13C_temporal-guilds_{0}.png".format(datetime.date.today())))
    plt.close()

    # trophic guilds
    guilds = df.Diet.unique()
    col_dic = {'Zooplanktivore': 'sandybrown', 'Algivore': 'mediumseagreen', 'Invertivore': 'tomato', 'Piscivore': 'steelblue'}
    fig = plt.figure(figsize=(3, 3))
    ax = sns.scatterplot(df.loc[:, 'd13C'], df.loc[:, 'd15N'], color='silver', s=12)
    for key in guilds:
        # find the species which are in the diet guild
        guild_species = set(df.loc[df.Diet == key, 'species_six'].unique())
        points = sub_df.loc[guild_species, ['d13C', 'd15N']]
        points = points.to_numpy()
        plt.scatter(points[:, 0], points[:, 1],  color=col_dic_simple[key], s=12)
    ax.set_xlabel('$\delta^{13} C$')
    ax.set_ylabel('$\delta^{15} N$')
    sns.despine(top=True, right=True)
    fig.tight_layout()
    plt.savefig(os.path.join(rootdir, "d15N_d13C_diet-guilds_{0}.png".format(datetime.date.today())))
    plt.close()

    first = True
    for key in dic_simple:
        cluster_sp = diel_patterns.loc[diel_patterns.cluster.isin(dic_simple[key]), 'species_six'].to_list()
        new_df = df.loc[df.species_six.isin(cluster_sp), ['species_six', 'Diet']].drop_duplicates().Diet.value_counts()
        new_df = new_df.reset_index()
        new_df['daytime'] = key
        if first:
            df_group = new_df
            first = False
        else:
            df_group = pd.concat([df_group, new_df])
    df_group = df_group.rename(columns={'Diet': 'species_n', 'index': 'Diet'}).reset_index(drop=True)

    colors = ['sandybrown', 'mediumseagreen', 'tomato', 'steelblue']
    customPalette = sns.set_palette(sns.color_palette(colors))
    ax = sns.barplot(x="daytime", y="species_n", hue="Diet", data=df_group, palette=customPalette)
    ax.set(xlabel=None)
    ax.set(ylabel="# of species")
    plt.savefig(os.path.join(rootdir, "diet-guilds_hist_{0}.png".format(datetime.date.today())))
    plt.close()