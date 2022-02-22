import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.stats.multitest as smt

from cichlidanalysis.utils.species_names import six_letter_sp_name
from cichlidanalysis.io.meta import extract_meta


def diel_pattern_stats_individ_bin(fish_tracks_ds, feature='movement'):
    """ To define if a fish is diurnal or nocturnal we use a wilcoxon test to find if day means are different from night
    means

    :param fish_tracks_ds:
    :param feature:
    :return:
    """
    fishes = fish_tracks_ds.FishID.unique()

    feature_dist_day = fish_tracks_ds.loc[fish_tracks_ds.daytime == 'd', [feature, 'FishID', 'day_n', 'species']].groupby(
        ['FishID', 'day_n']).mean().reset_index()
    feature_dist_night = fish_tracks_ds.loc[fish_tracks_ds.daytime == 'n', [feature, 'FishID', 'day_n']].groupby(
        ['FishID', 'day_n']).mean().reset_index()
    feature_dist_day = feature_dist_day.rename(columns={feature: "day"})
    feature_dist_night = feature_dist_night.rename(columns={feature: "night"})
    feature_dist = pd.merge(feature_dist_day, feature_dist_night, how='inner', on=["FishID", "day_n"])

    # as rest is opposite valence (high = low activity) we switch it for consistent colour of plotting
    # (blue = more nocturnal)
    if feature == 'rest':
        feature_dist['day'] = np.abs(feature_dist['day']-1)
        feature_dist['night'] = np.abs(feature_dist['night'] - 1)

    stats_array = np.zeros([len(fishes), 7])
    for fish_n, fish in enumerate(fishes):
        if len(feature_dist.loc[feature_dist.FishID == fish, 'day']) > 3:
            stats_array[fish_n, 0] = stats.shapiro(feature_dist.loc[feature_dist.FishID == fish, 'day'])[1]
            stats_array[fish_n, 1] = stats.shapiro(feature_dist.loc[feature_dist.FishID == fish, 'night'])[1]
            # stats_array[fish_n, 2:4] = stats.ttest_rel(feature_dist.loc[feature_dist.FishID == fish, 'day'],
            #                                            feature_dist.loc[feature_dist.FishID == fish, 'night'])
            try:
                # as wilcoxon can't handle a total difference == 0 catch these cases
                stats_array[fish_n, 2:4] = stats.wilcoxon(feature_dist.loc[feature_dist.FishID == fish, 'day'],
                                                       feature_dist.loc[feature_dist.FishID == fish, 'night'])
            except:
                stats_array[fish_n, 2:4] = np.NaN
            stats_array[fish_n, 4] = feature_dist.loc[feature_dist.FishID == fish, 'day'].mean() > \
                                     feature_dist.loc[feature_dist.FishID == fish, 'night'].mean()
            stats_array[fish_n, 5] = feature_dist.loc[feature_dist.FishID == fish, 'day'].mean() - \
                                     feature_dist.loc[feature_dist.FishID == fish, 'night'].mean()
            stats_array[fish_n, 6] = feature_dist.loc[feature_dist.FishID == fish, 'day'].mean() / \
                                     feature_dist.loc[feature_dist.FishID == fish, 'night'].mean()
        else:
            print("skipped {} as not enough time points".format(fish))
            stats_array[fish_n, 0:] = np.NaN

    df = pd.DataFrame(stats_array, columns=['norm_day', 'norm_night', 't_stat', 't_pval', 'day_higher', 'day_night_dif',
                                            'day_night_ratio'])
    df['FishID'] = fishes

    # multiple testing correction
    # ### bonferroni
    # corrected_alpha = 0.05 / len(df['t_pval'])
    # df['t_pval_corr_sig2'] = df['t_pval'] < corrected_alpha

    # ### FDR Benjamini/Hochberg
    t_vals = df.t_pval.replace(np.NaN, 1)
    df['t_pval_corr_sig'] = smt.fdrcorrection(t_vals, alpha=0.05, method='indep', is_sorted=False)[1]
    df.loc[df.t_pval == np.NaN, 't_pval_corr_sig'] = np.nan

    df['diel_pattern'] = 'undefined'
    for (index_label, row_series) in df.iterrows():
        if row_series.t_pval_corr_sig < 0.05:
            if row_series.day_higher == 1:
                df.loc[index_label, 'diel_pattern'] = 'diurnal'     # diurnal
            else:
                df.loc[index_label, 'diel_pattern'] = 'nocturnal'     # nocturnal

    df['species_six'] = 'blank'
    for fish in fishes:
        df.loc[df['FishID'] == fish, 'species_six'] = six_letter_sp_name(extract_meta(fish)['species'])

    return df


def diel_pattern_stats_species_bin(fish_tracks_ds, feature='movement'):
    """ To define if a species is diurnal or nocturnal we use the wilcoxon test to find if day means are different from
    night means

    :param fish_tracks_ds:
    :param feature:
    :return:
    """
    fishes = fish_tracks_ds.FishID.unique()
    all_species = fish_tracks_ds.species.unique()

    feature_dist_day = fish_tracks_ds.loc[fish_tracks_ds.daytime == 'd', [feature, 'FishID', 'species']].groupby(
        ['species', 'FishID']).mean().reset_index()
    feature_dist_night = fish_tracks_ds.loc[fish_tracks_ds.daytime == 'n', [feature, 'FishID', 'species']].groupby(
        ['species', 'FishID']).mean().reset_index()
    feature_dist_day = feature_dist_day.rename(columns={feature: "day"})
    feature_dist_night = feature_dist_night.rename(columns={feature: "night"})
    feature_dist = pd.merge(feature_dist_day, feature_dist_night, how='inner', on=["FishID", "species"])

    # as rest is opposite valence (high = low activity) we switch it for consistent colour of plotting
    # (blue = more nocturnal)
    if feature == 'rest':
        feature_dist['day'] = np.abs(feature_dist['day']-1)
        feature_dist['night'] = np.abs(feature_dist['night'] - 1)

    stats_array = np.zeros([len(all_species), 7])
    for species_n, species in enumerate(all_species):
        if len(feature_dist.loc[feature_dist.species == species, 'day']) > 3:
            stats_array[species_n, 0] = stats.shapiro(feature_dist.loc[feature_dist.species == species, 'day'])[1]
            stats_array[species_n, 1] = stats.shapiro(feature_dist.loc[feature_dist.species == species, 'night'])[1]
            # stats_array[fish_n, 2:4] = stats.ttest_rel(feature_dist.loc[feature_dist.FishID == fish, 'day'],
            #                                            feature_dist.loc[feature_dist.FishID == fish, 'night'])
            try:
                # as wilcoxon can't handle a total difference == 0 catch these cases
                stats_array[species_n, 2:4] = stats.wilcoxon(feature_dist.loc[feature_dist.species == species, 'day'],
                                                          feature_dist.loc[feature_dist.species == species, 'night'])
            except:
                stats_array[species_n, 2:4] = np.NaN
            stats_array[species_n, 4] = feature_dist.loc[feature_dist.species == species, 'day'].mean() > \
                                     feature_dist.loc[feature_dist.species == species, 'night'].mean()
            stats_array[species_n, 5] = feature_dist.loc[feature_dist.species == species, 'day'].mean() - \
                                     feature_dist.loc[feature_dist.species == species, 'night'].mean()
            stats_array[species_n, 6] = feature_dist.loc[feature_dist.species == species, 'day'].mean() / \
                                     feature_dist.loc[feature_dist.species == species, 'night'].mean()
        else:
            print("skipped {} as not enough time points".format(species))
            stats_array[species_n, 0:] = np.NaN

    df = pd.DataFrame(stats_array, columns=['norm_day', 'norm_night', 't_stat', 't_pval', 'day_higher', 'day_night_dif',
                                            'day_night_ratio'])
    df['species'] = all_species

    # ### FDR Benjamini/Hochberg
    # as FDR can't deal with NaNs replace with 1
    t_vals = df.t_pval.replace(np.NaN, 1)
    df['t_pval_corr_sig'] = smt.fdrcorrection(t_vals, alpha=0.05, method='indep', is_sorted=False)[1]
    df.loc[df.t_pval == np.NaN, 't_pval_corr_sig'] = np.nan

    df['diel_pattern'] = 'undefined'
    for (index_label, row_series) in df.iterrows():
        if row_series.t_pval_corr_sig < 0.05:
            if row_series.day_higher == 1:
                df.loc[index_label, 'diel_pattern'] = 'diurnal'     # diurnal
            else:
                df.loc[index_label, 'diel_pattern'] = 'nocturnal'     # nocturnal

    return df

def daily_more_than_pattern_individ(feature_v, species, plot=False):
    """ Daily activity pattern for individuals of all fish

    :param feature_v: feature vector with 'rest_mean_night', 'rest_mean_day', 'rest_mean_predawn', 'rest_mean_dawn',
    'rest_mean_dusk', 'rest_mean_postdusk'
    :param species:
    :param plot: to plot or not
    :return:
    """
    thresh = 1.1
    first = True
    for species_name in species:
        night = feature_v.loc[feature_v.species == species_name, 'rest_mean_night']
        day = feature_v.loc[feature_v.species == species_name, 'rest_mean_day']
        dawn_dusk = feature_v.loc[
            feature_v.species == species_name, ['rest_mean_predawn', 'rest_mean_dawn', 'rest_mean_dusk',
                                                'rest_mean_postdusk']].mean(axis=1)

        nocturnal = (night * thresh < day) & (night * thresh < dawn_dusk)
        dirunal = (day * thresh < night * 1.1) & (day * thresh < dawn_dusk)
        crepuscular = (dawn_dusk * thresh < night) & (dawn_dusk * thresh < day)
        cathemeral = (nocturnal * 1 + dirunal * 1 + crepuscular * 1) == 0

        individ_diel = pd.concat([dirunal, nocturnal, crepuscular, cathemeral], axis=1)
        individ_diel = individ_diel.rename(columns={0: 'diurnal', 1: 'nocturnal', 2: 'crepuscular', 3: 'cathemeral'})

        if first:
            all_individ_diel = individ_diel
            first = False
        else:
            all_individ_diel = pd.concat([all_individ_diel, individ_diel], axis=0)

        if plot:
            fig = plt.figure(figsize=(5, 5))
            plt.imshow(individ_diel)
            plt.xticks(np.arange(0, 4), labels=['diurnal', 'nocturnal', 'crepuscular', 'cathemeral'], rotation=45,
                       ha='right', va='top')
            plt.title(species_name)
            plt.tight_layout()

    return all_individ_diel


def daily_more_than_pattern_species(averages, plot=False):
    """ Return daily activity pattern for each species

    :param averages: average of feature_v with 'rest_mean_night', 'rest_mean_day', 'rest_mean_predawn', 'rest_mean_dawn',
    'rest_mean_dusk', 'rest_mean_postdusk'
    :param plot: if  to plot or not
    :return: species_diel: df with species and  call for diel pattern
    """
    thresh = 1.1
    night = averages.loc['rest_mean_night', :]
    day = averages.loc['rest_mean_day', :]
    dawn_dusk = averages.loc[['rest_mean_predawn', 'rest_mean_dawn', 'rest_mean_dusk', 'rest_mean_postdusk'], :].mean(
        axis=0)

    nocturnal = (night * thresh < day) & (night * thresh < dawn_dusk)
    dirunal = (day * thresh < night * 1.1) & (day * thresh < dawn_dusk)
    crepuscular = (dawn_dusk * thresh < night) & (dawn_dusk * thresh < day)
    cathemeral = (nocturnal * 1 + dirunal * 1 + crepuscular * 1) == 0

    species_diel = pd.concat([dirunal, nocturnal, crepuscular, cathemeral], axis=1)
    species_diel = species_diel.rename(columns={0: 'diurnal', 1: 'nocturnal', 2: 'crepuscular', 3: 'cathemeral'})
    if plot:
        fig = plt.figure(figsize=(5, 5))
        plt.imshow(species_diel)
        plt.yticks(np.arange(0, len(averages.columns)), labels=averages.columns)
        plt.xticks(np.arange(0, 4), labels=['diurnal', 'nocturnal', 'crepuscular', 'cathemeral'], rotation=45, ha='right',
                   va='top')
        plt.tight_layout()

    return species_diel


def day_night_ratio_individ(feature_v):
    """ Find the day/night ratio of non-rest for the daily average rest trace of each individual fish

    :param feature_v:
    :return:
    """
    # all individuals
    night = feature_v.loc[:, 'rest_mean_night']
    day = feature_v.loc[:, 'rest_mean_day']
    # dawn_dusk = feature_v.loc[:, ['rest_mean_predawn', 'rest_mean_dawn', 'rest_mean_dusk', 'rest_mean_postdusk']].mean(
    #     axis=1)

    day_night_ratio = np.abs(1 - day) / np.abs(1 - night)
    day_night_ratio = day_night_ratio.rename('ratio')
    day_night_ratio = day_night_ratio.to_frame()
    day_night_ratio["species_six"] = feature_v.species_six

    ax = sns.boxplot(data=day_night_ratio, y='species_six', x='ratio')
    ax = sns.swarmplot(data=day_night_ratio, y='species_six', x='ratio', color=".2")
    ax = plt.axvline(1, ls='--', color='k')
    plt.xlabel('Day/night ratio')
    plt.xscale('log')
    plt.tight_layout()

    return day_night_ratio


def day_night_ratio_species(averages):
    """ Find the day/night ratio of non-rest for the daily average average rest trace of each species

    :param averages:
    :return:
    """
    # for each species
    night = averages.loc['rest_mean_night', :]
    day = averages.loc['rest_mean_day', :]
    # dawn_dusk = averages.loc[['rest_mean_predawn', 'rest_mean_dawn', 'rest_mean_dusk', 'rest_mean_postdusk'], :]\
    #     .mean(axis=0)

    # ratios
    day_night_ratio = np.abs(1-day)/np.abs(1-night)
    day_night_ratio = day_night_ratio.rename('day_night_ratio')
    day_night_ratio = day_night_ratio.to_frame()

    return day_night_ratio


def replace_crep_peaks(fish_peaks, fish_feature, fish_num, epoques):
    """for crepuscular peaks, this finds if the first row (intra day 30min bin) is == 0, which means it didn't have a
    peak and replaces it's peak height with the value of the mode of the first row. It then corrects the peak amplitude.

    :param fish_peaks:
    :param fish_feature:
    :param fish_num:
    :return: fish_peaks
    """
    # check if any of the peaks need replacing
    if (fish_peaks[0, :] == 0).any():
        common_peak = stats.mode(fish_peaks[0, :])[0][0]
        no_peaks = np.where(fish_peaks[0, :] == 0)[0]
        for no_peak in no_peaks:
            # add in peak height
            fish_peaks[2, no_peak] = fish_feature.iloc[int(epoques[no_peak] + common_peak), fish_num]
            # correct peak amplitude
            fish_peaks[3, no_peak] = fish_peaks[3, no_peak] + fish_peaks[2, no_peak]
    return fish_peaks


def make_fish_peaks_df(fish_peaks, fish_id):
    """ makes df from np.array of fish peaks data (4 rows where [0, 2, 3] = ['peak_loc', 'peak_height',
    'peak_amplitude']

    :param fish_peaks:
    :param fish_id:
    :return: fish_peaks_df
    """
    fish_peaks_df = pd.DataFrame(fish_peaks[[0, 2, 3], :].T, columns=['peak_loc', 'peak_height', 'peak_amplitude'])
    fish_peaks_df = fish_peaks_df.reset_index().rename(columns={'index': 'crep_num'})
    fish_peaks_df['FishID'] = fish_id
    return fish_peaks_df


# def day_night_ratio_individ_30min(fish_tracks_bin, feature='movement'):
#     """ Find the day/night ratio of non-rest for the daily average rest trace of each individual fish
#
#     :param fish_tracks_bin:
#     :return:
#     """
#     # all individuals
#     night = fish_tracks_bin.loc[fish_tracks_bin.daytime == 'n', [feature, 'FishID']].groupby('FishID').mean()
#     day = fish_tracks_bin.loc[fish_tracks_bin.daytime == 'd', [feature, 'FishID']].groupby('FishID').mean()
#
#     day_night_ratio = np.abs(1 - day) / np.abs(1 - night)
#     day_night_ratio = day_night_ratio.rename(columns={feature: "ratio"})
#     day_night_ratio = day_night_ratio.reset_index()
#
#     ax = sns.boxplot(data=day_night_ratio, y='FishID', x='ratio')
#     ax = sns.swarmplot(data=day_night_ratio, y='FishID', x='ratio', color=".2")
#     ax = plt.axvline(1, ls='--', color='k')
#     plt.xlabel('Day/night ratio')
#     plt.xscale('log')
#     plt.tight_layout()
#
#     return day_night_ratio
