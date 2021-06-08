import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


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
