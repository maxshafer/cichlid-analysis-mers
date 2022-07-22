import os

import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_day_night_species_ave(rootdir, fish_diel_patterns, fish_diel_patterns_sp, feature, input_type='day_night_dif'):
    """ Plots the individual fish diurnality as scatter and bar plot. Colouring indicates the species level call,
    species are reorder by mean

    :return:
    """
    sorted_species_idx = fish_diel_patterns.groupby('species').median().sort_values(by=input_type).index

    # clrs = [(sns.color_palette(palette='RdBu')[0]), sns.color_palette(palette='RdBu')[5], (128/255, 128/255, 128/255)]
    # hue_ordering = ['diurnal', 'nocturnal', 'undefined']

    # row colours by median value
    box_cols = []
    sorted_day_night_dif = fish_diel_patterns.groupby('species').median().sort_values(by=input_type).loc[:,
                           input_type]
    scale_max = sorted_day_night_dif.max()
    scale_min = sorted_day_night_dif.min()

    # find the largest number and use this to scale so that 0 = mid colour scale
    scale_factor = np.max([scale_max, abs(scale_min)])

    sorted_day_night_dif_scaled = (sorted_day_night_dif + scale_factor)/(scale_factor + scale_factor)

    for species in sorted_species_idx:
        box_cols.append(plt.cm.get_cmap('bwr')(sorted_day_night_dif_scaled.loc[species]))

    # row colours by diel pattern
    row_cols = []
    for species in sorted_species_idx:
        sp_pattern = fish_diel_patterns_sp.loc[fish_diel_patterns_sp.species == species, "diel_pattern"].values[0]
        if sp_pattern == 'diurnal':
            row_cols.append('gold')
        elif sp_pattern == 'nocturnal':
            row_cols.append('gold')
        elif sp_pattern == 'undefined':
            row_cols.append((211/255, 211/255, 211/255))

    f, ax = plt.subplots(figsize=(10, 5))
    bp = sns.boxplot(data=fish_diel_patterns, x='species', y=input_type, palette=box_cols, ax=ax,
                     order=sorted_species_idx, fliersize=0, boxprops=dict(alpha=.7))
    sns.stripplot(data=fish_diel_patterns, x='species', y=input_type, ax=ax, size=4, order=sorted_species_idx, color='k')
    ax.set(ylabel='Day - night activity', xlabel='Species')
    ax.set_xticklabels(labels=sorted_species_idx, rotation=90)
    ax.tick_params(left=True, bottom=True)
    sns.despine()
    # statistical annotation
    y, h, col = fish_diel_patterns[input_type].max(), 2, 'k'
    for species_n, species_i in enumerate(sorted_species_idx):
        sig = fish_diel_patterns_sp.loc[fish_diel_patterns_sp.species == species_i, 't_pval_corr_sig'].values[0]
        if sig < 0.05:
            plt.text(species_n, y*1.05, "*", ha='center', va='bottom', color=col)
    if input_type == 'day_night_dif':
        plt.axhline(0, ls='--', color='k')
    else:
        plt.axhline(1, ls='--', color='k')
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "species_diurnal-nocturnal_30min_median-value_simple_{0}_{1}_{2}.png"
                             .format(feature, dt.date.today(), input_type)))
    plt.close()
    return


def plot_day_night_species(rootdir, fish_diel_patterns, feature, input_type='day_night_dif'):
    """ Plots the individual fish diurnality as scatter and bar plot. Colouring indicates the species level call,
    species are reorder by mean

    :return:
    """
    sorted_index = fish_diel_patterns.groupby('species').median().sort_values(by=input_type).index

    # clrs = [(sns.color_palette(palette='RdYlBu')[0]), sns.color_palette(palette='RdYlBu')[5], sns.color_palette(palette='RdYlBu')[-5]]
    clrs = [(sns.color_palette(palette='RdBu')[0]), sns.color_palette(palette='RdBu')[5], (128/255, 128/255, 128/255)] #(171/255, 221/255, 164/255)]
    hue_ordering = ['diurnal', 'nocturnal', 'undefined']

    # row colours by median value
    box_cols = []
    sorted_day_night_dif = fish_diel_patterns.groupby('species').median().sort_values(by=input_type).loc[:, input_type]
    sorted_day_night_dif_scaled =(sorted_day_night_dif-sorted_day_night_dif.min())/(sorted_day_night_dif.max()-sorted_day_night_dif.min())
    for i in sorted_index:
        box_cols.append(plt.cm.get_cmap('bwr')(sorted_day_night_dif_scaled.loc[i]))

    # row colours by diel pattern
    row_cols = []
    subset = fish_diel_patterns.loc[:, ['species', 'species_diel_pattern']].drop_duplicates(subset=["species"])
    for i in sorted_index:
        sp_pattern = subset.loc[subset.species == i, "species_diel_pattern"].values[0]
        if sp_pattern == 'diurnal':
            # row_cols.append((255 / 255, 224 / 255, 179 / 255))
            # row_cols.append(clrs[0])
            row_cols.append('gold')
        elif sp_pattern == 'nocturnal':
            # row_cols.append((153 / 255, 204 / 255, 255 / 255))
            # row_cols.append(clrs[1])
            row_cols.append('gold')
        elif sp_pattern == 'undefined':
            # row_cols.append((179 / 255, 230 / 255, 179 / 255))
            # row_cols.append((171/255, 221/255, 164/255))
            row_cols.append((211/255, 211/255, 211/255))
            # row_cols.append((0/255, 0/255, 0/255))

    # plotting horizontal
    f, ax = plt.subplots(figsize=(10, 5))
    bp = sns.boxplot(data=fish_diel_patterns, x='species', y=input_type, palette=box_cols, ax=ax,
                order=sorted_index, fliersize=0, boxprops=dict(alpha=.7))
    for patch, color in zip(bp.artists, row_cols):
        patch.set_edgecolor(color)
        patch.set_linewidth(3)
        # patch.set_alpha(1)
    sns.stripplot(data=fish_diel_patterns, x='species', y=input_type, hue='diel_pattern', ax=ax, size=4,
                  palette=clrs, hue_order=hue_ordering, order=sorted_index)
    ax.set(ylabel=input_type, xlabel='Species')
    ax.set_xticklabels(labels=sorted_index, rotation=90)
    if input_type == 'day_night_dif':
        ax = plt.axhline(0, ls='--', color='k')
    else:
        ax = plt.axhline(1, ls='--', color='k')
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "species_diurnal-nocturnal_30min_median-value_{0}_{1}_{2}.png".format(feature,
                             dt.date.today(), input_type)))
    plt.close()

    # plotting horizontal
    f, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=fish_diel_patterns, x='species', y=input_type, palette=row_cols, ax=ax,
                order=sorted_index, fliersize=0, boxprops=dict(alpha=.3))
    sns.stripplot(data=fish_diel_patterns, x='species', y=input_type, hue='diel_pattern', ax=ax, size=4,
                  palette=clrs, hue_order=hue_ordering, order=sorted_index)
    ax.set(ylabel=input_type, xlabel='Species')
    ax.set_xticklabels(labels=sorted_index, rotation=90)
    if input_type == 'day_night_dif':
        ax = plt.axhline(0, ls='--', color='k')
    else:
        ax = plt.axhline(1, ls='--', color='k')
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "species_diurnal-nocturnal_30min_diel-pattern_{0}_{1}_{2}.png".format(feature,
                                                                                        dt.date.today(), input_type)))
    plt.close()


def plot_cre_dawn_dusk_strip_v(rootdir, all_feature_combined, feature, peak_feature='peak_amplitude'):

    sorted_index = all_feature_combined.groupby(by='species').mean().sort_values(by=peak_feature).index
    grped_bplot = sns.catplot(y='species',
                              x=peak_feature,
                              hue="twilight",
                              kind="box",
                              legend=False,
                              height=10,
                              aspect=0.6,
                              data=all_feature_combined,
                              fliersize=0,
                              boxprops=dict(alpha=.3),
                              order=sorted_index,
                              palette="flare")
    plt.axvline(0, color='k', linestyle='--')
    grped_bplot = sns.stripplot(y='species',
                                x=peak_feature,
                                hue='twilight',
                                jitter=True,
                                dodge=True,
                                marker='o',
                                data=all_feature_combined,
                                order=sorted_index,
                                palette="flare")
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "species_crepuscularity_{0}_{1}_{2}.png".format(feature, peak_feature, dt.date.today())))


def colors_from_values(values, palette_name):
    """ https://stackoverflow.com/questions/36271302/changing-color-scale-in-seaborn-bar-plot

    :param values:
    :param palette_name:
    :return:
    """
    # normalize the values to range [0, 1]
    normalized = (values - min(values)) / (max(values) - min(values))
    # convert to indices
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    # use the indices to get the colors
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)


def plot_cre_dawn_dusk_strip_box(rootdir, cres_peaks_i, feature, peak_feature='peak_amplitude'):
    """ Plot the crepuscular data as a strip and box plot

    :param rootdir:
    :param all_feature_combined:
    :param feature:
    :return:
    """
    sorted_index = cres_peaks_i.groupby(by='species').mean().sort_values(by=peak_feature).index

    dawn_index = cres_peaks_i.groupby(by=['species', 'twilight']).median().reset_index()
    sorted_index = dawn_index.drop(dawn_index[dawn_index.twilight == 'dusk'].index).set_index('species').sort_values(by=peak_feature).index
    twilights = ['dawn', 'dusk']
    for period in twilights:
        grped_bplot = sns.catplot(x='species',
                                  y=peak_feature,
                                  kind="box",
                                  legend=False,
                                  height=5,
                                  aspect=2,
                                  data=cres_peaks_i.loc[cres_peaks_i.twilight == period],
                                  fliersize=0,
                                  boxprops=dict(alpha=.3),
                                  order=sorted_index,
                                  palette=colors_from_values(cres_peaks_i.loc[cres_peaks_i.twilight == period,
                                                                              [peak_feature, 'species']].groupby
                                                             ('species').median().reindex(sorted_index).loc[:, peak_feature], "flare"),
                                  saturation=1)

        sns.stripplot(x='species',
                      y=peak_feature,
                      data=cres_peaks_i.loc[cres_peaks_i.twilight == period],
                      order=sorted_index,
                      palette=colors_from_values(cres_peaks_i.loc[cres_peaks_i.twilight == period,
                                                                  [peak_feature, 'species']].groupby('species')
                                                 .median().reindex(sorted_index).loc[:, peak_feature], "flare"),
                      size=3,
                      jitter=0.5).set(title=period)
        grped_bplot.set_xticklabels(labels=sorted_index, rotation=90)
        if peak_feature == 'peak_amplitude':
            grped_bplot.set(ylabel='Peak amplitude from baseline', xlabel='Species')
        elif peak_feature == 'peak':
            grped_bplot.set(ylabel='Peak fraction', xlabel='Species')
        ax = plt.axhline(0, ls='--', color='k')
        plt.tight_layout()
        plt.savefig(os.path.join(rootdir, "species_crepuscular_30min_box_sort_{0}_{1}_{2}_{3}.png".format(period, dt.date.today(), feature, peak_feature)))
        plt.close()
    return
