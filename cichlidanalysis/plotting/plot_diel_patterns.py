import os

import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt


def plot_day_night_species(rootdir, fish_diel_patterns):
    """ Plots the individual fish diurnality as scatter and bar plot. Colouring indicates the species level call,
    species are reorder by mean

    :return:
    """
    sorted_index = fish_diel_patterns.groupby('species_six').median().sort_values(by='day_night_dif').index

    # clrs = [(sns.color_palette(palette='RdYlBu')[0]), sns.color_palette(palette='RdYlBu')[5], sns.color_palette(palette='RdYlBu')[-5]]
    clrs = [(sns.color_palette(palette='RdBu')[0]), sns.color_palette(palette='RdBu')[5], (128/255, 128/255, 128/255)] #(171/255, 221/255, 164/255)]
    hue_ordering = ['diurnal', 'nocturnal', 'undefined']

    # row colours by median value
    box_cols = []
    sorted_day_night_dif = fish_diel_patterns.groupby('species_six').median().sort_values(by='day_night_dif').day_night_dif
    sorted_day_night_dif_scaled =(sorted_day_night_dif-sorted_day_night_dif.min())/(sorted_day_night_dif.max()-sorted_day_night_dif.min())
    for i in sorted_index:
        box_cols.append(plt.cm.get_cmap('bwr')(sorted_day_night_dif_scaled.loc[i]))

    # row colours by diel pattern
    row_cols = []
    subset = fish_diel_patterns.loc[:, ['species_six', 'species_diel_pattern']].drop_duplicates(subset=["species_six"])
    for i in sorted_index:
        sp_pattern = subset.loc[subset.species_six == i, "species_diel_pattern"].values[0]
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
    bp = sns.boxplot(data=fish_diel_patterns, x='species_six', y='day_night_dif', palette=box_cols, ax=ax,
                order=sorted_index, fliersize=0, boxprops=dict(alpha=.7))
    for patch, color in zip(bp.artists, row_cols):
        patch.set_edgecolor(color)
        patch.set_linewidth(3)
        # patch.set_alpha(1)
    sns.stripplot(data=fish_diel_patterns, x='species_six', y='day_night_dif', hue='diel_pattern', ax=ax, size=4,
                  palette=clrs, hue_order=hue_ordering, order=sorted_index)
    ax.set(ylabel='Day mean - night mean', xlabel='Species')
    ax.set_xticklabels(labels=sorted_index, rotation=45)
    ax = plt.axhline(0, ls='--', color='k')
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "species_diurnal-nocturnal_30min_median-value_{0}.png".format(dt.date.today())))
    plt.close()

    # plotting horizontal
    f, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=fish_diel_patterns, x='species_six', y='day_night_dif', palette=row_cols, ax=ax,
                order=sorted_index, fliersize=0, boxprops=dict(alpha=.3))
    sns.stripplot(data=fish_diel_patterns, x='species_six', y='day_night_dif', hue='diel_pattern', ax=ax, size=4,
                  palette=clrs, hue_order=hue_ordering, order=sorted_index)
    ax.set(ylabel='Day mean - night mean', xlabel='Species')
    ax.set_xticklabels(labels=sorted_index, rotation=45)
    ax = plt.axhline(0, ls='--', color='k')
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "species_diurnal-nocturnal_30min_diel-pattern{0}.png".format(dt.date.today())))
    plt.close()

    # plotting vertical
    # f, ax = plt.subplots(figsize=(5, 10))
    # sns.boxplot(data=fish_diel_patterns, x='day_night_dif', y='species_six', palette=row_cols, ax=ax, order=sorted_index,
    #             fliersize=0)
    # sns.stripplot(data=fish_diel_patterns, x='day_night_dif', y='species_six', hue='diel_pattern', ax=ax, size=4,
    #               palette=clrs, hue_order=hue_ordering, order=sorted_index)
    # ax.set(xlabel='Day mean - night mean', ylabel='Species')
    # ax = plt.axvline(0, ls='--', color='k')
    # ax.set_xticks(rotation=45)
    # plt.tight_layout()


def plot_cre_dawn_dusk_strip_v(rootdir, all_feature_combined, feature):

    sorted_index = all_feature_combined.groupby(by='species_six').mean().sort_values(by='peak_amplitude').index
    grped_bplot = sns.catplot(y='species_six',
                              x='peak_amplitude',
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
    grped_bplot = sns.stripplot(y='species_six',
                                x='peak_amplitude',
                                hue='twilight',
                                jitter=True,
                                dodge=True,
                                marker='o',
                                data=all_feature_combined,
                                order=sorted_index,
                                palette="flare")
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "species_crepuscularity_{0}_{1}.png".format(feature, dt.date.today())))



def plot_cre_dawn_dusk_strip_box(rootdir, cres_peaks_i):
    """ Plot the crepuscular data as a strip and box plot

    :param rootdir:
    :param all_feature_combined:
    :param feature:
    :return:
    """
    sorted_index = cres_peaks_i.groupby(by='species_six').mean().sort_values(by='peak_amplitude').index

    grped_bplot = sns.catplot(x='species_six',
                              y='peak_amplitude',
                              kind="strip",
                              legend=False,
                              height=3,
                              aspect=4,
                              hue='twilight',
                              data=cres_peaks_i,
                              order=sorted_index,
                              palette="flare")
    grped_bplot.set_xticklabels(labels=sorted_index, rotation=45)
    grped_bplot.set(ylabel='Peak amplitude from baseline', xlabel='Species')
    ax = plt.axhline(0, ls='--', color='k')
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "species_crepuscular_30min_strip_{0}.png".format(dt.date.today())))

    grped_bplot = sns.catplot(x='species_six',
                              y='peak_amplitude',
                              kind="box",
                              legend=False,
                              height=5,
                              aspect=2,
                              data=cres_peaks_i,
                              fliersize=0,
                              boxprops=dict(alpha=.3),
                              order=sorted_index,
                              palette="flare")
    grped_bplot.set_xticklabels(labels=sorted_index, rotation=45)
    grped_bplot.set(ylabel='Peak amplitude from baseline', xlabel='Species')
    ax = plt.axhline(0, ls='--', color='k')

    grped_bplot = sns.stripplot(x='species_six',
                                y='peak_amplitude',
                                hue='twilight',
                                data=cres_peaks_i,
                                order=sorted_index,
                                palette="flare",
                                size=3)
    grped_bplot.set_xticklabels(labels=sorted_index, rotation=45)
    grped_bplot.set(ylabel='Peak amplitude from baseline', xlabel='Species')
    ax = plt.axhline(0, ls='--', color='k')
    plt.tight_layout()

    plt.savefig(os.path.join(rootdir, "species_crepuscular_30min_box_{0}.png".format(dt.date.today())))


    # for one fish
    # g = sns.catplot(x='species_six', y='peak_amplitude', data=all_feature_combined.loc[all_feature_combined.species_six == 'Neosav'],
    #                 hue='peak',  palette='vlag', col="twilight", legend=False)
    # for axes in g.axes.flat:
    #     _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=90)
    # plt.tight_layout()
