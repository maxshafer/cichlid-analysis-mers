import os

import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt


def plot_day_night_species(rootdir, fish_diel_patterns):
    """ Plots the individual fish diurnality as scatter and bar plot. Colouring indicates the speeecies level call

    :return:
    """
    # row colours
    row_cols = []
    subset = fish_diel_patterns.loc[:, ['species_six', 'species_diel_pattern']].drop_duplicates(subset=["species_six"])
    for index, row in subset.iterrows():
        if row['species_diel_pattern'] == 'diurnal':
            # row_cols.append(sns.color_palette()[1])
            row_cols.append((255 / 255, 224 / 255, 179 / 255))
        elif row['species_diel_pattern'] == 'nocturnal':
            # row_cols.append(sns.color_palette()[0])
            row_cols.append((153 / 255, 204 / 255, 255 / 255))
        elif row['species_diel_pattern'] == 'undefined':
            # row_cols.append(sns.color_palette()[2])
            row_cols.append((179 / 255, 230 / 255, 179 / 255))

    clrs = [(sns.color_palette()[1]), sns.color_palette()[0], sns.color_palette()[2]]
    hue_ordering = ['diurnal', 'nocturnal', 'undefined']

    # plotting
    f, ax = plt.subplots(figsize=(5, 10))
    sns.boxplot(data=fish_diel_patterns, x='day_night_dif', y='species_six', palette=row_cols, ax=ax,
                fliersize=0)
    sns.stripplot(data=fish_diel_patterns, x='day_night_dif', y='species_six', hue='diel_pattern', ax=ax, size=4,
                  palette=clrs, hue_order=hue_ordering)
    ax.set(xlabel='Day mean - night mean', ylabel='Species')
    ax = plt.axvline(0, ls='--', color='k')
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "species_diurnal-nocturnal_30min_{0}.png".format(dt.date.today())))
