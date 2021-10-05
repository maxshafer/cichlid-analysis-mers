from tkinter.filedialog import askdirectory
from tkinter import *
import warnings
import os
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from cichlidanalysis.io.io_feature_vector import load_feature_vectors
from cichlidanalysis.utils.species_names import six_letter_sp_name
from cichlidanalysis.io.meta import extract_meta
from cichlidanalysis.utils.species_metrics import add_metrics, tribe_cols
from cichlidanalysis.analysis.diel_pattern import daily_more_than_pattern_individ, daily_more_than_pattern_species, \
    day_night_ratio_individ, day_night_ratio_species

# debug pycharm problem
warnings.simplefilter(action='ignore', category=FutureWarning)

# pick folder
# Allows user to select top directory and load all als files here
root = Tk()
root.withdraw()
root.update()
rootdir = askdirectory(parent=root)
root.destroy()


def subset_feature_plt(averages_i, features, labelling):
    """

    :param averages_i:
    :param features:
    :param labelling:
    :return:
    """
    # fig = plt.figure(figsize=(5, 10))
    fig = sns.clustermap(averages_i.T.loc[:, features], col_cluster=False, yticklabels=True) #, cbar_kws=dict(use_gridspec=False,location="top")
    plt.tight_layout(pad=2)
    plt.close()

feature_v = load_feature_vectors(rootdir, "*als_fv2.csv")

# add species and  species_six
feature_v['species'] = 'undefined'
feature_v['species_six'] = 'undefined'
for id_n, id in enumerate(feature_v.fish_ID):
    sp = extract_meta(id)['species']
    feature_v.loc[id_n, 'species'] = sp
    feature_v.loc[id_n, 'species_six'] = six_letter_sp_name(sp)
species = feature_v['species'].unique()


tribe_col = tribe_cols()
metrics_path = '/Users/annikanichols/Desktop/cichlid_species_database.xlsx'
sp_metrics = add_metrics(feature_v.species_six.unique(), metrics_path)

feature_v['tribe'] = 'undefined'
for species_n in feature_v.species_six.unique():
    feature_v.loc[feature_v['species_six'] == species_n, 'tribe'] = \
        sp_metrics.loc[sp_metrics['species_abbreviation'] == species_n, 'tribe'].values[0]

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

## heatmap of fv
fig1, ax1 = plt.subplots()
fig1.set_figheight(6)
fig1.set_figwidth(12)
im_spd = ax1.imshow(averages_norm.T, aspect='auto', vmin=0, cmap='magma')
ax1.get_yaxis().set_ticks(np.arange(0, len(species)))
ax1.get_yaxis().set_ticklabels(averages_norm.columns, rotation=0)
ax1.get_xaxis().set_ticks(np.arange(0, averages_norm.shape[0]))

ax1.get_xaxis().set_ticklabels(averages_norm.index, rotation=90)
plt.title('Feature vector (normalised by feature)')
fig1.tight_layout(pad=3)

# clustered heatmap of  fv
fig = sns.clustermap(averages_norm, figsize=(10, 10), col_cluster=False, method='single', yticklabels=True)
plt.savefig(os.path.join(rootdir, "cluster_map_fv_{0}.png".format(datetime.date.today())))

# # total rest
# ax = sns.catplot(data=feature_v, y='species_six', x='total_rest', kind="swarm")
fig = plt.figure(figsize=(5, 10))
ax = sns.boxplot(data=feature_v, y='species_six', x='total_rest') #, hue='tribe')
ax = sns.swarmplot(data=feature_v, y='species_six', x='total_rest', color=".2")
ax.set(xlabel='Average total rest per day', ylabel='Species')
ax.set(xlim=(0, 24))
plt.tight_layout()
ax = plt.axvline(12, ls='--', color='k')
plt.savefig(os.path.join(rootdir, "total_rest_{0}.png".format(datetime.date.today())))

# histogram of total rest
feature_v_mean = feature_v.groupby('species_six').mean()
feature_v_mean = feature_v_mean.reset_index()
feature_v_mean['tribe'] = 'undefined'
for row in feature_v_mean['species_six']:
    feature_v_mean.loc[feature_v_mean['species_six'] == row, 'tribe'] = \
        sp_metrics.loc[sp_metrics['species_abbreviation'] == row, 'tribe'].values[0]

fig = plt.figure(figsize=(5, 10))
sns.histplot(data=feature_v_mean, x='total_rest', binwidth=1, hue='tribe', multiple="stack")

# total rest vs fish length
fig = plt.figure(figsize=(5, 5))
sns.regplot(data=feature_v, x='total_rest', y='fish_length_mm')

# total rest vs day speed
feature_v['spd_max_mean'] = pd.concat([feature_v['spd_mean_day'], feature_v['spd_mean_night'],
                                       feature_v['spd_mean_predawn'], feature_v['spd_mean_dawn'],
                                       feature_v['spd_mean_dusk'], feature_v['spd_mean_postdusk']], axis=1).max(axis=1)

fig = plt.figure(figsize=(5, 5))
sns.scatterplot(data=feature_v, x='total_rest', y='spd_mean_day', hue='tribe')
fig = plt.figure(figsize=(5, 5))
sns.scatterplot(data=feature_v, x='total_rest', y='spd_mean_night', hue='tribe')
fig = plt.figure(figsize=(5, 5))
sns.regplot(data=feature_v, x='total_rest', y='spd_max_mean')

# # bout lengths rest
fig = plt.figure(figsize=(5, 10))
ax = sns.boxplot(data=feature_v, y='species_six', x='rest_bout_mean_day', fliersize=1)
ax = sns.swarmplot(data=feature_v, y='species_six', x='rest_bout_mean_day', color=".2",  size=3)
ax.set(xlim=(0, 1250))
plt.tight_layout()
plt.savefig(os.path.join(rootdir, "rest_bout_mean_day_{0}.png".format(datetime.date.today())))

fig = plt.figure(figsize=(5, 10))
ax = sns.boxplot(data=feature_v, y='species_six', x='rest_bout_mean_night', fliersize=1)
ax = sns.swarmplot(data=feature_v, y='species_six', x='rest_bout_mean_night', color=".2",  size=3)
ax.set(xlim=(0, 1250))
plt.tight_layout()
plt.savefig(os.path.join(rootdir, "rest_bout_mean_night_{0}.png".format(datetime.date.today())))

feature_v['rest_bout_mean_dn_dif'] =  feature_v['rest_bout_mean_day'] - feature_v['rest_bout_mean_night']
fig = plt.figure(figsize=(5, 10))
ax = sns.boxplot(data=feature_v, y='species_six', x='rest_bout_mean_dn_dif', fliersize=1)
ax = sns.swarmplot(data=feature_v, y='species_six', x='rest_bout_mean_dn_dif', color=".2", size=3)
plt.axvline(0, ls='--', color='k')
ax.set(xlim=(-750, 600))
plt.tight_layout()
plt.savefig(os.path.join(rootdir, "rest_bout_mean_dn_dif_{0}.png".format(datetime.date.today())))


# # bout lengths non-rest
fig = plt.figure(figsize=(5, 10))
ax = sns.boxplot(data=feature_v, y='species_six', x='nonrest_bout_mean_day', fliersize=1)
ax = sns.swarmplot(data=feature_v, y='species_six', x='nonrest_bout_mean_day', color=".2",  size=3)
ax.set(xlim=(0, 1250))
plt.tight_layout()
plt.savefig(os.path.join(rootdir, "nonrest_bout_mean_day_{0}.png".format(datetime.date.today())))

fig = plt.figure(figsize=(5, 10))
ax = sns.boxplot(data=feature_v, y='species_six', x='nonrest_bout_mean_night', fliersize=1)
ax = sns.swarmplot(data=feature_v, y='species_six', x='nonrest_bout_mean_night', color=".2",  size=3)
ax.set(xlim=(0, 1250))
plt.tight_layout()
plt.savefig(os.path.join(rootdir, "nonrest_bout_mean_night_{0}.png".format(datetime.date.today())))

feature_v['nonrest_bout_mean_dn_dif'] = feature_v['nonrest_bout_mean_day'] - feature_v['nonrest_bout_mean_night']
fig = plt.figure(figsize=(5, 10))
ax = sns.boxplot(data=feature_v, y='species_six', x='nonrest_bout_mean_dn_dif', fliersize=1)
ax = sns.swarmplot(data=feature_v, y='species_six', x='nonrest_bout_mean_dn_dif', color=".2", size=3)
plt.axvline(0, ls='--', color='k')
ax.set(xlim=(-4000, 3000))
plt.tight_layout()
plt.savefig(os.path.join(rootdir, "nonrest_bout_mean_dn_dif_{0}.png".format(datetime.date.today())))


feature_v_mean['rest_bout_mean_dn_dif'] = feature_v_mean['rest_bout_mean_day'] - feature_v_mean['rest_bout_mean_night']
feature_v_mean['nonrest_bout_mean_dn_dif'] = feature_v_mean['nonrest_bout_mean_day'] - feature_v_mean['nonrest_bout_mean_night']
fig = plt.figure(figsize=(5, 10))
sns.histplot(data=feature_v_mean, x='rest_bout_mean_dn_dif', hue='tribe', multiple="stack")
fig = plt.figure(figsize=(5, 10))
sns.histplot(data=feature_v_mean, x='rest_bout_mean_night', hue='tribe', multiple="stack")
fig = plt.figure(figsize=(5, 10))
sns.histplot(data=feature_v_mean, x='rest_bout_mean_day', hue='tribe', multiple="stack")

data_names = ['spd_mean', 'move_mean', 'rest_mean', 'y_mean', 'spd_std', 'move_std', 'rest_std', 'y_std',
              'move_bout_mean', 'nonmove_bout_mean', 'rest_bout_mean', 'nonrest_bout_mean', 'move_bout_std',
              'nonmove_bout_std', 'rest_bout_std', 'nonrest_bout_std']
time_v2_m_names = ['predawn', 'dawn', 'day', 'dusk', 'postdusk', 'night']

spd_means = ['spd_mean_predawn', 'spd_mean_dawn', 'spd_mean_day', 'spd_mean_dusk', 'spd_mean_postdusk', 'spd_mean_night']
rest_means = ['rest_mean_predawn', 'rest_mean_dawn', 'rest_mean_day', 'rest_mean_dusk', 'rest_mean_postdusk', 'rest_mean_night']
move_means = ['move_mean_predawn', 'move_mean_dawn', 'move_mean_day', 'move_mean_dusk', 'move_mean_postdusk', 'move_mean_night']
rest_b_means = ['rest_bout_mean_predawn', 'rest_bout_mean_dawn', 'rest_bout_mean_day', 'rest_bout_mean_dusk',
                'rest_bout_mean_postdusk', 'rest_bout_mean_night']
nonrest_b_means = ['nonrest_bout_mean_predawn', 'nonrest_bout_mean_dawn', 'nonrest_bout_mean_day', 'nonrest_bout_mean_dusk',
                'nonrest_bout_mean_postdusk', 'nonrest_bout_mean_night']
move_b_means = ['move_bout_mean_predawn', 'move_bout_mean_dawn', 'move_bout_mean_day', 'move_bout_mean_dusk',
             'move_bout_mean_postdusk', 'move_bout_mean_night']
nonmove_b_means = ['nonmove_bout_mean_predawn', 'nonmove_bout_mean_dawn', 'nonmove_bout_mean_day', 'nonmove_bout_mean_dusk',
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

sns.clustermap(pd.concat([day_night_ratio_fish.ratio, diel_fish.crepuscular*1], axis=1), col_cluster=False,
               yticklabels=day_night_ratio_fish.species_six, cmap='RdBu_r', vmin=0, vmax=2)
plt.tight_layout()

sns.clustermap(pd.concat([day_night_ratio_sp, diel_species.crepuscular*1], axis=1), col_cluster=False, cmap='RdBu_r', vmin=0, vmax=2)
plt.tight_layout()