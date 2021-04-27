from tkinter.filedialog import askdirectory
from tkinter import *
import warnings
import os
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from cichlidanalysis.io.tracks import load_feature_vectors
from cichlidanalysis.utils.species_names import shorten_sp_name, six_letter_sp_name
from cichlidanalysis.io.meta import extract_meta

# debug pycharm problem
warnings.simplefilter(action='ignore', category=FutureWarning)

# pick folder
# Allows user to select top directory and load all als files here
root = Tk()
root.withdraw()
root.update()
rootdir = askdirectory(parent=root)
root.destroy()


feature_v = load_feature_vectors(rootdir, "*als_fv2.csv")

# add species and  species_six
feature_v['species'] = 'undefined'
feature_v['species_six'] = 'undefined'
for id_n, id in enumerate(feature_v.fish_ID):
    sp = extract_meta(id)['species']
    feature_v.loc[id_n, 'species'] = sp
    feature_v.loc[id_n, 'species_six'] = six_letter_sp_name(sp)
species = feature_v['species'].unique()

# make species average
for species_n, species_name in enumerate(species):
    # get speeds for each individual for a given species
    sp_subset = feature_v[feature_v.species == species_name]

    # calculate ave and stdv
    average = sp_subset.mean(axis=0)
    average = average.rename(species_name)
    if species_n == 0:
        averages = average
    else:
        averages = pd.concat([averages, average], axis=1, join='inner')
    stdv = sp_subset.std(axis=0)

averages_norm = averages.div(averages.sum(axis=1), axis=0)

## heatmap of fv
fig1, ax1 = plt.subplots()
fig1.set_figheight(6)
fig1.set_figwidth(6)
im_spd = ax1.imshow(averages_norm.T, aspect='auto', vmin=0, cmap='magma')
ax1.get_yaxis().set_ticks(np.arange(0, len(species)))
sp_names = shorten_sp_name(averages_norm.columns)
ax1.get_yaxis().set_ticklabels(sp_names, rotation=45)
ax1.get_xaxis().set_ticks(np.arange(0, averages_norm.shape[0]))

ax1.get_xaxis().set_ticklabels(averages_norm.index, rotation=90)
plt.title('Feature vector (normalised by feature)')
fig1.tight_layout(pad=3)

# # cluster by correlation
# X = averages.corr().values
# d = sch.distance.pdist(X)   # vector of ('55' choose 2) pairwise distances
# L = sch.linkage(d, method='complete')
# ind = sch.fcluster(L, 0.5*d.max(), 'distance')
# cols = [averages.columns.tolist()[i] for i in list((np.argsort(ind)))]
# averages = averages[cols]

fig = sns.clustermap(averages_norm, figsize=(10, 10), col_cluster=False, method='complete', yticklabels=True)
plt.savefig(os.path.join(rootdir, "cluster_map_fv_{0}.png".format(datetime.date.today())))

# ax = sns.heatmap(averages.iloc[averages.index == 'total_rest', :].T, yticklabels=True, vmin=0, vmax=24)
# ax = sns.scatterplot(data=averages.T, x='total_rest', y='total_rest',  hue=averages.columns)
ax = sns.catplot(data=feature_v, y='species_six', x='total_rest', kind="swarm")
ax.set(xlabel='Average total rest per day', ylabel='Species')
plt.tight_layout()
plt.savefig(os.path.join(rootdir, "total_rest_{0}.png".format(datetime.date.today())))


data_names = ['spd_mean', 'move_mean', 'rest_mean', 'y_mean', 'spd_std', 'move_std', 'rest_std', 'y_std',
              'move_bout_mean', 'nonmove_bout_mean', 'rest_bout_mean', 'nonrest_bout_mean', 'move_bout_std',
              'nonmove_bout_std', 'rest_bout_std', 'nonrest_bout_std']
time_v2_m_names = ['predawn', 'dawn', 'day', 'dusk', 'postdusk', 'night']


spd_means = ['spd_mean_predawn', 'spd_mean_dawn', 'spd_mean_day', 'spd_mean_dusk', 'spd_mean_postdusk', 'spd_mean_night']
rest_means = ['rest_mean_predawn', 'rest_mean_dawn', 'rest_mean_day', 'rest_mean_dusk', 'rest_mean_postdusk', 'rest_mean_night']
move_means = ['move_mean_predawn', 'move_mean_dawn', 'move_mean_day', 'move_mean_dusk', 'move_mean_postdusk', 'move_mean_night']
rest_means = ['rest_mean_predawn', 'rest_mean_dawn', 'rest_mean_day', 'rest_mean_dusk', 'rest_mean_postdusk', 'rest_mean_night']

nonmove_means = ['nonmove_bout_mean_predawn', 'nonmove_bout_mean_dawn', 'nonmove_bout_mean_day', 'nonmove_bout_mean_dusk',
             'nonmove_bout_mean_postdusk', 'nonmove_bout_mean_night']

movement_bouts = ['move_bout_mean', 'nonmove_bout_mean', 'move_bout_std']
rest_bouts = ['rest_bout_mean', 'nonrest_bout_mean']

fig = sns.heatmap(averages.loc[spd_means, :], yticklabels=True)
fig = sns.heatmap(feature_v.loc[rest_means, :], yticklabels=True)
fig = sns.heatmap(averages.loc[move_means, :], yticklabels=True)




night = feature_v.loc[:, 'rest_mean_night']
day = feature_v.loc[:, 'rest_mean_day']
dawn_dusk = feature_v.loc[:, ['rest_mean_predawn', 'rest_mean_dawn', 'rest_mean_dusk', 'rest_mean_postdusk']].mean(axis=1)

nocturnal = (night*1.1 < day) & (night*1.1 < dawn_dusk)
dirunal = (day*1.1 < night*1.1) & (day*1.1 < dawn_dusk)
crepuscular = (dawn_dusk*1.1 < night) & (dawn_dusk*1.1 < day)

plt.imshow(pd.concat([nocturnal, dirunal, crepuscular], axis=1))
plt.yticks(np.arange(0, len(feature_v)), labels=feature_v.species)
plt.xticks(np.arange(0, 3), labels=['nocturnal', 'diurnal', 'crepuscular'], rotation=45, ha='right', va='top')
plt.tight_layout()

night = averages.loc['rest_mean_night', :]
day = averages.loc['rest_mean_day', :]
dawn_dusk = averages.loc[['rest_mean_predawn', 'rest_mean_dawn', 'rest_mean_dusk', 'rest_mean_postdusk'], :].mean(axis=0)

nocturnal = (night*1.1 < day) & (night*1.1 < dawn_dusk)
dirunal = (day*1.1 < night*1.1) & (day*1.1 < dawn_dusk)
crepuscular = (dawn_dusk*1.1 < night) & (dawn_dusk*1.1 < day)

plt.imshow(pd.concat([nocturnal, dirunal, crepuscular], axis=1))
plt.yticks(np.arange(0, len(averages.columns)), labels=averages.columns)
plt.xticks(np.arange(0, 3), labels=['nocturnal', 'diurnal', 'crepuscular'], rotation=45, ha='right', va='top')
plt.tight_layout()

# fig = sns.heatmap(pd.concat[dawn_dusk, day, night], yticklabels=True)
