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
from cichlidanalysis.utils.species_names import shorten_sp_name
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


feature_v = load_feature_vectors(rootdir, "*als_fv.csv")

# add species
feature_v['species'] = 'undefined'
for id_n, id in enumerate(feature_v.fish_ID):
    feature_v.loc[id_n, 'species'] = extract_meta(id)['species']
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
        averages = pd.concat([averages, average], axis=1)
    stdv = sp_subset.std(axis=0)

averages_norm = averages.div(averages.sum(axis=1), axis=0)
averages_norm = averages_norm.drop(['move_median_d', 'move_median_n'])

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

fig = sns.clustermap(averages_norm.T, figsize=(7, 5), col_cluster=False, method='complete')
plt.savefig(os.path.join(rootdir, "cluster_map_fv_{0}.png".format(datetime.date.today())))
