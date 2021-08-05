from tkinter.filedialog import askdirectory
from tkinter import *
import warnings
import time
import os

import numpy as np

from cichlidanalysis.io.als_files import load_vertical_rest_als_files
from cichlidanalysis.analysis.processing import add_col

# debug pycharm problem
warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == '__main__':
    print("Running vertical_+position_als.py which will load *als_vertical_pos_hist_rest-non-rest.csv files and "
          "make plots")
    # Allows user to select top directory and load all als files here
    root = Tk()
    root.withdraw()
    root.update()
    rootdir = askdirectory(parent=root)
    root.destroy()

    print('hi')
    vp_hist = load_vertical_rest_als_files(rootdir)


import matplotlib.pyplot as plt
import seaborn as sns

from cichlidanalysis.utils.species_names import add_species_from_FishID


def plot_v_position_hists(vp_hist):
    vp_hist = add_species_from_FishID(vp_hist)

    bins = np.arange(0, 1.1, 0.1)
    bins = np.round(bins, 2).tolist()

    # individial fish histograms
    for state in ['rest', 'non_rest']:
        fig, ax = plt.subplots()
        plot = sns.heatmap(vp_hist.pivot('FishID', 'bin', state).T.iloc[::-1],
                           yticklabels=(np.round(bins[0:-1], 2).tolist())[::-1],
                           vmin=0, vmax=0.6,
                           cmap='Blues')
        # plot.set_xticklabels(labels=vp_hist.pivot('FishID', 'bin', 'rest').T.iloc[::-1].columns, rotation=90)
        plot.set_xticklabels(labels=vp_hist.loc[vp_hist.bin == 0, 'species_six'], rotation=90)
        ax.fig.suptitle(state)
        plt.tight_layout()

    # averaged histograms
    vp_hist_ave = vp_hist.groupby(['species_six', 'bin']).mean().reset_index()
    for state in ['rest', 'non_rest']:
        fig, ax = plt.subplots()
        plot = sns.heatmap(vp_hist_ave.pivot('bin', 'species_six', state).iloc[::-1],
                           yticklabels=(np.round(bins[0:-1], 2).tolist())[::-1],
                           vmin=0, vmax=0.6,
                           cmap='Blues')
        plot.set_xticklabels(labels=vp_hist_ave.pivot('bin', 'species_six', state).columns, rotation=90)
        ax.fig.suptitle(state)
        plt.tight_layout()

    for state in ['rest', 'non_rest']:
        individ_corr =  vp_hist.pivot('FishID', 'bin', state).T.corr()
        ax = sns.clustermap(individ_corr, figsize=(7, 5), method='single', metric='euclidean', vmin=-1, vmax=1,
                            cmap='RdBu_r', xticklabels=False, yticklabels=True)
        ax.fig.suptitle(state)




    plot = sns.heatmap(vp_hist.pivot('FishID', 'bin', 'rest').T.iloc[::-1], yticklabels=(np.round(bins[0:-1], 2).tolist())[::-1])
    fig, ax = plt.subplots()
    plot.set_xticklabels(labels=vp_hist.pivot('FishID', 'bin', 'rest').T.iloc[::-1].columns, rotation=90)
    plt.tight_layout()


    data_ave = data.groupby(['species', 'bin']).mean()
    data_ave = data_ave.reset_index()

    fig, ax = plt.subplots()
    plot = sns.heatmap(data_ave.pivot('species', 'bin', 'rest').T.iloc[::-1],
                       yticklabels=(np.round(bins[0:-1], 2).tolist())[::-1],
                       vmin=0, vmax=0.6,
                       cmap='Blues')
    plot.set_xticklabels(labels=data_ave.pivot('species', 'bin', 'rest').T.iloc[::-1].columns, rotation=90)
    plt.tight_layout()

    fig, ax = plt.subplots()
    plot = sns.heatmap(data_ave.pivot('species', 'bin', 'non_rest').T.iloc[::-1],
                       yticklabels=(np.round(bins[0:-1], 2).tolist())[::-1],
                       vmin=0, vmax=0.6,
                       cmap='Blues')
    plot.set_xticklabels(labels=data_ave.pivot('species', 'bin', 'non_rest').T.iloc[::-1].columns, rotation=90)
    plt.tight_layout()