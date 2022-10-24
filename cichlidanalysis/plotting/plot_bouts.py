import os

import matplotlib.pyplot as plt
import seaborn as sns


def plot_bout_lens_rest_day_night(rootdir, feature_v, diel_patterns):
    fig = plt.figure(figsize=(5, 10))
    ax = sns.boxplot(data=feature_v, y='six_letter_name_Ronco', x='rest_bout_mean_day', fliersize=1, hue="cluster",
                     order=diel_patterns.sort_values(by="cluster").species, dodge=False)
    ax = sns.swarmplot(data=feature_v, y='six_letter_name_Ronco', x='rest_bout_mean_day', color=".2", size=3,
                       order=diel_patterns.sort_values(by="cluster").species)
    ax.set(xlim=(0, 1250))
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "rest_bout_mean_day.png"))

    fig = plt.figure(figsize=(5, 10))
    ax = sns.boxplot(data=feature_v, y='six_letter_name_Ronco', x='rest_bout_mean_night', fliersize=1, hue="cluster",
                     order=diel_patterns.sort_values(by="cluster").species, dodge=False)
    ax = sns.swarmplot(data=feature_v, y='six_letter_name_Ronco', x='rest_bout_mean_night', color=".2", size=3,
                       order=diel_patterns.sort_values(by="cluster").species)
    ax.set(xlim=(0, 1250))
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "rest_bout_mean_night.png"))
    return



def plot_bout_lens_non_rest_day_night(rootdir, feature_v, diel_patterns):
    """

    :param rootdir:
    :param feature_v:
    :param diel_patterns:
    :return:
    """

    fig = plt.figure(figsize=(5, 10))
    ax = sns.boxplot(data=feature_v, y='six_letter_name_Ronco', x='nonrest_bout_mean_day', fliersize=1, hue="cluster",
                     order=diel_patterns.sort_values(by="cluster").species, dodge=False)
    ax = sns.swarmplot(data=feature_v, y='six_letter_name_Ronco', x='nonrest_bout_mean_day', color=".2", size=3,
                       order=diel_patterns.sort_values(by="cluster").species)
    ax.set(xlim=(0, 1250))
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "nonrest_bout_mean_day.png"))

    fig = plt.figure(figsize=(5, 10))
    ax = sns.boxplot(data=feature_v, y='six_letter_name_Ronco', x='nonrest_bout_mean_night', fliersize=1, hue="cluster",
                     order=diel_patterns.sort_values(by="cluster").species, dodge=False)
    ax = sns.swarmplot(data=feature_v, y='six_letter_name_Ronco', x='nonrest_bout_mean_night', color=".2", size=3,
                       order=diel_patterns.sort_values(by="cluster").species)
    ax.set(xlim=(0, 1250))
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "nonrest_bout_mean_night.png"))
    return

def plot_dn_dif_rest_bouts(rootdir, feature_v, diel_patterns):
    """

    :param rootdir:
    :param feature_v:
    :param diel_patterns:
    :return:
    """

    feature_v['rest_bout_mean_dn_dif'] = feature_v['rest_bout_mean_day'] - feature_v['rest_bout_mean_night']
    fig = plt.figure(figsize=(5, 10))
    ax = sns.boxplot(data=feature_v, y='six_letter_name_Ronco', x='rest_bout_mean_dn_dif', fliersize=1, hue="cluster",
                     order=diel_patterns.sort_values(by="cluster").species, dodge=False)
    ax = sns.swarmplot(data=feature_v, y='six_letter_name_Ronco', x='rest_bout_mean_dn_dif', color=".2", size=3,
                       order=diel_patterns.sort_values(by="cluster").species)
    plt.axvline(0, ls='--', color='k')
    ax.set(xlim=(-750, 600))
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "rest_bout_mean_dn_dif.png"))
    return


def plot_dn_dif_non_rest_bouts(rootdir, feature_v, diel_patterns):
    """

    :param rootdir:
    :param feature_v:
    :param diel_patterns:
    :return:
    """
    feature_v['nonrest_bout_mean_dn_dif'] = feature_v['nonrest_bout_mean_day'] - feature_v['nonrest_bout_mean_night']
    fig = plt.figure(figsize=(5, 10))
    ax = sns.boxplot(data=feature_v, y='six_letter_name_Ronco', x='nonrest_bout_mean_dn_dif', fliersize=1,
                     hue="cluster",
                     order=diel_patterns.sort_values(by="cluster").species, dodge=False)
    ax = sns.swarmplot(data=feature_v, y='six_letter_name_Ronco', x='nonrest_bout_mean_dn_dif', color=".2", size=3,
                       order=diel_patterns.sort_values(by="cluster").species)
    plt.axvline(0, ls='--', color='k')
    ax.set(xlim=(-4000, 3000))
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "nonrest_bout_mean_dn_dif.png"))
    return


def rest_bouts_hists(rootdir, feature_v_mean):
    """

    :param rootdir:
    :param feature_v_mean:
    :return:
    """
    fig = plt.figure(figsize=(5, 10))
    sns.histplot(data=feature_v_mean, x='rest_bout_mean_dn_dif', hue='cluster', multiple="stack")
    plt.savefig(os.path.join(rootdir, "rest_bout_mean_dn_dif_hist.png"))
    plt.close()
    fig = plt.figure(figsize=(5, 10))
    sns.histplot(data=feature_v_mean, x='rest_bout_mean_night', hue='cluster', multiple="stack")
    plt.savefig(os.path.join(rootdir, "rest_bout_mean_night_hist.png"))
    plt.close()
    fig = plt.figure(figsize=(5, 10))
    sns.histplot(data=feature_v_mean, x='rest_bout_mean_day', hue='cluster', multiple="stack")
    plt.savefig(os.path.join(rootdir, "rest_bout_mean_day_hist.png"))
    plt.close()
    return
