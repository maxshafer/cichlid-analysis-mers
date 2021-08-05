import os

import numpy as np
import pandas as pd


def hist_feature_rest(rootdir, fish_tracks, species, feature='vertical_pos', bins=np.arange(0, 1.1, 0.1)):
    """ This funtion gives histograms of feature data for rest and non-rest for each fish and saves out a csv

    :param rootdir:
    :param fish_tracks:
    :param feature:
    :return:
    """
    bins = np.round(bins, 2).tolist()
    fishes = fish_tracks['FishID'].unique()
    first = True
    for fish in fishes:
        hist_r, _ = np.histogram(a=fish_tracks.loc[(fish_tracks["rest"] == 0) & (fish_tracks["FishID"] == fish),
                                                   feature], bins=bins)
        hist_nr, _ = np.histogram(a=fish_tracks.loc[(fish_tracks["rest"] == 1) & (fish_tracks["FishID"] == fish),
                                                    feature], bins=bins)
        hist_r_norm = hist_r/ np.sum(hist_r)
        hist_nr_norm = hist_nr/ np.sum(hist_nr)
        df = pd.DataFrame(data=np.transpose(np.vstack((hist_r_norm, hist_nr_norm))), index=[bins[0:-1]],
                          columns=['non_rest', 'rest'])
        df['FishID'] = fish
        df = df.reset_index().set_index('FishID').rename(columns={'level_0': 'bin'})

        if first:
            df_fishes = df
            first = False
        else:
            df_fishes = pd.concat([df_fishes, df])
    df_fishes = df_fishes.reset_index().rename(columns={'level_0': 'FishID'})
    df_fishes.to_csv(os.path.join(rootdir, "{}_als_{}_hist_rest-non-rest.csv".format(species, feature)))

    # sns.barplot(data=df_fishes, x='bin', y='non_rest', hue='FishID')
    # df_fishes.pivot('FishID', 'bin', 'rest')
    # fig1, ax1 = plt.subplots()
    # sns.heatmap(df_fishes.pivot('FishID', 'bin', 'rest').T.iloc[::-1], yticklabels = (np.round(bins[0:-1], 2).tolist())[::-1])

    return df_fishes

