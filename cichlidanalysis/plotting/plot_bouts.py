# import os
#
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.dates import DateFormatter
# from matplotlib.ticker import (MultipleLocator)
# import seaborn as sns
#
# from cichlidanalysis.plotting.single_plots import fill_plot_ts
# from cichlidanalysis.analysis.processing import norm_hist
# from cichlidanalysis.io.meta import extract_meta
#
#
# def plot_bout_lengths_dn(fish_bouts, rootdir):
#     """ Plot rest and nonrest bouts for a species
#
#     :param fish_bouts:
#     :param rootdir:
#     :return:
#     """
#     fishes = fish_bouts['FishID'].unique()
#     species = set()
#     for fish in fishes:
#         fish_data = extract_meta(fish)
#         species.add(fish_data['species'])
#
#     for species_n in species:
#         # counts of bout lengths for on and off bout
#         fig1, ax1 = plt.subplots(2, 2)
#         fish_on_bouts_d = fish_bouts.loc[(fish_bouts['FishID'] == fish) & (fish_bouts['daynight'] == 'd'), "rest_length"]
#         fish_on_bouts_n = fish_bouts.loc[(fish_bouts['FishID'] == fish) & (fish_bouts['daynight'] == 'n'), "rest_length"]
#         fish_off_bouts_d = fish_bouts.loc[(fish_bouts['FishID'] == fish) & (fish_bouts['daynight'] == 'd'), "nonrest_length"]
#         fish_off_bouts_n = fish_bouts.loc[(fish_bouts['FishID'] == fish) & (fish_bouts['daynight'] == 'n'), "nonrest_length"]
#
#         bin_boxes_on = np.arange(0, 1000, 10)
#         bin_boxes_off = np.arange(0, 60*60*10, 10)
#         counts_on_bout_len_d, _, _ = ax1[0, 0].hist(fish_on_bouts_d.dt.total_seconds(), bins=bin_boxes_on, color='red')
#         counts_on_bout_len_n, _, _ = ax1[1, 0].hist(fish_on_bouts_n.dt.total_seconds(), bins=bin_boxes_on, color='blue')
#         counts_off_bout_len_d, _, _ = ax1[0, 1].hist(fish_off_bouts_d.dt.total_seconds(), bins=bin_boxes_off, color='red')
#         counts_off_bout_len_n, _, _ = ax1[1, 1].hist(fish_off_bouts_n.dt.total_seconds(), bins=bin_boxes_off, color='blue')
#
#         ax1[0, 0].set_ylabel("Day")
#         ax1[1, 0].set_ylabel("Night")
#         ax1[1, 0].set_xlabel("Rest bout lengths")
#         ax1[1, 1].set_xlabel("Active bout lengths")
#
#         # normalised fractions of bout lengths for on and off bout
#         counts_on_bout_len_d_norm = norm_hist(counts_on_bout_len_d)
#         counts_on_bout_len_n_norm = norm_hist(counts_on_bout_len_n)
#         counts_off_bout_len_d_norm = norm_hist(counts_off_bout_len_d)
#         counts_off_bout_len_n_norm = norm_hist(counts_off_bout_len_n)
#
#         fig2, ax2 = plt.subplots(1, 2)
#         ax2[0].plot(bin_boxes_on[0:-1], counts_on_bout_len_d_norm, 'r', label='Day')
#         ax2[0].plot(bin_boxes_on[0:-1], counts_on_bout_len_n_norm, color='blueviolet', label='Night')
#         ax2[1].plot(bin_boxes_off[0:-1], counts_off_bout_len_d_norm, color='indianred', label='Day')
#         ax2[1].plot(bin_boxes_off[0:-1], counts_off_bout_len_n_norm, color='darkblue', label='Night')
#         ax2[0].set_ylabel("Fraction")
#         ax2[0].set_xlabel("Rest bouts lengths (s)")
#         ax2[1].set_xlabel("Active bouts lengths (s)")
#         ax2[0].set_xlim([-1, 400])
#         ax2[1].set_xlim([-1, 1000])
#         ax2[0].set_ylim([0, 1])
#         ax2[1].set_ylim([0, 1])
#         # ax2[0].set_yscale('log')
#         # ax2[1].set_yscale('log')
#         ax2[1].legend()
#
#         # cumulative sum of rest/active bout lengths
#         fig3, ax3 = plt.subplots(1, 2)
#         ax3[0].plot(bin_boxes_on[0:-1], np.cumsum(counts_on_bout_len_d_norm), color='red', label='Day')
#         ax3[0].plot(bin_boxes_on[0:-1], np.cumsum(counts_on_bout_len_n_norm), color='blueviolet', label='Night')
#         ax3[1].plot(bin_boxes_off[0:-1], np.cumsum(counts_off_bout_len_d_norm), color='indianred', label='Day')
#         ax3[1].plot(bin_boxes_off[0:-1], np.cumsum(counts_off_bout_len_n_norm), color='darkblue', label='Night')
#         ax3[0].set_xlabel("Bout length (s)")
#         ax3[1].set_xlabel("Bout length (s)")
#         ax3[0].set_ylabel("Fraction of rest bouts")
#         ax3[1].set_ylabel("Fraction of active bouts")
#         ax3[0].set_ylim([0, 1])
#         ax3[1].set_ylim([0, 1])
#         ax3[0].set_xlim([-20, 400])
#         ax3[1].set_xlim([-20, 10000])
#         ax3[0].legend()
#         ax3[1].legend()
#         fig3.suptitle("Cumulative movement bouts for {}".format(fish), fontsize=8)
#         plt.tight_layout()
#         plt.savefig(os.path.join(rootdir, "cumsum_rest_nonrest_bouts_{0}.png".format(species_n.replace(' ', '-'))))
#
#
# # # get start time 24h distribution
# fishes = fish_bouts['FishID'].unique()
# hist_start_t = pd.Series(index=np.arange(0, 24, 1))
# for fish_n, fish in enumerate(fishes):
#     counts = fish_bouts.loc[fish_bouts['FishID'] == fish, "bout_start"].groupby(fish_bouts["bout_start"].dt.hour).count()
#     hist_c = pd.concat([hist_start_t, counts], axis=1)
# hist_start_t = hist_start_t.drop(hist_start_t.columns[0], axis=1)
# hist_start_t.columns = (fishes.tolist())
# hist_start_t = hist_start_t.fillna(0)

# def get_bout_subset_count(fish_bouts_i, measure='rest_length'):
# # geet total rest in 1 hour blocks
# fishes = fish_bouts['FishID'].unique()
# hist_start_t = pd.Series(index=np.arange(0, 24, 1))
# for fish_n, fish in enumerate(fishes):
#     counts = fish_bouts.loc[fish_bouts['FishID'] == fish, "rest_length"].groupby(fish_bouts["bout_start"].dt.hour).count()
#     hist_c = pd.concat([hist_start_t, counts], axis=1)
# hist_start_t = hist_start_t.drop(hist_start_t.columns[0], axis=1)
# hist_start_t.columns = (fishes.tolist())
# hist_start_t = hist_start_t.fillna(0)
#
#
# def get_bout_subset_sum(fish_bouts_i, measure='nonrest_length'):
#     # get sum of measure in 1 hour blocks
#     fishes = fish_bouts['FishID'].unique()
#     hist_sum = pd.Series(index=np.arange(0, 24, 1))
#     for fish_n, fish in enumerate(fishes):
#         counts = fish_bouts.loc[fish_bouts['FishID'] == fish, measure].groupby(fish_bouts["bout_start"].dt.hour).sum()
#         hist_sum = pd.concat([hist_sum, counts], axis=1)
#     hist_sum = hist_sum.drop(hist_sum.columns[0], axis=1)
#     hist_sum.columns = (fishes.tolist())

# change_times_unit = change_times_h
# plt.figure(figsize=(6, 4))
# for cols in np.arange(0, hist_c.columns.shape[0]):
#     ax = sns.lineplot(x=hist_c.index, y=(hist_c).iloc[:, cols])
# ax.axvspan(0, change_times_unit[0], color='lightblue', alpha=0.5, linewidth=0)
# ax.axvspan(change_times_unit[0], change_times_unit[1], color='wheat', alpha=0.5, linewidth=0)
# ax.axvspan(change_times_unit[2], change_times_unit[3], color='wheat', alpha=0.5, linewidth=0)
# ax.axvspan(change_times_unit[3], 24, color='lightblue', alpha=0.5, linewidth=0)
# # ax.set_ylim([0, 1])
# ax.set_xlim([0, 24])
# plt.xlabel("Time (h)")
# plt.ylabel("Counts of rest bouts")
# ax.xaxis.set_major_locator(MultipleLocator(6))
# plt.title(species_f)
# plt.savefig(os.path.join(rootdir, "movement_30min_ave_individual{0}.png".format(species_f.replace(' ', '-'))))