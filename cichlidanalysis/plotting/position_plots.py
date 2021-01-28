import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.linear_model import LinearRegression


def spd_vs_y(meta, fish_tracks_30m, fish_IDs, rootdir):
    # speed vs Y position, for each fish, for combine fish of species, separated between day and night

    for idx, species in enumerate(meta.loc["species"].unique()):
        # vector for combining all fish of species x
        vp_sp_30m, sf_sp_30m = [], []
        vp_sp_30m_day, sf_sp_30m_day = [], []
        vp_sp_30m_night, sf_sp_30m_night = [], []

        for fish in fish_IDs:
            # 30min binned time resolution
            vertical_pos_30m = fish_tracks_30m[fish_tracks_30m.FishID == fish].vertical_pos

            speed_fish_30m = fish_tracks_30m[fish_tracks_30m.FishID == fish].speed_mm
            full_data_30m = ~np.isnan(vertical_pos_30m) & ~np.isnan(speed_fish_30m)

            vp_30m = vertical_pos_30m[full_data_30m]
            sf_30m = speed_fish_30m[full_data_30m]

            vp_sp_30m = np.concatenate([vp_sp_30m, vp_30m])
            sf_sp_30m = np.concatenate([sf_sp_30m, sf_30m])

            # #### 30min binned time resolution ##### #
            # get vertical position for day and night for fish
            vertical_pos_30m_day = fish_tracks_30m[(fish_tracks_30m.FishID == fish) &
                                                   (fish_tracks_30m.daynight == 'd')].vertical_pos

            speed_fish_30m_day = fish_tracks_30m[(fish_tracks_30m.FishID == fish) &
                                                 (fish_tracks_30m.daynight == 'd')].speed_mm
            full_data_30m_day = ~np.isnan(vertical_pos_30m_day) & ~np.isnan(speed_fish_30m_day)

            vp_sp_30m_day = np.concatenate([vp_sp_30m_day, vertical_pos_30m_day[full_data_30m_day]])
            sf_sp_30m_day = np.concatenate([sf_sp_30m_day, speed_fish_30m_day[full_data_30m_day]])

            vertical_pos_30m_night = fish_tracks_30m[(fish_tracks_30m.FishID == fish) &
                                                     (fish_tracks_30m.daynight == 'n')].vertical_pos

            speed_fish_30m_night = fish_tracks_30m[(fish_tracks_30m.FishID == fish) &
                                                   (fish_tracks_30m.daynight == 'n')].speed_mm
            full_data_30m_night = ~np.isnan(vertical_pos_30m_night) & ~np.isnan(speed_fish_30m_night)

            vp_sp_30m_night = np.concatenate([vp_sp_30m_night, vertical_pos_30m_night[full_data_30m_night]])
            sf_sp_30m_night = np.concatenate([sf_sp_30m_night, speed_fish_30m_night[full_data_30m_night]])

        # for day and night for 30m data
        fig2, ax2 = plt.subplots(1, 1)
        ax2.plot(sf_sp_30m_day, vp_sp_30m_day, linestyle='', marker='o', markersize=1, color='r', alpha=0.25, label='Day')
        ax2.plot(sf_sp_30m_night, vp_sp_30m_night, linestyle='', marker='o', markersize=1, color='b', alpha=0.25, label='Night')
        ax2.set_xlim([0, 60])
        ax2.set_ylim([0, 1])
        plt.xlabel("Speed mm/s")
        plt.ylabel("Vertical position")
        ax2.legend()
        fig2.suptitle(species)
        fig2.savefig(os.path.join(rootdir, "spd_vs_y_30min_DN.png"))


# # split data into day and night
# position_night_x = horizontal_pos.iloc[np.where(change_times_s[0] > tv_24h_sec)[0], ]
# position_night_x = horizontal_pos.iloc[np.where(tv_24h_sec[0:-1] > change_times_s[3])[0], ]
#
# position_night_y = vertical_pos.iloc[np.where(change_times_s[0] > tv_24h_sec)[0], ]
# position_night_y = vertical_pos.iloc[np.where(tv_24h_sec[0:-1] > change_times_s[3])[0], ]
#
# position_day_x = horizontal_pos.iloc[np.where((change_times_s[0] < tv_24h_sec[0:-1]) &
#                                               (tv_24h_sec[0:-1] < change_times_s[3]))[0], ]
# position_day_y = vertical_pos.iloc[np.where((change_times_s[0] < tv_24h_sec[0:-1]) &
#                                             (tv_24h_sec[0:-1] < change_times_s[3]))[0], ]
#
# # need to clean up data between fish, either use the vertical_pos/ horizontal_pos, or scale by x/ylim for x_nt, y_nt
# individuals = True
# fig1, ax1 = plt.subplots(2, len(meta.loc["species"].unique()))
# for idx, species in enumerate(meta.loc["species"].unique()):
#     position_day_x_sub = position_day_x.loc[:, (meta.loc["species"] == species)].to_numpy()
#     position_day_y_sub = position_day_y.loc[:, (meta.loc["species"] == species)].to_numpy()
#     position_night_x_sub = position_night_x.loc[:, (meta.loc["species"] == species)].to_numpy()
#     position_night_y_sub = position_night_y.loc[:, (meta.loc["species"] == species)].to_numpy()
#
#     if individuals:
#         fig2, ax2 = plt.subplots(2,  position_day_x_sub.shape[1])
#         fig2.suptitle("Individual fish averages for {}".format(species))
#         for individ in np.arange(0, position_day_x_sub.shape[1]):
#             position_day_xy, xedges_day, yedges_day, _ = plt.hist2d(position_day_x_sub[:, individ],
#                                                                     position_day_y_sub[:, individ],
#                                                                     bins=[3, 10], cmap='inferno', range=[[0, 1], [0, 1]])
#             position_night_xy, xedges_night, yedges_night, _ = plt.hist2d(
#                 position_night_x_sub[:, individ],
#                 position_night_y_sub[:, individ],
#                 bins=[3, 10], cmap='inferno', range=[[0, 1], [0, 1]])
#
#             # ax2[0, individ].set_title(individ)
#             ax2[0, individ].imshow(position_day_xy.T)
#             ax2[0, individ].invert_yaxis()
#             ax2[0, individ].get_xaxis().set_ticks([])
#             ax2[0, individ].get_yaxis().set_ticks([])
#             ax2[1, individ].clear()
#             ax2[1, individ].imshow(position_night_xy.T)
#             ax2[1, individ].get_xaxis().set_ticks([])
#             ax2[1, individ].get_yaxis().set_ticks([])
#             ax2[1, individ].invert_yaxis()
#             if individ == 0:
#                 ax2[0, individ].set_ylabel("Day")
#                 ax2[1, individ].set_ylabel("Night")
#         plt.savefig(os.path.join(rootdir, "xy_ave_DN_individuals_{0}.png".format(species_f.replace(' ', '-'))))
#
#     else:
#     # reshape all the data
#         position_day_x_sub = np.reshape(position_day_x_sub, position_day_x_sub.shape[0] * position_day_x_sub.shape[1])
#         position_day_y_sub = np.reshape(position_day_y_sub, position_day_y_sub.shape[0] * position_day_y_sub.shape[1])
#         position_night_x_sub = np.reshape(position_night_x_sub, position_night_x_sub.shape[0] * position_night_x_sub.shape[1])
#         position_night_y_sub = np.reshape(position_night_y_sub, position_night_y_sub.shape[0] * position_night_y_sub.shape[1])
#
#     # Creating bins
#     x_min = 0
#     x_max = np.nanmax(position_day_x_sub)
#
#     y_min = 0
#     y_max = np.nanmax(position_day_y_sub)
#
#     x_bins = np.linspace(x_min, x_max, 4)
#     y_bins = np.linspace(y_min, y_max, 11)
#
#     fig3 = plt.figure(figsize=(4, 4))
#     position_day_xy, xedges_day, yedges_day, _ = plt.hist2d(position_day_x_sub[~np.isnan(position_day_x_sub)],
#                                                             position_day_y_sub[~np.isnan(position_day_y_sub)],
#                      cmap='inferno', bins=[x_bins, y_bins])
#     plt.close(fig3)
#
#     # need to properly normalise by counts! To get frequency!!!!!!!!
#     position_day_xy = (position_day_xy / sum(sum(position_day_xy)))*100
#     fig3 = plt.figure(figsize=(4, 4))
#     plt.imshow(position_day_xy.T, cmap='inferno', vmin=0, vmax=25)
#     plt.title("Day")
#     cbar = plt.colorbar(label="% occupancy")
#     plt.gca().invert_yaxis()
#     plt.gca().set_xticks([])
#     plt.gca().set_yticks([])
#     plt.savefig(os.path.join(rootdir, "xy_ave_Day_{0}.png".format(species_f.replace(' ', '-'))))
#
#     fig4 = plt.figure(figsize=(4, 4))
#     position_night_xy, xedges_night, yedges_night, _ = plt.hist2d(position_night_x_sub[~np.isnan(position_night_x_sub)],
#                                                                   position_night_y_sub[~np.isnan(position_night_y_sub)],
#                     bins=[3, 10], cmap='inferno')
#     plt.close(fig4)
#
#     position_night_xy = (position_night_xy / sum(sum(position_night_xy)))*100
#     fig4 = plt.figure(figsize=(4, 4))
#     plt.imshow(position_night_xy.T, cmap='inferno', vmin=0, vmax=25)
#     plt.title("Night")
#     cbar = plt.colorbar(label="% occupancy")
#     plt.gca().invert_yaxis()
#     plt.gca().set_xticks([])
#     plt.gca().set_yticks([])
#     plt.savefig(os.path.join(rootdir, "xy_ave_Night_{0}.png".format(species_f.replace(' ', '-'))))
#
#     # find better way to deal with lack of second dimension when only one species
#     if len(meta.loc["species"].unique()) == 1:
#         ax1[0].set_title(species)
#         ax1[0].set_ylabel("Day")
#         ax1[0].imshow(position_day_xy.T, cmap='inferno')
#         ax1[0].invert_yaxis()
#         ax1[0].get_xaxis().set_ticks([])
#         ax1[0].get_yaxis().set_ticks([])
#         ax1[1].clear()
#         ax1[1].imshow(position_night_xy.T, cmap='inferno')
#         ax1[1].get_xaxis().set_ticks([])
#         ax1[1].get_yaxis().set_ticks([])
#         ax1[1].invert_yaxis()
#         ax1[1].set_ylabel("Night")
#     else:
#         ax1[0, idx].title(species)
#         ax1[0, idx].set_ylabel("Day")
#         ax1[0, idx].imshow(position_day_xy.T)
#         ax1[0, idx].invert_yaxis()
#         ax1[0, idx].get_xaxis().set_ticks([])
#         ax1[0, idx].get_yaxis().set_ticks([])
#         ax1[1, idx].clear()
#         ax1[1, idx].imshow(position_night_xy.T)
#         ax1[1, idx].get_xaxis().set_ticks([])
#         ax1[1, idx].get_yaxis().set_ticks([])
#         ax1[1, idx].invert_yaxis()
#         ax1[1, idx].set_ylabel("Night")
# fig1.savefig(os.path.join(rootdir, "xy_ave_DN_all.png"))