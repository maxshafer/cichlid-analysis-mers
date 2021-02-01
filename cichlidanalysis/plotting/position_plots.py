import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.linear_model import LinearRegression

from cichlidanalysis.io.meta import check_fish_species


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


def plot_position_maps_individuals(meta, fish_tracks, rootdir):
    """ Plots individual heatmaps for day and night for each species"""

    for idx, species in enumerate(meta.loc["species"].unique()):

        fish_IDs = fish_tracks['FishID'].unique()
        # find fish of species
        fish_for_plot = check_fish_species(fish_IDs, species)

        fig2, ax2 = plt.subplots(2, len(fish_for_plot))
        fig2.suptitle("Individual fish averages for {}".format(species))

        for idx1, fish in enumerate(fish_for_plot):
            position_day_x_fish = fish_tracks.loc[(fish_tracks.daynight == 'd') & (fish_tracks.FishID == fish), ["horizontal_pos"]].to_numpy()[:,0]
            position_day_y_fish = fish_tracks.loc[(fish_tracks.daynight == 'd') & (fish_tracks.FishID == fish), ["vertical_pos"]].to_numpy()[:,0]
            position_night_x_fish = fish_tracks.loc[(fish_tracks.daynight == 'n') & (fish_tracks.FishID == fish), ["horizontal_pos"]].to_numpy()[:,0]
            position_night_y_fish = fish_tracks.loc[(fish_tracks.daynight == 'n') & (fish_tracks.FishID == fish), ["vertical_pos"]].to_numpy()[:,0]

            position_day_xy, xedges_day, yedges_day, _ = plt.hist2d(position_day_x_fish, position_day_y_fish,
                                                            bins=[3, 10], cmap='inferno', range=[[0, 1], [0, 1]])
            position_night_xy, xedges_night, yedges_night, _ = plt.hist2d(position_night_x_fish, position_night_y_fish,
                                                            bins=[3, 10], cmap='inferno', range=[[0, 1], [0, 1]])

            # normalise 2dhist
            position_day_xy = position_day_xy/sum(sum(position_day_xy))
            position_night_xy = position_night_xy/sum(sum(position_night_xy))

            ax2[0, idx1].imshow(position_day_xy.T, vmin=0, vmax=0.2)
            ax2[0, idx1].invert_yaxis()
            ax2[0, idx1].get_xaxis().set_ticks([])
            ax2[0, idx1].get_yaxis().set_ticks([])

            ax2[1, idx1].clear()
            ax2[1, idx1].imshow(position_night_xy.T, vmin=0, vmax=0.2)
            ax2[1, idx1].get_xaxis().set_ticks([])
            ax2[1, idx1].get_yaxis().set_ticks([])
            ax2[1, idx1].invert_yaxis()
            if idx1 == 0:
                ax2[0, idx1].set_ylabel("Day")
                ax2[1, idx1].set_ylabel("Night")
        # colorbar?
        plt.savefig(os.path.join(rootdir, "xy_ave_DN_individuals_{0}.png".format(species.replace(' ', '-'))))


# def plot_position_maps(meta, fish_tracks, rootdir, species_f):
#     """ Average plot for day/night for each species
#     :param meta:
#     :param fish_tracks:
#     :param rootdir:
#     :param species_f:
#     :return:
#     """
#     # split data into day and night
#     position_day_y = fish_tracks.loc[fish_tracks.daynight == 'd', ["vertical_pos", "FishID", "ts"]]
#     position_day_x = fish_tracks.loc[fish_tracks.daynight == 'd', "horizontal_pos"]
#     position_night_y = fish_tracks.loc[fish_tracks.daynight == 'n', "vertical_pos"]
#     position_night_x = fish_tracks.loc[fish_tracks.daynight == 'n', "horizontal_pos"]
#
#     for idx, species in enumerate(meta.loc["species"].unique()):
#     fig1, ax1 = plt.subplots(2, len(meta.loc["species"].unique()))
#
#         # Creating bins
#         x_min = 0
#         x_max = np.nanmax(position_day_x_sub)
#
#         y_min = 0
#         y_max = np.nanmax(position_day_y_sub)
#
#         x_bins = np.linspace(x_min, x_max, 4)
#         y_bins = np.linspace(y_min, y_max, 11)
#
#         fig3 = plt.figure(figsize=(4, 4))
#         position_day_xy, xedges_day, yedges_day, _ = plt.hist2d(position_day_x_sub[~np.isnan(position_day_x_sub)],
#                                                                 position_day_y_sub[~np.isnan(position_day_y_sub)],
#                          cmap='inferno', bins=[x_bins, y_bins])
#         plt.close(fig3)
#
#         # need to properly normalise by counts! To get frequency!!!!!!!!
#         position_day_xy = (position_day_xy / sum(sum(position_day_xy)))*100
#         fig3 = plt.figure(figsize=(4, 4))
#         plt.imshow(position_day_xy.T, cmap='inferno', vmin=0, vmax=25)
#         plt.title("Day")
#         cbar = plt.colorbar(label="% occupancy")
#         plt.gca().invert_yaxis()
#         plt.gca().set_xticks([])
#         plt.gca().set_yticks([])
#         plt.savefig(os.path.join(rootdir, "xy_ave_Day_{0}.png".format(species_f.replace(' ', '-'))))
#
#         fig4 = plt.figure(figsize=(4, 4))
#         position_night_xy, xedges_night, yedges_night, _ = plt.hist2d(position_night_x_sub[~np.isnan(position_night_x_sub)],
#                                                                       position_night_y_sub[~np.isnan(position_night_y_sub)],
#                         bins=[3, 10], cmap='inferno')
#         plt.close(fig4)
#
#         position_night_xy = (position_night_xy / sum(sum(position_night_xy)))*100
#         fig4 = plt.figure(figsize=(4, 4))
#         plt.imshow(position_night_xy.T, cmap='inferno', vmin=0, vmax=25)
#         plt.title("Night")
#         cbar = plt.colorbar(label="% occupancy")
#         plt.gca().invert_yaxis()
#         plt.gca().set_xticks([])
#         plt.gca().set_yticks([])
#         plt.savefig(os.path.join(rootdir, "xy_ave_Night_{0}.png".format(species_f.replace(' ', '-'))))
#
#         # find better way to deal with lack of second dimension when only one species
#         if len(meta.loc["species"].unique()) == 1:
#             ax1[0].set_title(species)
#             ax1[0].set_ylabel("Day")
#             ax1[0].imshow(position_day_xy.T, cmap='inferno')
#             ax1[0].invert_yaxis()
#             ax1[0].get_xaxis().set_ticks([])
#             ax1[0].get_yaxis().set_ticks([])
#             ax1[1].clear()
#             ax1[1].imshow(position_night_xy.T, cmap='inferno')
#             ax1[1].get_xaxis().set_ticks([])
#             ax1[1].get_yaxis().set_ticks([])
#             ax1[1].invert_yaxis()
#             ax1[1].set_ylabel("Night")
#         else:
#             ax1[0, idx].title(species)
#             ax1[0, idx].set_ylabel("Day")
#             ax1[0, idx].imshow(position_day_xy.T)
#             ax1[0, idx].invert_yaxis()
#             ax1[0, idx].get_xaxis().set_ticks([])
#             ax1[0, idx].get_yaxis().set_ticks([])
#             ax1[1, idx].clear()
#             ax1[1, idx].imshow(position_night_xy.T)
#             ax1[1, idx].get_xaxis().set_ticks([])
#             ax1[1, idx].get_yaxis().set_ticks([])
#             ax1[1, idx].invert_yaxis()
#             ax1[1, idx].set_ylabel("Night")
#     fig1.savefig(os.path.join(rootdir, "xy_ave_DN_all.png"))

