import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.linear_model import LinearRegression


def spd_vs_y(meta, fish_tracks, fish_tracks_30m, fish_IDs, rootdir):
    # speed vs Y position, for each fish, for combine fish of species, separated between day and night
    # there's a lot of working and testing around in this, improve in future

    for idx, species in enumerate(meta.loc["species"].unique()):
        # vector for combining all fish of species x
        vp_sp, vp_sp_30m = [], []
        sf_sp, sf_sp_30m = [], []
        vp_sp_day, vp_sp_30m_day = [], []
        sf_sp_day, sf_sp_30m_day = [], []
        vp_sp_night, vp_sp_30m_night = [], []
        sf_sp_night, sf_sp_30m_night = [], []

        for fish in fish_IDs:
            # # full time resolution
            # vertical_pos = fish_tracks[fish_tracks.FishID == fish].y_nt
            # vertical_pos -= fish_tracks[fish_tracks.FishID == fish].y_nt.min()
            # vertical_pos /= fish_tracks[fish_tracks.FishID == fish].y_nt.max()
            # # flip Y axis
            # vertical_pos = abs(1 - vertical_pos)
            # speed_fish = fish_tracks[fish_tracks.FishID == fish].speed_mm
            # full_data = ~np.isnan(vertical_pos) & ~np.isnan(speed_fish)
            #
            # vp = vertical_pos[full_data]
            # sf = speed_fish[full_data]
            #
            # # down sample data as sns too slow to properly plot all data
            # vp_ds = vp[::100]
            # sf_ds = sf[::100]
            #
            # vp_sp = np.concatenate([vp_sp, vp_ds])
            # sf_sp = np.concatenate([sf_sp, sf_ds])

            # fig4 = plt.figure(figsize=(4, 4))
            # sns.kdeplot(sf_ds, vp_ds, cmap="Reds", fill=True, levels=20)
            # plt.xlim([0, 150])
            # plt.ylim([0, 1])
            # plt.title(fish, fontsize=8)

            # 30min binned time resolution
            vertical_pos_30m = fish_tracks_30m[fish_tracks_30m.FishID == fish].y_nt
            # use fish_tracks[fish_tracks.FishID == fish].y_nt so that it iss scaled on the same area
            vertical_pos_30m -= fish_tracks[fish_tracks.FishID == fish].y_nt.min()
            vertical_pos_30m /= fish_tracks[fish_tracks.FishID == fish].y_nt.max()

            # flip Y axis
            vertical_pos_30m = abs(1 - vertical_pos_30m)
            speed_fish_30m = fish_tracks_30m[fish_tracks_30m.FishID == fish].speed_mm
            full_data_30m = ~np.isnan(vertical_pos_30m) & ~np.isnan(speed_fish_30m)

            vp_30m = vertical_pos_30m[full_data_30m]
            sf_30m = speed_fish_30m[full_data_30m]

            vp_sp_30m = np.concatenate([vp_sp_30m, vp_30m])
            sf_sp_30m = np.concatenate([sf_sp_30m, sf_30m])

            # # seems too slow to properly plot all data
            # fig4 = plt.figure(figsize=(4, 4))
            # sns.kdeplot(sf_30m, vp_30m, cmap="Blues", fill=True, levels=20)
            # plt.plot(sf_30m, vp_30m, linestyle='', marker='o', markersize=0.3, color='r')
            # plt.xlim([0, 60])
            # plt.ylim([0, 1])
            # plt.title(fish, fontsize=8)

            # # calculate simple linear regression
            # vp_30m = vp_30m.values.reshape(-1, 1)
            # model = LinearRegression().fit(vp_30m, sf_30m)
            # r_sq = model.score(vp_30m, sf_30m)
            # y = np.linspace(0, 1, 11)
            # x = y * model.coef_ + model.intercept_
            # plt.plot(x, y, 'k')

            # # full time resolution
            # vertical_pos_night = fish_tracks[(fish_tracks.FishID == fish) & (fish_tracks.daynight == 'night')].y_nt
            # # use fish_tracks[fish_tracks.FishID == fish].y_nt so that it iss scaled on the same area
            # vertical_pos_night -= fish_tracks[fish_tracks.FishID == fish].y_nt.min()
            # vertical_pos_night /= fish_tracks[fish_tracks.FishID == fish].y_nt.max()
            #
            # # flip Y axis
            # vertical_pos_night = abs(1 - vertical_pos_night)
            # speed_fish_night = fish_tracks[(fish_tracks.FishID == fish) & (fish_tracks.daynight == 'night')].speed_mm
            # full_data = ~np.isnan(vertical_pos_night) & ~np.isnan(speed_fish_night)
            #
            # # down sample data as sns too slow to properly plot all data
            # vp_night = vertical_pos_night[full_data][::100]
            # sf_night = speed_fish_night[full_data][::100]
            #
            # vp_sp_night = np.concatenate([vp_sp_night, vp_night])
            # sf_sp_night = np.concatenate([sf_sp_night, sf_night])
            #
            # vertical_pos_day = fish_tracks[(fish_tracks.FishID == fish) & (fish_tracks.daynight == 'day')].y_nt
            # # use fish_tracks[fish_tracks.FishID == fish].y_nt so that it iss scaled on the same area
            # vertical_pos_day -= fish_tracks[fish_tracks.FishID == fish].y_nt.min()
            # vertical_pos_day /= fish_tracks[fish_tracks.FishID == fish].y_nt.max()
            #
            # # flip Y axis
            # vertical_pos_day = abs(1 - vertical_pos_day)
            # speed_fish_day = fish_tracks[(fish_tracks.FishID == fish) & (fish_tracks.daynight == 'day')].speed_mm
            # full_data = ~np.isnan(vertical_pos_day) & ~np.isnan(speed_fish_day)
            #
            # # down sample data as sns too slow to properly plot all data
            # vp_day = vertical_pos_day[full_data][::100]
            # sf_day = speed_fish_day[full_data][::100]
            #
            # vp_sp_day = np.concatenate([vp_sp_day, vp_day])
            # sf_sp_day = np.concatenate([sf_sp_day, sf_day])

            # #### 30min binned time resolution ##### #
            vertical_pos_30m_day = fish_tracks_30m[(fish_tracks_30m.FishID == fish) &
                                                   (fish_tracks_30m.daynight == 'day')].y_nt
            # use fish_tracks[fish_tracks.FishID == fish].y_nt so that it iss scaled on the same area
            vertical_pos_30m_day -= fish_tracks[fish_tracks.FishID == fish].y_nt.min()
            vertical_pos_30m_day /= fish_tracks[fish_tracks.FishID == fish].y_nt.max()

            # flip Y axis
            vertical_pos_30m_day = abs(1 - vertical_pos_30m_day)
            speed_fish_30m_day = fish_tracks_30m[(fish_tracks_30m.FishID == fish) &
                                                 (fish_tracks_30m.daynight == 'day')].speed_mm
            full_data_30m_day = ~np.isnan(vertical_pos_30m_day) & ~np.isnan(speed_fish_30m_day)

            vp_sp_30m_day = np.concatenate([vp_sp_30m_day, vertical_pos_30m_day[full_data_30m_day]])
            sf_sp_30m_day = np.concatenate([sf_sp_30m_day, speed_fish_30m_day[full_data_30m_day]])


            vertical_pos_30m_night = fish_tracks_30m[(fish_tracks_30m.FishID == fish) &
                                                     (fish_tracks_30m.daynight == 'night')].y_nt
            # use fish_tracks[fish_tracks.FishID == fish].y_nt so that it iss scaled on the same area
            vertical_pos_30m_night -= fish_tracks[fish_tracks.FishID == fish].y_nt.min()
            vertical_pos_30m_night /= fish_tracks[fish_tracks.FishID == fish].y_nt.max()

            # flip Y axis
            vertical_pos_30m_night = abs(1 - vertical_pos_30m_night)
            speed_fish_30m_night = fish_tracks_30m[(fish_tracks_30m.FishID == fish) &
                                                   (fish_tracks_30m.daynight == 'night')].speed_mm
            full_data_30m_night = ~np.isnan(vertical_pos_30m_night) & ~np.isnan(speed_fish_30m_night)

            vp_sp_30m_night = np.concatenate([vp_sp_30m_night, vertical_pos_30m_night[full_data_30m_night]])
            sf_sp_30m_night = np.concatenate([sf_sp_30m_night, speed_fish_30m_night[full_data_30m_night]])


        # fig4 = plt.figure(figsize=(4, 4))
        # sns.kdeplot(sf_sp, vp_sp, cmap="Reds", fill=True, levels=20)
        # plt.xlim([0, 150])
        # plt.ylim([0, 1])
        # plt.title(species, fontsize=8)
        # fig4.savefig(os.path.join(rootdir, "spd_vs_y_all.png"))

        # # for full days, 30min
        # fig5 = plt.figure(figsize=(4, 4))
        # sns.kdeplot(sf_sp_30m, vp_sp_30m, cmap="Blues", fill=True, levels=20)
        # plt.plot(sf_sp_30m, vp_sp_30m, linestyle='', marker='o', markersize=0.3, color='r')
        # plt.xlim([0, 60])
        # plt.ylim([0, 1])
        # plt.title(species, fontsize=8)
        # fig5.savefig(os.path.join(rootdir, "spd_vs_y_30min.png"))

        # # for day and night all
        # fig2, ax2 = plt.subplots(1, 2)
        # sns.kdeplot(sf_sp_day, vp_sp_day, cmap="Reds", fill=True, levels=20, ax=ax2[0])
        # sns.kdeplot(sf_sp_night, vp_sp_night, cmap="Reds", fill=True, levels=20, ax=ax2[1])
        # ax2[0].title.set_text('Day')
        # ax2[1].title.set_text('Night')
        # for i in np.arange(ax2.shape[0]):
        #     ax2[i].set_xlim([0, 150])
        #     ax2[i].set_ylim([0, 1])
        # fig2.suptitle(species)
        # fig2.savefig(os.path.join(rootdir, "spd_vs_y_all_day_vs_night.png"))

        # # for day and night all and then 30m
        # fig2, ax2 = plt.subplots(1, 2)
        # sns.kdeplot(sf_sp_30m_day, vp_sp_30m_day, cmap="Blues", fill=True, levels=20, ax=ax2[0])
        # ax2[0].plot(sf_sp_30m_day, vp_sp_30m_day, linestyle='', marker='o', markersize=0.3, color='r')
        # sns.kdeplot(sf_sp_30m_night, vp_sp_30m_night, cmap="Blues", fill=True, levels=20, ax=ax2[1])
        # ax2[1].plot(sf_sp_30m_night, vp_sp_30m_night, linestyle='', marker='o', markersize=0.3, color='r')
        # ax2[0].title.set_text('Day')
        # ax2[1].title.set_text('Night')
        # for i in np.arange(ax2.shape[0]):
        #     ax2[i].set_xlim([0, 60])
        #     ax2[i].set_ylim([0, 1])
        # fig2.suptitle(species)
        # fig2.savefig(os.path.join(rootdir, "spd_vs_y_30min.png"))

        # for day and night all and then 30m
        fig2, ax2 = plt.subplots(1, 1)
        # sns.kdeplot(sf_sp_30m_day, vp_sp_30m_day, cmap="Reds", fill=True, levels=20, ax=ax2, alpha=0.5)
        ax2.plot(sf_sp_30m_day, vp_sp_30m_day, linestyle='', marker='o', markersize=1, color='r', alpha=0.25, label='Day')
        # sns.kdeplot(sf_sp_30m_night, vp_sp_30m_night, cmap="Blues", fill=True, levels=20, ax=ax2, alpha=0.5)
        ax2.plot(sf_sp_30m_night, vp_sp_30m_night, linestyle='', marker='o', markersize=1, color='b', alpha=0.25, label='Night')
        ax2.set_xlim([0, 60])
        ax2.set_ylim([0, 1])
        plt.xlabel("Speed mm/s")
        plt.ylabel("Vertical position")
        ax2.legend()
        fig2.suptitle(species)
        fig2.savefig(os.path.join(rootdir, "spd_vs_y_30min_DN.png"))

        # #### plotting normalised by row speed vs y position row. ####
        # xedges = np.arange(0, 1.1, 0.1)
        # yedges = np.arange(0, 70, 5)
        # H_day, _, _ = np.histogram2d(vp_sp_day, sf_sp_day, bins=(xedges, yedges))
        # H_norm_rows_day = H_day / H_day.max(axis=1, keepdims=True)
        #
        # H_night, _, _ = np.histogram2d(vp_sp_night, sf_sp_night, bins=(xedges, yedges))
        # H_norm_rows_night = H_night / H_night.max(axis=1, keepdims=True)
        #
        # fig2, ax2 = plt.subplots(2, 2)
        # ax2[0, 0].pcolormesh(H_day)
        # ax2[0, 1].pcolormesh(H_night)
        # ax2[1, 0].pcolormesh(H_norm_rows_day)
        # ax2[1, 1].pcolormesh(H_norm_rows_night)
        #
        # ax2[0, 0].title.set_text('Day')
        # ax2[0, 1].title.set_text('Night')
        # fig2.suptitle(species)
