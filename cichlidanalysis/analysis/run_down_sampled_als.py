from tkinter.filedialog import askdirectory
from tkinter import *
import warnings
import time

import datetime as dt
import seaborn as sns

from cichlidanalysis.io.tracks import load_ds_als_files
from cichlidanalysis.utils.timings import load_timings
from cichlidanalysis.utils.species_names import shorten_sp_name
from cichlidanalysis.plotting.speed_plots import plot_spd_30min_combined
from cichlidanalysis.analysis.processing import feature_daily

# debug pycharm problem
warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':
    # pick folder
    # Allows user to select top directory and load all als files here
    root = Tk()
    root.withdraw()
    root.update()
    rootdir = askdirectory(parent=root)
    root.destroy()

    t0 = time.time()
    fish_tracks_ds = load_ds_als_files(rootdir, "*als_30m.csv")
    t1 = time.time()
    print("time to load tracks {}".format(t1-t0))

    # get each fish ID and all species
    fish_IDs = fish_tracks_ds['FishID'].unique()
    species = fish_tracks_ds['species'].unique()

    # get timings
    fps, tv_ns, tv_sec, tv_24h_sec, num_days, tv_s_type, change_times_s, change_times_ns, change_times_h, day_ns, day_s,\
        change_times_d, change_times_m = load_timings(fish_tracks_ds[fish_tracks_ds.FishID == fish_IDs[0]].shape[0])
    change_times_unit = [7*2, 7.5*2, 18.5*2, 19*2]
    change_times_datetime = [dt.datetime.strptime("1970-1-2 07:00:00", '%Y-%m-%d %H:%M:%S'),
                             dt.datetime.strptime("1970-1-2 07:30:00", '%Y-%m-%d %H:%M:%S'),
                             dt.datetime.strptime("1970-1-2 18:30:00", '%Y-%m-%d %H:%M:%S'),
                             dt.datetime.strptime("1970-1-2 19:00:00", '%Y-%m-%d %H:%M:%S'),
                             dt.datetime.strptime("1970-1-3 00:00:00", '%Y-%m-%d %H:%M:%S')]
    day_unit = dt.datetime.strptime("1970:1", "%Y:%d")

    feature, ymax, span_max, ylabeling = 'vertical_pos', 1, 0.8, 'Vertical position'
    averages_vp, date_time_obj_vp, sp_vp_combined = plot_spd_30min_combined(fish_tracks_ds, feature, ymax, span_max,
                                                                            ylabeling, change_times_datetime, rootdir)

    feature, ymax, span_max, ylabeling = 'speed_mm', 95, 80, 'Speed mm/s'
    averages_spd, date_time_obj_sp, sp_spd_combined = plot_spd_30min_combined(fish_tracks_ds, feature, ymax, span_max,
                                                                              ylabeling, change_times_datetime, rootdir)

    feature, ymax, span_max, ylabeling = 'rest', 95, 80, 'Rest'
    averages_spd, date_time_obj_sp, sp_spd_combined = plot_spd_30min_combined(fish_tracks_ds, feature, ymax, span_max,
                                                                              ylabeling, change_times_datetime, rootdir)
    aves_ave_spd = feature_daily(averages_spd)
    aves_ave_vp = feature_daily(averages_vp)
    aves_ave_rest = feature_daily(averages_spd)

    # reorganising
    species_short = shorten_sp_name(species)

    fig = sns.clustermap(aves_ave_spd.T, figsize=(7, 5), col_cluster=False, method='complete', metric='correlation')
    fig = sns.clustermap(aves_ave_vp.T, figsize=(7, 5), col_cluster=False, method='complete', metric='correlation')
    fig = sns.clustermap(aves_ave_rest.T, figsize=(7, 5), col_cluster=False, method='complete', metric='correlation')



    # # https://matplotlib.org/matplotblog/posts/create-ridgeplots-in-matplotlib/
    # cmap = cm.get_cmap('turbo')
    # colour_array = np.arange(0, 1, 1 / len(species))
    #
    # gs = grid_spec.GridSpec(len(species), 1)
    # fig = plt.figure(figsize=(16, 9))
    # # date_form = DateFormatter('%Y-%m-%d')
    # date_form = DateFormatter('%H:%M:%S')
    # ax_objs = []
    # averages = np.zeros([len(species), 303])
    #
    # first = 1
    # for species_n, species_name in enumerate(species):
    #     # get speeds for each individual for a given species
    #     spd = fish_tracks_ds[fish_tracks_ds.species == species_name][[feature, 'FishID', 'ts']]
    #     sp_spd = spd.pivot(columns='FishID', values=feature, index='ts')
    #     if first:
    #         sp_spd_combined = sp_spd
    #         first = 0
    #     else:
    #         frames = [sp_spd_combined, sp_spd]
    #         sp_spd_combined = pd.concat(frames, axis=1)
    #
    #     # calculate ave and stdv
    #     average = sp_spd.mean(axis=1)
    #     averages[species_n, :] = average[0:303]
    #     # stdv = sp_spd.std(axis=1)
    #
    #     # create time vector in datetime format
    #     # tv = fish_tracks_ds.loc[fish_tracks_ds.FishID == fish_IDs[0], 'ts']
    #     date_time_obj = []
    #     for i in sp_spd.index:
    #         date_time_obj.append(dt.datetime.strptime(i, '%Y-%m-%d %H:%M:%S'))
    #
    #     # creating new axes object
    #     ax_objs.append(fig.add_subplot(gs[species_n:species_n + 1, 0:]))
    #
    #     days_to_plot = (date_time_obj[-1] - date_time_obj[0]).days + 1
    #
    #     for day_n in range(days_to_plot):
    #         # ax_objs[-1].axvspan(dt.datetime.strptime("1970-1-2 00:00:00", '%Y-%m-%d %H:%M:%S')+timedelta(days=day_n),
    #         #                     change_times_datetime[0]+timedelta(days=day_n), color='lightblue',
    #         #                     alpha=0.5, linewidth=0)
    #         # ax_objs[-1].axvspan(change_times_datetime[0]+timedelta(days=day_n),
    #         #                     change_times_datetime[1]+timedelta(days=day_n),  color='wheat',
    #         #                     alpha=0.5, linewidth=0)
    #         # ax_objs[-1].axvspan(change_times_datetime[2]+timedelta(days=day_n), change_times_datetime[3]+timedelta
    #         # (days=day_n), color='wheat', alpha=0.5, linewidth=0)
    #         # ax_objs[-1].axvspan(change_times_datetime[3]+timedelta(days=day_n), change_times_datetime[4]+timedelta
    #         # (days=day_n), color='lightblue', alpha=0.5, linewidth=0)
    #
    #         ax_objs[-1].fill_between([dt.datetime.strptime("1970-1-2 00:00:00", '%Y-%m-%d %H:%M:%S')+timedelta(days=day_n),
    #                                   change_times_datetime[0]+timedelta(days=day_n)], [span_max, span_max], 0,
    #                                  color='lightblue', alpha=0.5, linewidth=0, zorder=1)
    #         ax_objs[-1].fill_between([change_times_datetime[0]+timedelta(days=day_n),
    #                             change_times_datetime[1]+timedelta(days=day_n)], [span_max, span_max], 0,  color='wheat',
    #                             alpha=0.5, linewidth=0)
    #         ax_objs[-1].fill_between([change_times_datetime[2]+timedelta(days=day_n), change_times_datetime[3]+timedelta
    #         (days=day_n)], [span_max, span_max], 0, color='wheat', alpha=0.5, linewidth=0)
    #         ax_objs[-1].fill_between([change_times_datetime[3]+timedelta(days=day_n), change_times_datetime[4]+timedelta
    #         (days=day_n)], [span_max, span_max], 0, color='lightblue', alpha=0.5, linewidth=0)
    #
    #     # plotting the distribution
    #     ax_objs[-1].plot(date_time_obj, average, lw=1, color='w')
    #     ax_objs[-1].fill_between(date_time_obj, average, 0, color=cmap(colour_array[species_n]), zorder=2)
    #
    #     # setting uniform x and y lims
    #     ax_objs[-1].set_xlim(min(date_time_obj), dt.datetime.strptime("1970-1-8 08:30:00", '%Y-%m-%d %H:%M:%S'))
    #     ax_objs[-1].set_ylim(0, ymax)
    #
    #     # make background transparent
    #     rect = ax_objs[-1].patch
    #     rect.set_alpha(0)
    #
    #     if species_n == len(species) - 1:
    #         ax_objs[-1].set_xlabel("Time", fontsize=10, fontweight="bold")
    #         ax_objs[-1].xaxis.set_major_locator(MultipleLocator(20))
    #         ax_objs[-1].xaxis.set_major_formatter(date_form)
    #         ax_objs[-1].yaxis.tick_right()
    #         ax_objs[-1].yaxis.set_label_position("right")
    #         ax_objs[-1].set_ylabel(ylabeling)
    #
    #     else:
    #         # remove borders, axis ticks, and labels
    #         ax_objs[-1].set_xticklabels([])
    #         ax_objs[-1].set_xticks([])
    #         ax_objs[-1].set_yticks([])
    #         ax_objs[-1].set_yticklabels([])
    #         ax_objs[-1].set_ylabel('')
    #
    #     spines = ["top", "right", "left", "bottom"]
    #     for s in spines:
    #         ax_objs[-1].spines[s].set_visible(False)
    #
    #     short_name = shorten_sp_name(species_name)
    #     shortened_sp_name = species_name[0] + ". " + species_name.split(' ')[1]
    #     ax_objs[-1].text(0.9, 0, short_name[0], fontweight="bold", fontsize=10, ha="right", rotation=-45)
    #     gs.update(hspace=-0.1)
    # plt.show()

    # # clustering
    # X = aves_ave.corr().values
    # d = sch.distance.pdist(X)   # vector of ('55' choose 2) pairwise distances
    # L = sch.linkage(d, method='complete')
    # ind = sch.fcluster(L, 0.5*d.max(), 'distance')
    # cols = [aves_ave.columns.tolist()[i] for i in list((np.argsort(ind)))]
    # aves_ave = aves_ave[cols]
    #
    # ## heatmap average
    # changes = [7, 19]
    # changes_ticks = []
    # for i in np.arange(0, 6):
    #     changes_ticks = np.append(changes_ticks, np.add(changes, 24*i))
    # changes_ticks = np.multiply(changes_ticks, 2)
    #
    # fig1, ax1 = plt.subplots()
    # fig1.set_figheight(5)
    # fig1.set_figwidth(15)
    # im_spd = ax1.imshow(averages, aspect='auto', vmin=0, vmax=70)
    # ax1.get_yaxis().set_ticks(np.arange(0, len(species)))
    # ax1.get_yaxis().set_ticklabels(species_short, rotation=45)
    # ax1.get_xaxis().set_ticks(np.arange(0, averages.shape[1], 12))
    # ax1.get_xaxis().set_ticks(changes_ticks)
    # ax1.get_xaxis().set_ticklabels(['7am', '7pm']*6)
    # cbar = fig1.colorbar(im_spd, label="Speed mm/s")
    # fig1.tight_layout(pad=2)
    #
    # ## heatmap daily average
    # fig1, ax1 = plt.subplots()
    # fig1.set_figheight(6)
    # fig1.set_figwidth(6)
    # im_spd = ax1.imshow(aves_ave.T, aspect='auto', vmin=0, vmax=50, cmap='magma')
    # ax1.get_yaxis().set_ticks(np.arange(0, len(species)))
    # sp_names = shorten_sp_name(aves_ave.columns)
    # ax1.get_yaxis().set_ticklabels(sp_names, rotation=45)
    # ax1.get_xaxis().set_ticks(np.arange(0, aves_ave.shape[1], 12))
    # ax1.get_xaxis().set_ticks(changes_ticks[0:2])
    # ax1.get_xaxis().set_ticklabels(['7am', '7pm'])
    # cbar = fig1.colorbar(im_spd, label="Speed mm/s")
    # plt.title('Daily average speed mm/s')
    # fig1.tight_layout(pad=3)
    #
    # ## heatmap individuals
    # changes = [7, 19]
    # changes_ticks = []
    # for i in np.arange(0, 6):
    #     changes_ticks = np.append(changes_ticks, np.add(changes, 24*i))
    # changes_ticks = np.multiply(changes_ticks, 2)
    #
    # fig1, ax1 = plt.subplots()
    # fig1.set_figheight(5)
    # fig1.set_figwidth(15)
    # im_spd = ax1.imshow(sp_spd_combined.T, aspect='auto', vmin=0, vmax=70)
    # ax1.get_yaxis().set_ticks(np.arange(0, len(fish_IDs)))
    # ax1.get_yaxis().set_ticklabels(sp_spd_combined.columns, rotation=45)
    # ax1.get_xaxis().set_ticks(np.arange(0, sp_spd_combined.shape[1], 12))
    # ax1.get_xaxis().set_ticks(changes_ticks)
    # ax1.get_xaxis().set_ticklabels(['7am', '7pm']*6)
    # cbar = fig1.colorbar(im_spd, label="Speed mm/s")
    # fig1.tight_layout(pad=3)
    #
