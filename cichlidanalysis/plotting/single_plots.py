import glob
import os
import math
import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import cv2

from cichlidanalysis.io.meta import load_yaml

def filled_plot(tv_internal, speed, change_times_unit, day_unit):
    days_to_plot = math.ceil(np.max(tv_internal)/day_unit)
    figa, ax = plt.subplots(figsize=(15, 5))
    ax.plot(tv_internal[0:-1], speed[:], color='black')
    for day_n in range(days_to_plot):
        ax.axvspan(0+day_unit*day_n, change_times_unit[0]+day_unit*day_n, color='lightblue', alpha=0.5, linewidth=0)
        ax.axvspan(change_times_unit[0]+day_unit*day_n, change_times_unit[1]+day_unit*day_n, color='wheat', alpha=0.5,
                   linewidth=0)
        ax.axvspan(change_times_unit[2]+day_unit*day_n, change_times_unit[3]+day_unit*day_n, color='wheat', alpha=0.5,
                   linewidth=0)
        ax.axvspan(change_times_unit[3]+day_unit*day_n, day_unit + day_unit * day_n, color='lightblue', alpha=0.5,
                   linewidth=0)
    ax.set_xlim([8, days_to_plot*day_unit])
    return figa, ax


def plot_hist(bin_edges_plt, data, xlabel_name, fps_plot):
    # make histogram of values and remove NaNs
    hist = np.histogram(data[~np.isnan(data)], bins=bin_edges_plt)[0]
    bin_width = (bin_edges_plt[1] - bin_edges_plt[0])

    figa, ax1_plot = plt.subplots()
    color = 'tab:red'
    ax1_plot.set_xlabel(xlabel_name)
    ax1_plot.set_ylabel('Fraction of time', color=color)
    ax1_plot.bar((bin_edges_plt[:-1] + (bin_width / 2)) / fps_plot, hist / sum(hist), width=bin_width / fps_plot,
                 color=color)
    ax1_plot.tick_params(axis='y', labelcolor=color)
    ax1_plot.set_ylim(0, 0.5)
    # ax1.set_ylim(0, 0.5)

    ax2_plot = ax1_plot.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2_plot.set_ylabel('cumsum fraction of bouts', color=color)  # we already handled the x-label with ax1
    ax2_plot.plot((bin_edges_plt[:-1]) / fps_plot, np.cumsum(hist / sum(hist)), color=color)
    ax2_plot.tick_params(axis='y', labelcolor=color)
    ax2_plot.set_ylim(0, 1)

    figa.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def plot_scatter(x_data, y_data, speed):
    # plot scatter of position coloured by speed
    plt.figure()
    plt.scatter(x_data, y_data, c=speed[:], cmap="viridis", vmax=np.nanpercentile(speed, 98))


def plot_speed(speed, tv_internal):
    plt.figure()
    plt.plot(tv_internal[0:-1], speed)


def image_minmax(rootdir, ymin, ymax, fish_ID, meta):
    """

    :param rootdir:
    :param ymin:
    :param ymax:
    :param fish_ID:
    :param meta:
    :return:
    """
    track_roi = load_yaml(rootdir, "roi_file")

    vid_paths = glob.glob(os.path.join(rootdir, "*.mp4"))
    if len(vid_paths) > 0:
        try:
            cap = cv2.VideoCapture(vid_paths[0])
        except:
            print("problem reading video file, check path")
            return
        ret, frame = cap.read()

        if track_roi:
            curr_roi = track_roi["roi_0"]
            frame = frame[curr_roi[1]:curr_roi[1] + curr_roi[3], curr_roi[0]:curr_roi[0] + curr_roi[2]]
        fig1, ax1 = plt.subplots()
        plt.imshow(frame)
        plt.plot([0, 500], [ymin, ymin], color='r')
        plt.plot([0, 500], [ymax, ymax], color='r')
        plt.savefig(os.path.join(rootdir, "{0}_ylims_{1}.png".format(fish_ID, meta["species"].replace(' ', '-'))))
        plt.close()
        cap.release()
        cv2.destroyAllWindows()
        return


def sec_axis_h(ax0, start_total_sec_i):
    # for plotting second axis on top whicch plots movie number

    def time2movie_h(x):
        return x - start_total_sec_i / 60 / 60

    def movie2time_h(x):
        return x + start_total_sec_i / 60 / 60

    ax0.xaxis.set_major_locator(MultipleLocator(6))
    ax0.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax0.xaxis.set_minor_locator(MultipleLocator(1))
    secax = ax0.secondary_xaxis('top', functions=(time2movie_h, movie2time_h))
    secax.xaxis.set_major_locator(MultipleLocator(6))
    secax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    secax.xaxis.set_minor_locator(MultipleLocator(1))
    secax.set_xlabel('Movie number')


def plot_hist_2(bin_edges_plt, data1, data1_label, data2, data2_label, xlabel_name, fps_plot):
    # make histogram of values and remove NaNs
    hist1 = np.histogram(data1[~np.isnan(data1)], bins=bin_edges_plt)[0]
    hist2 = np.histogram(data2[~np.isnan(data2)], bins=bin_edges_plt)[0]
    bin_width = (bin_edges_plt[1] - bin_edges_plt[0])

    figa, ax1_plot = plt.subplots()
    color = 'tab:red'
    ax1_plot.set_ylabel('Fraction of time', color=color)
    ax1_plot.set_xlabel(xlabel_name)
    ax1_plot.bar((bin_edges_plt[:-1] + (bin_width / 2)) / fps_plot, hist1 / sum(hist1), width=bin_width / fps_plot,
                 color=color, alpha=0.5)
    ax1_plot.bar((bin_edges_plt[:-1] + (bin_width / 2)) / fps_plot, hist2 / sum(hist2), width=bin_width / fps_plot,
                 color='b', alpha=0.5)
    ax1_plot.tick_params(axis='y', labelcolor=color)

    ax2_plot = ax1_plot.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2_plot.set_ylabel('Cumulative sum', color=color)  # we already handled the x-label with ax1
    ax2_plot.plot((bin_edges_plt[:-1]) / fps_plot, np.cumsum(hist1 / sum(hist1)), color='r', label=data1_label)
    ax2_plot.plot((bin_edges_plt[:-1]) / fps_plot, np.cumsum(hist2 / sum(hist2)), color='b', label=data2_label)
    ax2_plot.tick_params(axis='y', labelcolor=color)
    ax2_plot.set_ylim(0, 1)
    plt.legend(loc='upper right')

    figa.tight_layout()  # otherwise the right y-label is slightly clipped


def fill_plot_ts(ax, change_times_unit, tv_internal):
    if isinstance(tv_internal.iloc[-1], datetime.datetime):
        td = tv_internal.iloc[-1] - tv_internal.iloc[0]
        days = td.round('d')
        if td > days:
            days = days + '1d'
        days_to_plot = days.days + 1

        for day_n in range(days_to_plot):
            ax.axvspan(0+day_n, change_times_unit[0]+day_n, color='lightblue', alpha=0.5, linewidth=0)
            ax.axvspan(change_times_unit[0]+day_n, change_times_unit[1]+day_n, color='wheat', alpha=0.5, linewidth=0)
            ax.axvspan(change_times_unit[2]+day_n, change_times_unit[3]+day_n, color='wheat', alpha=0.5, linewidth=0)
            ax.axvspan(change_times_unit[3]+day_n, day_n+1, color='lightblue', alpha=0.5, linewidth=0)

    else:
        print("wrong format, needs to be in datetime")
        return
    ax.set_xlim([1, days_to_plot - 1])


def fill_plot_tp(ax, change_times_unit, day_unit, tv_internal):
    days_to_plot = math.ceil(tv_internal[-1]/day_unit)
    for day_n in range(days_to_plot):
        ax.axvspan(0+day_unit*day_n, change_times_unit[0]+day_unit*day_n, color='lightblue', alpha=0.5, linewidth=0)
        ax.axvspan(change_times_unit[0]+day_unit*day_n, change_times_unit[1]+day_unit*day_n, color='wheat', alpha=0.5, linewidth=0)
        ax.axvspan(change_times_unit[2]+day_unit*day_n, change_times_unit[3]+day_unit*day_n, color='wheat', alpha=0.5, linewidth=0)
        ax.axvspan(change_times_unit[3]+day_unit*day_n, day_unit + day_unit * day_n, color='lightblue', alpha=0.5, linewidth=0)
    ax.set_xlim([0, days_to_plot*day_unit])
