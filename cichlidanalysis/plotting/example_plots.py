import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.signal import find_peaks
from matplotlib.ticker import (MultipleLocator)

from cichlidanalysis.io.als_files import load_als_files, load_bin_als_files
from cichlidanalysis.analysis.behavioural_state import define_rest, plotting_clustering_states
from cichlidanalysis.analysis.processing import threshold_data
from cichlidanalysis.analysis.bouts import find_bout_start_ends_pd
from cichlidanalysis.analysis.crepuscular_pattern import crespuscular_daily_ave_fish, crepuscular_peaks


def plot_example_spd_move_rest():
    # rootdir = select_dir_path()
    # rootdir = "/Volumes/BZ/RG Schier/Scientific Data/Cichlid_sleep_videos/FISH20210825_LT_Neobue/20210825_c1_Neolamprologus-buescheri/FISH20210825_c1_r0_Neolamprologus-buescheri_su"
    rootdir = "/Volumes/BZ/RG Schier/Scientific Data/Cichlid_sleep_videos/FISH20220209_LT_Lamsig/20220209_c1_Lamprologus-signatus/FISH20220209_c1_r0_Lamprologus-signatus_su"

    MOVE_THRESH = 15

    # ### Behavioural state - calculated from Movement ###
    TIME_WINDOW_SEC = 60
    FRACTION_THRESH = 0.05

    move_colour = 'forestgreen'
    rest_colour = 'darkorchid'

    SEC_TO_NS = 10 ** 9
    fish_tracks = load_als_files(rootdir)
    fish_IDs = fish_tracks['FishID'].unique()

    fish_tracks['movement'] = np.nan
    for fish in fish_IDs:
        # threshold the speed_mm with 15mm/s
        fish_tracks.loc[(fish_tracks.FishID == fish), 'movement'] = threshold_data(
            fish_tracks.loc[(fish_tracks.FishID == fish), "speed_mm"], MOVE_THRESH)

    fish_tracks = define_rest(fish_tracks, TIME_WINDOW_SEC, FRACTION_THRESH)

    range_to_plot = range(4150400, 4600000)
    fig, ax1 = plt.subplots(figsize=(6, 3))
    spd = fish_tracks.speed_mm[range_to_plot].to_numpy()
    ax1.plot(spd)
    ax1.axhline(MOVE_THRESH, c='k')

    movements = fish_tracks.movement[range_to_plot].to_numpy()
    bout_start_t, bout_end_t, bout_lengths = find_bout_start_ends_pd(movements)
    for bout_n in range(len(bout_start_t)):
        ax1.axvspan(bout_start_t[bout_n], bout_end_t[bout_n], color=move_colour, alpha=0.5, linewidth=0)

    rests = fish_tracks.rest[range_to_plot].to_numpy()
    bout_start_t, bout_end_t, bout_lengths = find_bout_start_ends_pd(rests)
    for bout_n in range(len(bout_start_t)):
        ax1.axvspan(bout_start_t[bout_n], bout_end_t[bout_n], color=rest_colour, alpha=0.5,
                    linewidth=0)

    ax1.set_ylabel("Speed mm/s")
    steps_sec = 60
    tick_position = np.arange(0, len(range_to_plot), step=steps_sec * 10)
    tick_labels = np.arange(0, len(tick_position)*steps_sec, step=steps_sec).tolist()
    tick_labels_str = [str(e) for e in tick_labels]

    ax1.set_xticks(tick_position)
    ax1.set_xticklabels(tick_labels_str)
    ax1.set_xlabel("Seconds")

    ax1.set_ylim([0, 120])
    ax1.set_xlim([100, 3100])
    move_patch = mpatches.Patch(color=move_colour, label='Movement')
    rest_patch = mpatches.Patch(color=rest_colour, label='Rest')
    ax1.legend(handles=[move_patch, rest_patch])

    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "Example_movement_rest.png"), dpi=300)
    plt.close('all')


def plot_example_cres_peaks():
    rootdir = "/Volumes/BZ/RG Schier/Scientific Data/Cichlid_sleep_videos/_analysis2/Neobue"
    fish_tracks_bin = load_bin_als_files(rootdir, "*als_30m.csv")

    change_times_unit = [7 * 2, 7.5 * 2, 18.5 * 2, 19 * 2]
    feature = 'speed_mm'
    species = fish_tracks_bin.species[0]
    fishes = fish_tracks_bin.FishID.unique()

    border_top = np.ones(48)
    border_bottom = np.ones(48) * 1.05
    dawn_s, dawn_e, dusk_s, dusk_e = [6 * 2, 8 * 2, 18 * 2, 20 * 2]
    border_bottom[6 * 2:8 * 2] = 0
    border_bottom[18 * 2:20 * 2] = 0

    peak_prom = 0.15
    if feature == 'speed_mm':
        border_top = border_top * 200
        border_bottom = border_bottom * 200
        peak_prom = 7

    fish_example = fishes[1]
    example_fish_spd = fish_tracks_bin.loc[fish_tracks_bin.FishID == fish_example, :].reset_index(drop=True)

    fig2, ax = plt.subplots(figsize=(5, 4))
    ax.axvline(6 * 2, c='indianred')
    ax.axvline(8 * 2, c='indianred')
    ax.axvline(18 * 2, c='indianred')
    ax.axvline(20 * 2, c='indianred')

    ax.axvspan(0, change_times_unit[0], color='lightblue', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[0], change_times_unit[1], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[2], change_times_unit[3], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[3], 24 * 2, color='lightblue', alpha=0.5, linewidth=0)
    ax.set_ylim([0, 60])
    ax.set_xlim([0, 24 * 2])
    plt.xlabel("Time (h)")
    plt.ylabel("Speed (mm/s)")

    x = example_fish_spd.loc[0:47, 'speed_mm'].to_numpy()
    peaks, _ = find_peaks(x, distance=4, prominence=peak_prom, height=(border_bottom, border_top))
    ax.plot(x)
    ax.plot(peaks, x[peaks], "o", color="r")
    ax.set_ylabel("Speed mm/s")

    steps_hours = 6
    tick_position = np.arange(0, 48, step=steps_hours * 2)
    tick_labels = np.arange(0, len(tick_position)*steps_hours, step=steps_hours).tolist()
    tick_labels_str = [str(e) for e in tick_labels]
    ax.set_xticks(tick_position)
    ax.set_xticklabels(tick_labels_str)
    plt.savefig(os.path.join(rootdir, "Example_find_peaks.png"))



if __name__ == '__main__':
    print("cichlids")
    plot_example_cres_peaks()
    plot_example_spd_move_rest()

