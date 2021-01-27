# This script will allow you to set up a fish tracking on a recording
# First you will be prompted to select the folder for the recording
# Next you will be prompted to run the background script
# then the define_ROI script
# Then you can press run

import datetime
import os
import glob

from tkinter.filedialog import askdirectory, askopenfilename
from tkinter import Tk
import cv2

from cichlidanalysis.tracking.rois import define_roi_still
from cichlidanalysis.tracking.offline_tracker import tracker
from cichlidanalysis.tracking.helpers import update_csvs
from cichlidanalysis.tracking.backgrounds import background_vid, update_background
from cichlidanalysis.io.meta import extract_meta, load_yaml


if __name__ == '__main__':
    background_update = 'm'
    while background_update not in {'y', 'n'}:
        background_update = input("Update background? y/n: \n")

    if background_update == 'y':
        background_update_files = 'm'
        while background_update_files not in {'a', 'n'}:
            background_update_files = input("Update background for all movies (a) or one (n)?: \n")

        percentile = 90
        if background_update_files == 'a':
            # percentile = input("Run with which percentile? 90 is default")
            update_background(percentile)

        elif background_update_files == 'n':
            root = Tk()
            root.withdraw()
            root.update()
            video_file_back = askopenfilename(title="Select movie file",
                                         filetypes=(("mp4 files", "*.mp4"), ("all files", "*.*")))
            root.destroy()

            # percentile = input("Run with which percentile? 90 is default")
            background_vid(video_file_back, 200, percentile)

    track_videos = 'm'
    while track_videos not in {'y', 'n'}:
        track_videos = input("Track videos? y/n: \n")

    if track_videos == 'y':
        track_all = 'm'
        while track_all not in {'y', 'n'}:
            track_all = input("Track all videos? y/n: \n")

        if track_all == 'y':
            # Allows a user to select top directory
            root = Tk()
            root.withdraw()
            root.update()
            vid_dir = askdirectory()
            root.destroy()

            os.chdir(vid_dir)
            video_files = glob.glob("*.mp4")
            video_files.sort()

            cam_dir = os.path.split(vid_dir)[0]

            backgrounds = glob.glob("*background.png")
            if len(backgrounds) < 1:
                print("Didn't find remade background, will use original background in camera folder")
                os.chdir(cam_dir)
                backgrounds = glob.glob("*.png")
            backgrounds.sort()

            rec_name = os.path.split(vid_dir)[1]

        elif track_all == 'n':
            # Allows a user to select top directory
            root = Tk()
            root.withdraw()
            root.update()
            video_file = askopenfilename(title="Select movie file",
                                         filetypes=(("mp4 files", "*.mp4"), ("all files", "*.*")))
            root.destroy()

            vid_dir = os.path.split(video_file)[0]
            cam_dir = os.path.split(vid_dir)[0]
            rec_name = os.path.split(vid_dir)[1]
            video_files = [os.path.split(video_file)[1]]

            os.chdir(os.path.split(video_file)[0])
            backgrounds = glob.glob(video_file[0:-4] + "*background.png")
            if len(backgrounds) < 1:
                print("Didn't find remade background, will use original background in camera folder")
                os.chdir(cam_dir)
                backgrounds = glob.glob("*" + video_file[-20:-10] + "*.png")

        fish_data = extract_meta(rec_name)

        track_roi = 'm'
        while track_roi not in {'y', 'n'}:
            track_roi = input("Track with another roi? y/n: \n")

        if track_roi == 'y':
            if track_all == 'n':
                roi_on_one = input("You are now changing the ROI for  only one video, this  is not recommended!\n "
                      "y to continue, n to  stop: \n")
                if roi_on_one == 'n':
                    exit()

            # Define video rois
            os.chdir(cam_dir)
            background_full = cv2.imread(backgrounds[0])
            # crop background to roi
            rec_rois = load_yaml(cam_dir, "roi_file")
            curr_roi = rec_rois["roi_" + str(fish_data['roi'][1:])]
            background_crop = background_full[curr_roi[1]:curr_roi[1] + curr_roi[3], curr_roi[0]:curr_roi[0] +
                                                                                                 curr_roi[2]]
            if background_crop.ndim == 3:
                background_crop = cv2.cvtColor(background_crop, cv2.COLOR_BGR2GRAY)

            vid_rois = load_yaml(vid_dir, "roi_file")
            if not vid_rois:
                define_roi_still(background_crop, vid_dir)
                vid_rois = load_yaml(vid_dir, "roi_file")

            for idx, val in enumerate(video_files):
                background_full = cv2.imread(backgrounds[idx], 0)
                background_crop = background_full[curr_roi[1]:curr_roi[1] + curr_roi[3], curr_roi[0]:curr_roi[0] +
                                                                                                     curr_roi[2]]
                tracker(os.path.join(vid_dir, val), background_crop, vid_rois, threshold=35, display=False,
                        area_size=100)

        else:
            vid_rois = load_yaml(cam_dir, "roi_file")
            width_trim, height_trim = vid_rois['roi_{}'.format(fish_data['roi'][-1])][2:4]
            rois = {'roi_0': (0, 0, width_trim, height_trim)}

            for idx, val in enumerate(video_files):
                background = cv2.imread(backgrounds[idx], 0)
                tracker(os.path.join(vid_dir, val), background, rois, threshold=35, display=False, area_size=100)

        # find cases where a movie has multiple csv files, add exclude tag to the ones from not today (date in file
        # names) and replace timestamps.

        date = datetime.datetime.now().strftime("%Y%m%d")
        os.chdir(vid_dir)

        # find all csvs
        all_files = glob.glob("*.csv".format(date))
        all_files.sort()

        # find csvs with today's date (which are the recent ones)
        new_files = glob.glob("*_{}_*.csv".format(date))
        new_files.sort()

        remove = ["exclude", "meta.csv", "als.csv"]

        select_files = []
        for file_a in all_files:
            counting = 0
            for tag in remove:
                if tag in file_a:
                    counting += 1
            if counting == 0:
                select_files.append(file_a)

        old_files = [file_a for file_a in select_files if file_a not in new_files]

        # find old_files which fit with the video_files (the movies being re-tracked)
        retracked = []
        for video in video_files:
            for old_f in old_files:
                if video[0:18] == old_f[0:18]:
                    retracked.append(old_f)

        for retrack_n, retracked_file in enumerate(retracked):
            for new_f in new_files:
                if retracked_file[0:18] == new_f[0:18]:
                    print("updating timestamps of {} and adding exclude tag to {}".format(new_f, retracked_file))
                    update_csvs(os.path.join(vid_dir, retracked_file), os.path.join(vid_dir, new_f))
