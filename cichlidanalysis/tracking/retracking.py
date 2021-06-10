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
import cv2.cv2 as cv2

from cichlidanalysis.tracking.rois import define_roi_still
from cichlidanalysis.tracking.offline_tracker import tracker
from cichlidanalysis.tracking.helpers import correct_tags
from cichlidanalysis.tracking.backgrounds import background_vid, update_background
from cichlidanalysis.io.meta import extract_meta, load_yaml
from cichlidanalysis.io.tracks import remove_tags

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
            backgrounds = remove_tags(backgrounds, remove=["frame"])
            new_bgd = True
            if len(backgrounds) < 1:
                print("Didn't find remade background, will use original background in camera folder")
                os.chdir(cam_dir)
                backgrounds = glob.glob("*.png")
                new_bgd = False
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
            backgrounds = remove_tags(backgrounds, remove=["frame"])
            new_bgd = True
            if len(backgrounds) < 1:
                print("Didn't find remade background, will use original background in camera folder")
                os.chdir(cam_dir)
                backgrounds = glob.glob("*" + video_file[-20:-10] + "*.png")
                new_bgd = False

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

            # ##  Define video rois ##
            # load recording roi
            rec_rois = load_yaml(cam_dir, "roi_file")
            curr_roi = rec_rois["roi_" + str(fish_data['roi'][1:])]

            # load video roi (if previously defined) or if not, then pick background and define a new ROI
            vid_rois = load_yaml(vid_dir, "roi_file")
            if not vid_rois:
                # allow user to pick the background image which to set the roi with
                root = Tk()
                root.withdraw()
                root.update()
                background_file = askopenfilename(title="Select background", filetypes=(("image files", "*.png"),))
                root.destroy()
                background_full = cv2.imread(background_file)

                # crop background to roi
                # have issue where roi can go over and then cropping the right background is an issue, rare case
                if curr_roi[1] + curr_roi[3] > background_full.shape[0]:
                    print("something off with roi/background size... readjusting")
                    off_by_on_y = background_full.shape[0] - (curr_roi[1] + curr_roi[3])
                    curr_roi = (curr_roi[0], curr_roi[1] + off_by_on_y, curr_roi[2], curr_roi[3])

                background_crop = background_full[curr_roi[1]:curr_roi[1] + curr_roi[3], curr_roi[0]:curr_roi[0] +
                                                                                                     curr_roi[2]]
                if background_crop.ndim == 3:
                    background_crop = cv2.cvtColor(background_crop, cv2.COLOR_BGR2GRAY)

                define_roi_still(background_crop, vid_dir)
                vid_rois = load_yaml(vid_dir, "roi_file")

            for idx, val in enumerate(video_files):
                movie_n = val.split("_")[1]
                background_of_movie = [i for i in backgrounds if i.split("_")[1] == movie_n]
                if not background_of_movie:
                    print("didn't find background, stopping tracking")
                    break
                print("tracking with background {}".format(background_of_movie))
                background_full = cv2.imread(background_of_movie[0], 0)
                if new_bgd:
                    background_crop = background_full
                else:
                    background_crop = background_full[curr_roi[1]:curr_roi[1] + curr_roi[3], curr_roi[0]:curr_roi[0] +
                                                                                                         curr_roi[2]]
                tracker(os.path.join(vid_dir, val), background_crop, vid_rois, threshold=35, display=False,
                        area_size=100)

        else:
            vid_rois = load_yaml(cam_dir, "roi_file")
            width_trim, height_trim = vid_rois['roi_{}'.format(fish_data['roi'][-1])][2:4]
            rois = {'roi_0': (0, 0, width_trim, height_trim)}

            for idx, val in enumerate(video_files):
                movie_n = val.split("_")[1]
                background_of_movie = [i for i in backgrounds if (i.split('/')[-1]).split("_")[1] == movie_n]
                print("tracking with background {}".format(background_of_movie))
                background = cv2.imread(background_of_movie[0], 0)

                tracker(os.path.join(vid_dir, val), background, rois, threshold=35, display=False, area_size=100)

        # find cases where a movie has multiple csv files, add exclude tag to the ones from not today (date in file
        # names) and replace timestamps.
        date = datetime.datetime.now().strftime("%Y%m%d")
        correct_tags(date, vid_dir)
