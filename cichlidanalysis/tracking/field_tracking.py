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
from cichlidanalysis.io.tracks import remove_tags, get_file_paths_from_nums
from cichlidanalysis.io.movies import get_movie_paths


if __name__ == '__main__':
    background_create = 'm'
    while background_create not in {'y', 'n'}:
        background_create = input("Create background? y/n: \n")

    if background_create == 'y':
        background_create_files = 'n'

        percentile = 90

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
        track_all = 'n'

        # Allows a user to select top directory
        root = Tk()
        root.withdraw()
        root.update()
        video_file = askopenfilename(title="Select movie file",
                                      filetypes=(("mp4 files", "*.mp4"), ("all files", "*.*")))
        root.destroy()

        # This is finding the directory structure that doesn't exist for field videos
        vid_dir = os.path.split(video_file)[0]
        cam_dir = os.path.split(vid_dir)[0]
        rec_name = os.path.split(vid_dir)[1]
        video_files = [os.path.split(video_file)[1]]

        # This determines the name of the background file
        os.chdir(os.path.split(video_file)[0])
        backgrounds = glob.glob(video_file[0:-4] + "*background.png")
        backgrounds = remove_tags(backgrounds, remove=["frame"])
        new_bgd = True
        if len(backgrounds) < 1:
            print("Didn't find remade background, will use original background in camera folder")
            os.chdir(cam_dir)
            backgrounds = glob.glob("*" + video_file[-20:-10] + "*.png")
            new_bgd = False

        # fish_data = extract_meta(rec_name)

        track_roi = 'y'
        if track_roi == 'y':
            # # This loads the original recording roi, which shouldn't exist
            # # Will need to default pick a new one
            # rec_rois = load_yaml(cam_dir, "roi_file")
            # curr_roi = rec_rois["roi_" + str(fish_data['roi'][1:])]
            #
            # # load video roi (if previously defined) or if not, then pick background and define a new ROI
            # vid_rois = load_yaml(vid_dir, "roi_file")
            #if not vid_rois:
                # allow user to pick the background image which to set the roi with
####################### I think I want to start here
            root = Tk()
            root.withdraw()
            root.update()
            background_file = askopenfilename(title="Select background to define new ROI", filetypes=(("image files", "*.png"),))
            root.destroy()
            background_full = cv2.imread(background_file)


            define_roi_still(background_full, vid_dir)

            vid_rois = load_yaml(vid_dir, "roi_file")


            ## This will track the video (should only be one)
            background_of_movie = background_file
            print("tracking with background {}".format(background_of_movie))

            background_full = cv2.imread(background_of_movie, 0)

            # background_crop = background_full[vid_rois[1]:vid_rois[1] + vid_rois[3], vid_rois[0]:vid_rois[0] + vid_rois[2]]
# threshold = 35, area = 100
            tracker(video_file, background_full, vid_rois, threshold=5, display=False,
                    area_size=10)


        # else:
        #     vid_rois = load_yaml(cam_dir, "roi_file")
        #     width_trim, height_trim = vid_rois['roi_{}'.format(fish_data['roi'][-1])][2:4]
        #     rois = {'roi_0': (0, 0, width_trim, height_trim)}
        #
        #     for idx, val in enumerate(video_files):
        #         movie_n = val.split("_")[1]
        #         background_of_movie = [i for i in backgrounds if (i.split('/')[-1]).split("_")[1] == movie_n]
        #         print("tracking with background {}".format(background_of_movie[0]))
        #         background = cv2.imread(background_of_movie[0], 0)
        #
        #         # check if using an old background (need to crop) or new
        #         if new_bgd:
        #             background_crop = background
        #         else:
        #             curr_roi_n = vid_dir.split("_")[-3][1]
        #             curr_roi = vid_rois['roi_{}'.format(curr_roi_n)]
        #             background_crop = background[curr_roi[1]:curr_roi[1] + curr_roi[3], curr_roi[0]:curr_roi[0] +
        #                                                                                          curr_roi[2]]
        #             # extremely rarely the background needs to be padded, this hack can be used
        #             # Used for:
        #             # FISH20211103_c5_r1_Lepidiolamprologus-elongatus_su, FISH20211006_c3_r0_Neolamprologus-brevis_su
        #             # import numpy as np
        #             # background_crop = np.vstack([background_crop, np.zeros([1, curr_roi[2]], dtype='uint8')])
        #
        #         tracker(os.path.join(vid_dir, val), background_crop, rois, threshold=35, display=False, area_size=100)

        # find cases where a movie has multiple csv files, add exclude tag to the ones from not today (date in file
        # names) and replace timestamps.
        date = datetime.datetime.now().strftime("%Y%m%d")
        correct_tags(date, vid_dir)
