# In some cases the camera is bumped, the water refilled or something else happens to disturb th video tracking.
# To get around it we use this script. There are two cases which is will take care of:
# if you want to NaN a region, or if you want to spilt the video and recalculate a background image for each part and
# retrack. This second option also retracks the next video as it assumes that the background used wasn't good enough.
import os
import glob
from tkinter.filedialog import askdirectory, askopenfilename
from tkinter import Tk
import copy
import sys
import datetime

import cv2
import numpy as np

from cichlidanalysis.io.meta import load_yaml, extract_meta
from cichlidanalysis.io.tracks import load_track
# from cichlidanalysis.analysis.processing import remove_high_spd_xy, remove_high_spd, smooth_speed, neg_values, coord_smooth

# adding line Annika - testing github


# load offending movie, median (camera/roi?) and track
# identify point when disturbance starts, identify point when disturbance ends (video viewer?)
# remake medians for each part of the video
# retrack the video parts
# copy the timepoints from the original track.

# function called by trackbar, sets the next frame to be read
def background_vid_split(videofilepath, nth_frame, percentile, split_range):
    """ (str, int, int, list)
     This function will create a median image of the defined area"""
    try:
        cap = cv2.VideoCapture(videofilepath)
    except:
        print("problem reading video file, check path")
        return

    counter = 0
    gatheredFramess = []
    while cap.isOpened() and (counter < split_range[1]):
        ret, frame = cap.read()
        if frame is None:
            break
        if counter % nth_frame == 0 and counter in np.arange(split_range[0], split_range[1]):
            print("Frame {}".format(counter))
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # settings with Blur settings for the video loop
            gatheredFramess.append(image)
        counter += 1

    background = np.percentile(gatheredFramess, int(percentile), axis=0).astype(dtype=np.uint8)
    cv2.imshow('Calculated Background from {} percentile'.format(percentile), background)

    vid_name = videofilepath[0:-4]
    print("saving background")
    range_s = str(split_range[0]).zfill(5)
    range_e = str(split_range[1]).zfill(5)
    cv2.imwrite('{0}_per{1}_frame{2}-{3}_background.png'.format(vid_name, percentile, range_s, range_e), background)

    cap.release()
    cv2.destroyAllWindows()
    return background


def getFrame(frame_nr):
    global video
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_nr)


def split_select(video_path, background_cropped):
    """ Function that takes a video path, a median file, and rois. It then uses background subtraction and centroid
    tracking to find the XZ coordinates of the largest contour. This script has a threshold bar which allows you to try
    different levels. Once desired threshold level is found. Press 'q' to quit and the selected value will be used """
    split_start, split_end = [], []
    # load video
    global video
    video = cv2.VideoCapture(video_path)
    nr_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # set up image display and trackbar for
    cv2.namedWindow('Splitting finder')
    cv2.createTrackbar("Frame", "Splitting finder", 0, nr_of_frames, getFrame)
    cv2.startWindowThread()

    playing = 1
    ret, frame = video.read()
    frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame.shape != background_cropped.shape:
        # add padding to the median
        if frame_bw.shape[0] != background_cropped.shape[0] and frame_bw.shape[1] == background_cropped.shape[1]:
            background_cropped = np.concatenate((background_cropped, frame_bw[background_cropped.shape[0]:
                                                                              frame_bw.shape[0], :]), axis=0)

        if frame_bw.shape[0] == background_cropped.shape[0] and frame_bw.shape[1] != background_cropped.shape[1]:
            background_cropped = np.concatenate((background_cropped, frame_bw[:, background_cropped.shape[1]:frame_bw.
                                                 shape[1]]), axis=1)

    while video.isOpened():
        if playing:
            ret, frame = video.read()
            frame_nr = video.get(cv2.CAP_PROP_POS_FRAMES)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frameDelta = cv2.absdiff(frame, background_cropped)
        cv2.putText(frameDelta, "Press enter to select frame, press space bar to pause", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frameDelta, "'a' for backwards, 'd' advance, 'enter' = save out the start/end values", (5, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1)
        cv2.putText(frameDelta, "frame = {}, split_start ('s') = {}, split_end ('e') = {}".format(frame_nr,
                                                                                                  split_start,
                                                                                                  split_end), (5, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("Splitting finder", frameDelta)

        k = cv2.waitKey(33)

        if k == 27 or k == ord("q"):
            cv2.destroyAllWindows()
            break

        elif k == 32:
            if playing == 0:
                playing = 1
            elif playing == 1:
                playing = 0

        elif k == ord("a"):
            frame_nr = video.get(cv2.CAP_PROP_POS_FRAMES)
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_nr - 2)
            ret, frame = video.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        elif k == ord("d"):
            frame_nr = video.get(cv2.CAP_PROP_POS_FRAMES)
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_nr)
            ret, frame = video.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        elif k == ord("s"):
            split_start = video.get(cv2.CAP_PROP_POS_FRAMES)

        elif k == ord("e"):
            split_end = video.get(cv2.CAP_PROP_POS_FRAMES)

        elif k == 13:  # or finished:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            video.release()
            print("Splitting video between frame {} and frame {}".format(split_start, split_end))
            return int(split_start), int(split_end)

    # print("Finished cleaning up")
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    # video.release()
    # return int(split_start), int(split_end)


if __name__ == '__main__':
    # find file path for video and load track
    # Allows a user to select file
    root = Tk()
    root.withdraw()
    root.update()
    video_path = askopenfilename(title="Select movie file", filetypes=(("mp4 files", "*.mp4"), ("all files", "*.*")))
    root.destroy()

    vid_folder_path = os.path.split(video_path)[0]
    vid_timestamp = os.path.split(video_path)[1][0:-10]
    cam_folder_path = os.path.split(vid_folder_path)[0]
    vid_folder_name = os.path.split(vid_folder_path)[1]

    track_path = video_path[0:-4] + ".csv"
    video_name = os.path.split(video_path)[1]

    displacement_internal, track_single = load_track(track_path)
    meta = load_yaml(vid_folder_path, "meta_data")
    rois = load_yaml(cam_folder_path, "roi_file")
    config = load_yaml(cam_folder_path, "config")
    fish_data = extract_meta(vid_folder_name)

    os.chdir(cam_folder_path)
    files = glob.glob("*.png")
    files.sort()
    files.insert(0, files.pop(files.index(min(files, key=len))))
    if "{}_Median.png".format(vid_timestamp) in files:
        previous_median_name = files[files.index("{}_Median.png".format(vid_timestamp)) - 1]
        print(previous_median_name)
    else:
        previous_median_name = files[files.index("{}_per90_Background.png".format(vid_timestamp)) - 1]
        print(previous_median_name)

    # find and load background file
    background_path = os.path.join(cam_folder_path, "{}".format(previous_median_name))
    if len(glob.glob(background_path)) != 1:
        print('too many or too few background files in folder:' + cam_folder_path)
        sys.exit()
    else:
        background_full = cv2.imread(glob.glob(background_path)[0], 0)

    roi_n = rois["roi_" + fish_data['roi'][1]]
    background = background_full[roi_n[1]:roi_n[1] + roi_n[3], roi_n[0]:roi_n[0] + roi_n[2]]

    split_s, split_e = split_select(video_path, background)

    while split_e < split_s:
        print("Split start must be smaller than split end, retry")
        split_s, split_e = split_select(video_path, background)

    retrack = 'm'
    while retrack not in {'y', 'n'}:
        retrack = input("Retrack the split movie? y/n: \n")

    if retrack == 'y':
        # remake backgrounds from the split.
        os.chdir(vid_folder_path)

        # load original track (need timestamps)
        orig_file = glob.glob("*{}.csv".format(video_name[0:-4]))
        na, track_single_orig = load_track(os.path.join(vid_folder_path, orig_file[0]))

        split_range = ([0, split_s], [split_e, track_single_orig.shape[0]])
        backgrounds = []
        for part in split_range:
            backgrounds.append(background_vid_split(video_path, 100, 90, part))

        # retrack part of the movie with the correct background. Will also need to use the second background for the movie afterwards
        # making roi for full video
        vid_rois = load_yaml(cam_folder_path, "roi_file")
        width_trim, height_trim = vid_rois['roi_{}'.format(fish_data['roi'][-1])][2:4]
        rois = {'roi_0': (0, 0, width_trim, height_trim)}

        for idx, curr_background in enumerate(backgrounds):
            area_s = 100
            thresh = 35
            tracker(video_path, curr_background, rois, threshold=thresh, display=False, area_size=area_s,
                    split_range=split_range[idx])

            # add in the right timepoints (of a primary track - not a full retrack)
            # load the newly tracked csv
            date = datetime.datetime.now().strftime("%Y%m%d")
            range_s = str(split_range[idx][0]).zfill(5)
            range_e = str(split_range[idx][1]).zfill(5)
            filename = video_path[0:-4] + "_tracks_{}_Thresh_{}_Area_{}_Range{}-{}_.csv".format(date, thresh,
                                                                                                area_s, range_s,
                                                                                                range_e)
            _, track_single_split = load_track(filename)
            # track_single_cleaned = copy.copy(track_single_split)
            # replace the frame col with the data from the original track
            if idx == 0:
                # add NaNs (-1 for exclude) and timestamps
                dummy = np.empty([split_range[1][0] - split_range[0][1], 4])
                dummy[:] = -1
                track_single_split = np.concatenate((track_single_split, dummy))
                track_single_split[:, 0] = track_single_orig[split_range[0][0]:split_range[1][0], 0]

            else:
                track_single_split[0:split_range[idx][1] - split_range[idx][0], 0] = track_single_orig[
                                                                                     split_range[idx][0]:
                                                                                     split_range[idx][1], 0]
            # save over
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            np.savetxt(filename, track_single_split, delimiter=",")

        # track the next video (as median will be messed up if it's a background problem.
        # find next movie path
        next_movie_num = str(int(video_name.split("_")[1]) + 1)
        next_movie_name = glob.glob("*_{}_*.mp4".format(next_movie_num))[0]
        next_movie_path = os.path.join(vid_folder_path, next_movie_name)
        tracker(next_movie_path, backgrounds[1], rois, threshold=thresh, display=False, area_size=area_s)
        # find newly made csv and rename it so it will be used by the loading script.
        next_csv_name = glob.glob("*{}*_{}_*.csv".format(next_movie_num, date))[0]
        os.rename(os.path.join(vid_folder_path, next_csv_name), next_movie_path[0:-4] + "_cleaned.csv")
        # load csv file and replace timestamps
        _, track_single_retracked = load_track(next_movie_path[0:-4] + "_cleaned.csv")
        _, track_single_orig2 = load_track(next_movie_path[0:-4] + ".csv")
        track_single_retracked[:, 0] = track_single_orig2[:, 0]
        # # save over
        resave_path = next_movie_path[0:-4] + "_cleaned.csv"
        os.makedirs(os.path.dirname(resave_path), exist_ok=True)
        np.savetxt(resave_path, track_single_retracked, delimiter=",")

    else:
        # load track and mark the excluded part (keeps copy of the track as it saves out a "cleaned" version which is
        # prioritised), later '-1' are replaced by nans (after nans from non tracking are interpolated)
        os.chdir(vid_folder_path)
        orig_file = glob.glob("*{}.csv".format(video_name[0:-4]))
        na, track_single = load_track(os.path.join(vid_folder_path, orig_file[0]))
        track_single_cleaned = copy.copy(track_single)
        track_single_cleaned[split_s:split_e, 1:4] = -1

        filename = video_path[0:-4] + "_cleaned.csv"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.savetxt(filename, track_single_cleaned, delimiter=",")
        print("done")
