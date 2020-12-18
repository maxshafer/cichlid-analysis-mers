### this script loads a video and it's corresponding track, it plots the centroid over the video and allows you to scroll
# through and control the playback speed
# Inspiration from:
# https://stackoverflow.com/questions/54674343/how-do-we-create-a-trackbar-in-open-cv-so-that-i-can-use-it-to-skip-to-specific
# and for speed graph:
# https://stackoverflow.com/questions/32111705/overlay-a-graph-over-a-video

# importing libraries
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
import sys
import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt

from cichlidanalysis.io.meta import load_yaml, extract_meta
from cichlidanalysis.io.tracks import extract_tracks_from_fld
from cichlidanalysis.analysis.processing import int_nan_streches, remove_high_spd_xy, smooth_speed


# function called by trackbar, sets the next frame to be read
def getFrame(frame_nr):
    global video
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_nr)


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
vid_folder = os.path.split(vid_folder_path)[1]

track_single, displacement_internal = extract_tracks_from_fld(vid_folder_path, vid_timestamp)

video_name = os.path.split(video_path)[1]

meta = load_yaml(vid_folder_path, "meta_data")
rois = load_yaml(cam_folder_path, "roi_file")
config = load_yaml(cam_folder_path, "config")
fish_data = extract_meta(vid_folder)

os.chdir(cam_folder_path)
files = glob.glob("*.png")
files.sort()
files.insert(0, files.pop(files.index(min(files, key=len))))
if "{}_Median.png".format(vid_timestamp) in files:
    previous_median_name = files[files.index("{}_Median.png".format(vid_timestamp))-1]
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

# interpolate between NaN streches
try:
    x_n = int_nan_streches(track_single[:, 1])
    y_n = int_nan_streches(track_single[:, 2])
except:
    x_n = track_single[:, 1]
    y_n = track_single[:, 2]

# find displacement
displacement_internal_mm_s = displacement_internal * config["mm_per_pixel"] * config['fps']
speed_full_i = np.sqrt(np.diff(x_n) ** 2 + np.diff(y_n) ** 2)
speed_t, x_nt, y_nt = remove_high_spd_xy(speed_full_i, x_n, y_n)

speed_sm = smooth_speed(speed_t, win_size=5)
speed_sm_mm = speed_sm * config["mm_per_pixel"]
speed_sm_mm_ps = speed_sm_mm * config['fps']

threshold = (0.25 * meta["fish_length_mm"])/config["mm_per_pixel"]

# open video
video = cv2.VideoCapture(video_path)

# get total number of frames
nr_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# create display window
cv2.namedWindow("Speed of {}".format(video_name))
cv2.namedWindow("Background subtraction of {}".format(video_name))

# set wait for each frame, determines playbackspeed
playSpeed = 50

# add trackbar
cv2.createTrackbar("Frame", "Speed of {}".format(video_name), 0, nr_of_frames, getFrame)

ret, frame = video.read()
height, width = frame.shape[:2]
frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
if frame_bw.shape != background.shape:
    # add padding to the median
    if frame_bw.shape[0] != background.shape[0] and frame_bw.shape[1] == background.shape[1]:
        background = np.concatenate((background, frame_bw[background.shape[0]:frame_bw.shape[0], :]), axis=0)

    if frame_bw.shape[0] == background.shape[0] and frame_bw.shape[1] != background.shape[1]:
        background = np.concatenate((background, frame_bw[:, background.shape[1]:frame_bw.shape[1]]), axis=1)

# max_sp = int(np.nanpercentile(speed_sm, 99) + 5)
max_sp = np.nanmax(speed_sm_mm_ps)
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
plt.ion()
plt.show()
playing = 1


while 1:
    # show frame, break the loop if no frame is found
    curr_frame = video.get(cv2.CAP_PROP_POS_FRAMES) -1

    if ret:
        try:
            cX, cY = (int(track_single[int(curr_frame), 1]), int(track_single[int(curr_frame), 2]))
            cv2.circle(frame, (int(x_nt[int(curr_frame)]), int(y_nt[int(curr_frame)])), 4, (0, 255, 255), 4)
            cv2.circle(frame, (cX, cY), 4, (0, 0, 255), 2)
            cv2.putText(frame, "yellow: corrected centroid, red: raw centroid, frame: {}".format(curr_frame), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 200), 1)
        except:
            cv2.rectangle(img=frame, pt1=(0, 0), pt2=(10, 10), color=(0, 0, 255), thickness=-1)

        cv2.imshow("Speed of {}".format(video_name), frame)
        frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frameDelta = cv2.absdiff(frame_bw, background)
        image_thresholded = cv2.threshold(frameDelta, 35, 255, cv2.THRESH_TOZERO)[1]
        (contours, _) = cv2.findContours(image_thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contourOI_ = (max(contours, key=cv2.contourArea))
            area = cv2.contourArea(contourOI_)
            if area > 100:
                contourOI = (cv2.convexHull(contourOI_))
            cv2.drawContours(frameDelta, contourOI_.astype(int), -1, 255, 1)

        frameDelta = cv2.cvtColor(frameDelta, cv2.COLOR_GRAY2BGR)
        try:
            cv2.circle(frameDelta, (int(x_nt[int(curr_frame)]), int(y_nt[int(curr_frame)])), 4, (0, 255, 255), 4)
            cv2.circle(frameDelta, (cX, cY), 4, (0, 0, 255), 2)
        except:
            cv2.rectangle(frame, 0, 10, 255, 1)
        cv2.imshow("Background subtraction of {}".format(video_name), frameDelta)


    ax.clear()
    dummy = np.zeros([int(max_sp), 400, 3]).astype(np.uint8)
    plt.imshow(dummy, origin='lower')

    if curr_frame > 200:
        track_curr = 200
    else:
        track_curr = curr_frame
    plt.plot([track_curr, track_curr], [-0.01, max_sp], color='r')

    win_min = int(curr_frame - 20 * 10)
    win_max = int(curr_frame + 20 * 10)

    if win_min < 0:
        win_min = 0
    if win_max > speed_sm.shape[0]:
        win_max = speed_sm.shape[0]

    values = speed_sm_mm_ps[win_min:win_max]
    values_2 = displacement_internal_mm_s[win_min:win_max]
    plt.plot(values)
    plt.plot(values_2)
    plt.plot([0, 400], [threshold, threshold])

    ax.set_ylim([0, max_sp])
    ax.set_xlim([0, 400])
    ax.set_aspect('auto')
    ax.legend(["current frame", "speed_sm_mm_ps", "speed_raw_mm_ps", "threshold (0.25 bl)"])
    plt.ylabel("mm/s")

    k = cv2.waitKey(33)
    # stop playback when q is pressed
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

    elif k == ord("d"):
        frame_nr = video.get(cv2.CAP_PROP_POS_FRAMES)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_nr)
        ret, frame = video.read()

    if playing:
        # Get the next videoframe
        ret, frame = video.read()
    else:
        ret = True


# release resources
video.release()
cv2.destroyAllWindows()
