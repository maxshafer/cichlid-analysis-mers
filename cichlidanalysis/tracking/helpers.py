import os
import glob
import copy

import cv2.cv2 as cv2
import numpy as np
from tkinter.filedialog import askdirectory
from tkinter import Tk

from cichlidanalysis.io.tracks import load_track, remove_tags


def threshold_select(video_path, median_full, rois):
    """ Function that takes a video path, a median file, and rois. It then uses background subtraction and centroid
    tracking to find the XZ coordinates of the largest contour. This script has a threshold bar which allows you to try
    different levels. Once desired threshold level is found. Press 'q' to quit and the selected value will be used """
    t_pos = 35
    frame_id = 0

    # create trackbar for setting threshold
    def nothing(x):
        pass

    # create window
    cv2.namedWindow("Live thresholded")

    # create threshold trackbar
    cv2.createTrackbar('threshold', "Live thresholded", t_pos, 255, nothing)

    # load video
    video = cv2.VideoCapture(video_path)

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            print("reached end of video")
            video.release()
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frameDelta_full = cv2.absdiff(median_full, gray)

        # tracking
        cx = list()
        cy = list()
        contourOI = list()
        contourOI_ = list()
        t_pos = cv2.getTrackbarPos('threshold', "Live thresholded")
        for roi in range(0, len(rois) - 1):
            # for the frame define an ROI and crop image
            curr_roi = rois["roi_" + str(roi)]
            frameDelta = frameDelta_full[curr_roi[1]:curr_roi[1] + curr_roi[3], curr_roi[0]:curr_roi[0] + curr_roi[2]]
            image_thresholded = cv2.threshold(frameDelta, t_pos, 255, cv2.THRESH_TOZERO)[1]
            (contours, _) = cv2.findContours(image_thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                contourOI_.append(max(contours, key=cv2.contourArea))
                area = cv2.contourArea(contourOI_[roi])
                if area > 0:
                    contourOI.append(cv2.convexHull(contourOI_[roi]))
                    M = cv2.moments(contourOI[roi])
                    cx.append(int(M["m10"] / M["m00"]))
                    cy.append(int(M["m01"] / M["m00"]))
                else:
                    print("no large enough contour found for roi {}!".format(roi))
                    contourOI_[-1] = False
                    contourOI.append(False)
                    cx.append(np.nan)
                    cy.append(np.nan)
            else:
                print("no contour found for roi {}!".format(roi))
                contourOI_.append(False)
                contourOI.append(False)
                cx.append(np.nan)
                cy.append(np.nan)

        full_image_thresholded = (cv2.threshold(frameDelta_full, t_pos, 255, cv2.THRESH_TOZERO)[1])
        # Live display of full resolution and ROIs
        cv2.putText(full_image_thresholded, "Framenum: {}".format(frame_id), (30, full_image_thresholded.shape[0] -
                                                                              30),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=255)

        for roi in range(0, len(rois) - 1):
            if np.all(contourOI_[roi] != False):
                curr_roi = rois["roi_" + str(roi)]
                # add in contours
                corrected_contour = np.empty(contourOI_[roi].shape)
                corrected_contour[:, 0, 0] = contourOI_[roi][:, 0, 0] + curr_roi[0]
                corrected_contour[:, 0, 1] = contourOI_[roi][:, 0, 1] + curr_roi[1]
                cv2.drawContours(full_image_thresholded, corrected_contour.astype(int), -1, 255, 1)

                # add in centroid
                cv2.circle(full_image_thresholded, (cx[roi] + curr_roi[0], cy[roi] + curr_roi[1]), 8, 255, 1)

            cv2.imshow("Live thresholded", full_image_thresholded)
            cv2.imshow("Live", gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1

    print("Tracking finished on video cleaning up")
    cv2.destroyAllWindows()
    return t_pos


def exclude_tag_csv(orig_csv_path_i):
    # check if old path exists
    if os.path.isfile(orig_csv_path_i):
        if orig_csv_path_i[-12:] == "_exclude.csv":
            print("there's already a exclude tag on this file: {}".format(orig_csv_path_i))
            return
        else:
            # make new path name
            tagged_path = orig_csv_path_i[0:-4] + "_exclude.csv"

            # check if it already exists
            if os.path.isfile(tagged_path):
                print("there's already a file by that name, can't tag file")
                return
            else:
                os.rename(orig_csv_path_i, tagged_path)
    else:
        print("no file found - check path name")


def copy_timestamps(orig_csv_path_i, new_csv_path_i):
    """This script will copy the timestamps from a original track"""
    # load csv file and replace timestamps
    _, track_single_orig = load_track(orig_csv_path_i)
    _, track_single_retracked = load_track(new_csv_path_i)

    if track_single_retracked[0, 0] == 0:
        track_single_retracked[:, 0] = track_single_orig[:, 0]

        # save over
        os.makedirs(os.path.dirname(new_csv_path_i), exist_ok=True)
        np.savetxt(new_csv_path_i, track_single_retracked, delimiter=",")
    else:
        print("Timestamps already copied")


def update_csvs(orig_csv_path, new_csv_path):
    """This script will copy the timestamps from a original track, and rename the old track so it has the "exclude" tag.
    This will not work on "range" files"""
    orig_csv_name = os.path.split(orig_csv_path)[1]
    # orig_csv_folder = os.path.split(orig_csv_path)[0]
    new_csv_name = os.path.split(new_csv_path)[1]

    # check that file is not a range file
    if "Range" in new_csv_name:
        print("new csv file is a range, not adding exclude tag")
    elif "exclude" in orig_csv_name:
        copy_timestamps(orig_csv_path, new_csv_path)
        print("old csv file already has exclude tag, copying timestamps but not adding tag")
    else:
        copy_timestamps(orig_csv_path, new_csv_path)
        # add "exclude" tag to old csv track file
        exclude_tag_csv(orig_csv_path)


def correct_tags(retracking_date, vid_directory):
    """ find cases where a movie has multiple csv files, add exclude tag to the ones from not today (date in file
    names), exclude any old tracks, check if timestamps have been replaced.

    :param retracking_date: "%Y%m%d"
    :param vid_directory: path of a recording
    :return:
    """

    os.chdir(vid_directory)
    video_files = glob.glob("*.mp4")
    video_files.sort()

    # find all csvs
    all_files = glob.glob("*.csv")
    all_files.sort()
    all_files = remove_tags(all_files, remove=["meta.csv", "als.csv"])

    # find csvs with retracking date (which are the recent ones)
    new_files = glob.glob("*_{}_*.csv".format(retracking_date))
    new_files.sort()

    if len(new_files) == 0:
        print("no new csv files found with this date - double check date")
        return

    old_files = [file_a for file_a in all_files if file_a not in new_files]

    # ## find split tracks and replace with excluded old track ##
    splits = []
    for track in old_files:
        if "_Range" in track:
            splits.append(track)

    # find original tracks for each movie as these have the timestamps
    orig_tracks = copy.copy(old_files)
    orig_tracks = remove_tags(orig_tracks, remove=["meta.csv", "als.csv", "_Range", "cleaned"])
    orig_tracks.sort()

    # add old timestamps to newly re-tracked data
    for orig_n, orig_file in enumerate(orig_tracks):
        for new_f in new_files:
            if orig_file[0:18] == new_f[0:18]:
                print("updating timestamps of {} and adding exclude tag to {}".format(new_f, orig_file))
                update_csvs(os.path.join(vid_directory, orig_file), os.path.join(vid_directory, new_f))

    old_files_excluding_orig = [file_a for file_a in old_files if file_a not in orig_tracks]

    # add exclude tag to the old files (excluding orig - which were already "exlcuded" in last lines:
    for file in old_files_excluding_orig:
        exclude_tag_csv(os.path.join(vid_directory, file))


if __name__ == '__main__':

    correct_new_tracks = 'm'
    while correct_new_tracks not in {'y', 'n'}:
        correct_new_tracks = input("Correct new tracks? y/n: \n")

    if correct_new_tracks == 'y':
        date_tag = input("What is the date tag for the new tracks? YYYYMMDD: \n")
        # Allows a user to select top directory
        root = Tk()
        root.withdraw()
        root.update()
        rootdir = askdirectory(parent=root, title="Select video folder (which has the movies and tracks)")
        root.destroy()

        correct_tags(date_tag, rootdir)
