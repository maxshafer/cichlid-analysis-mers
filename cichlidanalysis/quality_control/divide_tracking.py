# this script will ask for a movie file and a time frame to sub divide. It will then create backgrounds and track
# each epoque

import os
import glob
import datetime

import numpy as np

from cichlidanalysis.io.meta import load_yaml, extract_meta
from cichlidanalysis.io.tracks import load_track, get_latest_tracks
from cichlidanalysis.io.movies import get_movie_paths
from cichlidanalysis.quality_control.split_tracking import background_vid_split
from cichlidanalysis.tracking.offline_tracker import tracker

def divide_video(video_path, chunk_size=20, fps=10):

    vid_folder_path = os.path.split(video_path)[0]
    cam_folder_path = os.path.split(vid_folder_path)[0]
    vid_folder_name = os.path.split(vid_folder_path)[1]

    os.chdir(vid_folder_path)
    video_name = os.path.split(video_path)[1]

    # load original track (need timestamps)
    # orig_file = glob.glob("*{}.csv".format(video_name[0:-4]))
    _, latest_file = get_latest_tracks(vid_folder_path, video_name[0:-4])
    na, track_single_orig = load_track(os.path.join(vid_folder_path, latest_file[0]))

    meta = load_yaml(vid_folder_path, "meta_data")
    rois = load_yaml(cam_folder_path, "roi_file")
    config = load_yaml(cam_folder_path, "config")
    fish_data = extract_meta(vid_folder_name)

    os.chdir(cam_folder_path)
    files = glob.glob("*.png")
    files.sort()
    files.insert(0, files.pop(files.index(min(files, key=len))))

    roi_n = rois["roi_" + fish_data['roi'][1]]

    # make the chunks
    chunks = np.arange(0, track_single_orig.shape[0]+1, int(chunk_size)*fps*60)
    if chunks[-1] != track_single_orig.shape[0]:
        chunks[-1] = track_single_orig.shape[0]
        print("correcting last timepoint")

    # making roi for full video
    width_trim, height_trim = rois['roi_{}'.format(fish_data['roi'][-1])][2:4]
    vid_rois = {'roi_0': (0, 0, width_trim, height_trim)}
    area_s = 100
    thresh = 35

    print("remaking backgrounds and retracking")
    for chunk_n in np.arange(0, len(chunks)-1):
        split_ends = [chunks[chunk_n], chunks[chunk_n+1]]
        background = background_vid_split(video_path, 100, 90, split_ends)

        # in case a new ROI has been used for tracking, use this.
        crop_vid_rois = load_yaml(vid_folder_path, "roi_file")
        if crop_vid_rois:
            vid_rois = crop_vid_rois
            print("tracking with new roi")
        tracker(video_path, background, vid_rois, threshold=thresh, display=False, area_size=area_s,
                split_range=split_ends)

        # add in the right timepoints (of a primary track - not a full retrack)
        # load the newly tracked csv
        date = datetime.datetime.now().strftime("%Y%m%d")
        range_s = str(split_ends[0]).zfill(5)
        range_e = str(split_ends[1]).zfill(5)
        filename = video_path[0:-4] + "_tracks_{}_Thresh_{}_Area_{}_Range{}-{}_.csv".format(date, thresh, area_s,
                                                                                            range_s, range_e)
        _, track_single_split = load_track(filename)
        # replace the frame col with the data from the original track (track index always starts from 0)
        track_single_split[0:split_ends[1] - split_ends[0], 0] = track_single_orig[split_ends[0]: split_ends[1], 0]

        # save over
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.savetxt(filename, track_single_split, delimiter=",")


if __name__ == '__main__':
    # find file path for video and load track
    # Allows a user to select file
    fps = 10

    videos_path, _, _ = get_movie_paths()

    chunk_size = '11'
    while 60 % int(chunk_size) != 0:
        chunk_size = input("Retrack the movie in smaller chunks? \nChunk size in min (60 needs to be divisible by it)?:")

    for video_path in videos_path:
        divide_video(video_path, chunk_size)

