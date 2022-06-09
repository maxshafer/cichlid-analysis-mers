import cv2.cv2 as cv2
import numpy as np


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
    if np.shape(gatheredFramess)[0] > 4:
        background = np.percentile(gatheredFramess, int(percentile), axis=0).astype(dtype=np.uint8)
        cv2.imshow('Calculated Background from {} percentile'.format(percentile), background)
        vid_name = videofilepath[0:-4]
        print("saving background")
        range_s = str(split_range[0]).zfill(5)
        range_e = str(split_range[1]).zfill(5)
        cv2.imwrite('{0}_per{1}_frame{2}-{3}_background.png'.format(vid_name, percentile, range_s, range_e), background)
    else:
        print("not enough frames to make a good background (min 5)")
        background = []

    cap.release()
    cv2.destroyAllWindows()
    return background
