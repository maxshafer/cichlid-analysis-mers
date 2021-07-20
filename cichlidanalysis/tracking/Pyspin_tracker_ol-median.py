##########
# This software is licensed under the GNU General Public License v3.0
# Copyright (c) 2019 Vassilis Bitsikas
# Modified by Annika Nichols
##########
import time
import datetime
import os

import cv2
import numpy as np
import imageio
import PySpin
import glob

from tkinter.filedialog import askdirectory
from tkinter import *

from cichlidanalysis.io.meta import load_yaml

# Allows a user to select a directory
root = Tk()
root.withdraw()
root.update()
rootdir = askdirectory()
root.destroy()

# load yaml config file
params = load_yaml(rootdir, "config")

# find and load background file
if len(glob.glob(os.path.join(rootdir, "*.png"))) != 1:
    print('too many or too few background files in folder:' + rootdir)
    sys.exit()
else:
    background_full = cv2.imread(glob.glob(os.path.join(rootdir, "*.png"))[0], 0)

# threshold position (param file??)
t_pos = 35
date = datetime.datetime.now().strftime("%Y%m%d")

width_trim = 1280
height_trim = 960
ExposureTime = 8
percentile = 90

# find and load roi file
use_full_roi = False
if len(glob.glob(rootdir + "\\roi_file*")) != 1:
    print("too many or too few roi files in folder:" + rootdir)
    cont = input("continue with full ROI? y/n")
    if cont == "n":
        sys.exit()
    else:
        print("continuing to run with full ROI")
        use_full_roi = True
        rois = {'roi_0': (0, 0, width_trim, height_trim)}
else:
    rois = load_yaml(rootdir, "roi_file")

# cam numbers:
cam_n = ['19463146', '19463369', '19463356', '19503389', '19503390', '19463370']

camera = cam_n.index(params["cam_ID"])
roi_path = [f.path for f in os.scandir(rootdir) if f.is_dir()]

#############################
#############################
#############################

# camera selection and control #

# Retrieve singleton reference to system object
system = PySpin.System.GetInstance()
# Retrieve list of cameras from the system
cam_list = system.GetCameras()
# Retrieve the specific camera
cam = cam_list.GetBySerial(params["cam_ID"])
print('Device serial number retrieved: %s' % params["cam_ID"])
# Initialize camera
cam.Init()
# Retrieve GenICam nodemap
nodemap = cam.GetNodeMap()

# set acquisition. Continuous acquisition. Auto exposure off. Set frame rate.
# In order to access the node entries, they have to be casted to a pointer type (CEnumerationPtr here)
node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
    print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')

# Retrieve entry node from enumeration node
node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(node_acquisition_mode_continuous):
    print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')

# Retrieve integer value from entry node
acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

# Set integer value from entry node as new value of enumeration node
node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

print('Acquisition mode set to continuous...')

# Set frame rate.
acquisition_frame_rate_auto = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionFrameRateAuto"))
acquisition_frame_rate_auto_off_node = acquisition_frame_rate_auto.GetEntryByName("Off")
acquisition_frame_rate_auto_off = acquisition_frame_rate_auto_off_node.GetValue()
acquisition_frame_rate_auto.SetIntValue(acquisition_frame_rate_auto_off)
if cam.AcquisitionFrameRate.GetAccessMode() == PySpin.RW:
    cam.AcquisitionFrameRate.SetValue(params["fps"])
else:
    raise SystemExit("You need to enable frame rate control: go to SpinView, Features, "
                     "Acquisition control and make sure Acquisition Frame Rate Control "
                     "Enabled is checked")

#set exposure
cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
cam.ExposureMode.SetValue(PySpin.ExposureMode_Timed)
cam.ExposureTime.SetValue(ExposureTime * 1e3)


# Retrieve Stream Parameters device nodemap
s_node_map = cam.GetTLStreamNodeMap()

# Retrieve Buffer Handling Mode Information
handling_mode = PySpin.CEnumerationPtr(s_node_map.GetNode('StreamBufferHandlingMode'))
if not PySpin.IsAvailable(handling_mode) or not PySpin.IsWritable(handling_mode):
    raise SystemExit('Unable to set Buffer Handling mode (node retrieval). Aborting...\n')

handling_mode_entry = PySpin.CEnumEntryPtr(handling_mode.GetCurrentEntry())
if not PySpin.IsAvailable(handling_mode_entry) or not PySpin.IsReadable(handling_mode_entry):
    raise SystemExit('Unable to set Buffer Handling mode (Entry retrieval). Aborting...\n')

# Set stream buffer Count Mode to manual
stream_buffer_count_mode = PySpin.CEnumerationPtr(s_node_map.GetNode('StreamBufferCountMode'))
if not PySpin.IsAvailable(stream_buffer_count_mode) or not PySpin.IsWritable(stream_buffer_count_mode):
    raise SystemExit('Unable to set Buffer Count Mode (node retrieval). Aborting...\n')

stream_buffer_count_mode_manual = PySpin.CEnumEntryPtr(stream_buffer_count_mode.GetEntryByName('Manual'))
if not PySpin.IsAvailable(stream_buffer_count_mode_manual) or not PySpin.IsReadable(stream_buffer_count_mode_manual):
    raise SystemExit('Unable to set Buffer Count Mode entry (Entry retrieval). Aborting...\n')

stream_buffer_count_mode.SetIntValue(stream_buffer_count_mode_manual.GetValue())
print('Stream Buffer Count Mode set to manual...')

# Retrieve and modify Stream Buffer Count
buffer_count = PySpin.CIntegerPtr(s_node_map.GetNode('StreamBufferCountManual'))
if not PySpin.IsAvailable(buffer_count) or not PySpin.IsWritable(buffer_count):
    raise SystemExit('Unable to set Buffer Count (Integer node retrieval). Aborting...\n')

# Display Buffer Info
print('\nDefault Buffer Handling Mode: %s' % handling_mode_entry.GetDisplayName())
print('Default Buffer Count: %d' % buffer_count.GetValue())
print('Maximum Buffer Count: %d' % buffer_count.GetMax())

buffer_count.SetValue(50)

handling_mode_entry = handling_mode.GetEntryByName('OldestFirstOverwrite')
handling_mode.SetIntValue(handling_mode_entry.GetValue())
print('\n\nBuffer Handling Mode has been set to %s' % handling_mode_entry.GetDisplayName())


frame = 0
data = list()
time_counter = 0
t0 = time.perf_counter()
movie_number = 0
frame_chunk = 0
frame_id = 0

print("starting image aquisition")
# Fetch frame
cam.BeginAcquisition()
image_result = cam.GetNextImage()
width = image_result.GetWidth()
height = image_result.GetHeight()


# create track bar for setting threshold
def nothing(x):
    pass


# create window
cv2.namedWindow("{} - Live thresholded".format(params["cam_ID"]))

# create threshold track bar
cv2.createTrackbar('threshold', "{} - Live thresholded".format(params["cam_ID"]), t_pos, 255, nothing)

# Create roi specific cropped background in list
background = list()
for roi in range(0, len(rois) - 1):
    curr_roi = rois["roi_" + str(roi)]
    background.append(background_full.reshape((height_trim, width_trim))[curr_roi[1]:curr_roi[1] + curr_roi[3],
                      curr_roi[0]:curr_roi[0] + curr_roi[2]])

# hours to record total
end_time = datetime.datetime.now() + datetime.timedelta(hours=params["tot_hours"])

problem = 0
# Start the loop
while 1 + params["fps"] * params["tot_hours"] * 60 * 60 > frame_id:
    try:
        next_image = cam.GetNextImage(1000)
    except PySpin.SpinnakerException:
        # print(next_image.GetImageStatus())
        print("Problem retrieving from camera, frame_ID prior: {0}".format(frame_id))
        if problem == 2:
            s_node_map = cam.GetTLStreamNodeMap()
            buffer_count = PySpin.CIntegerPtr(s_node_map.GetNode('StreamTotalBufferCount'))
            print('Stream Total Buffer Count: %d' % buffer_count.GetValue())
            for roi in range(0, len(rois) - 1):
                writers[roi].close()
            break
        else:
            problem += 1
            continue

    # next_image = cam.GetNextImage()
    frame_id = next_image.GetFrameID()
    frame_timestamp = next_image.GetTimeStamp()

    if frame_chunk % 3000 == 0:
        ## Display Buffer Info
        print("Frame_ID: {}".format(frame_id))
        s_node_map = cam.GetTLStreamNodeMap()
        buffer_count = PySpin.CIntegerPtr(s_node_map.GetNode('StreamTotalBufferCount'))
        print('Stream Total Buffer Count: %d' % buffer_count.GetValue())
        buffer_count = PySpin.CIntegerPtr(s_node_map.GetNode('StreamBufferUnderrunCount'))
        print('Stream Buffer Underrun Count (dropped frames): %d' % buffer_count.GetValue())

    full_image = next_image.GetData().reshape((height, width))[0:height_trim, 0:width_trim]
    next_image.Release()
    image = list()
    for roi in range(0, len(rois) - 1):
        # for the first frame define an ROI and crop image
        curr_roi = rois["roi_" + str(roi)]
        image.append(full_image[curr_roi[1]:curr_roi[1] + curr_roi[3],
                     curr_roi[0]:curr_roi[0] + curr_roi[2]])

    if time_counter == 0:
        date_f = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # as there can be multiple rois the writer, data and moviename are kept in lists
        movie_names = list()
        writers = list()
        data = list()
        background_frames = []
        for roi in range(0, len(rois) - 1):
            # movie_names.append(rootdir + r"\\{}\{}_{}_roi-{}.mp4".format("FISH" + date + "_roi_" + str(roi),
            #                                                                     date_f, movie_number, roi))
            # os.makedirs(os.path.dirname(movie_names[roi]), exist_ok=True)
            movie_names.append(roi_path[roi] + r"\\{}_{}_roi-{}.mp4".format(date_f, str(movie_number).zfill(3), roi))
            writers.append(imageio.get_writer(movie_names[roi], fps=params["fps"], codec="libx264"))
            data.append(list())

    if frame_chunk % 200 == 0:
        background_frames.append(full_image)

    # measures the time between this and last call
    t1 = time.perf_counter()
    dt = t1 - t0
    t0 = t1

    current_time = datetime.datetime.now().strftime("%y%m%d%H%M%S%f")
    t_pos = cv2.getTrackbarPos('threshold', "{} - Live thresholded".format(params["cam_ID"]))

    # tracking
    cX = list()
    cY = list()
    contourOI = list()
    contourOI_ = list()
    image_thresholded = list()
    ##### RECONRDING vs TRACKING roi! (same)
    for roi in range(0, len(rois) - 1):
        frameDelta = cv2.absdiff(background[roi], image[roi])
        image_thresholded.append(cv2.threshold(frameDelta, t_pos, 255, cv2.THRESH_TOZERO)[1])
        (contours, _) = cv2.findContours(image_thresholded[roi].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contourOI_.append(max(contours, key=cv2.contourArea))
            area = cv2.contourArea(contourOI_[roi])
            if area > 100:
                contourOI.append(cv2.convexHull(contourOI_[roi]))
                M = cv2.moments(contourOI[roi])
                cX.append(int(M["m10"] / M["m00"]))
                cY.append(int(M["m01"] / M["m00"]))
                data[roi].append((frame_timestamp, cX[roi], cY[roi], area))
            else:
                # print("no large enough contour found for roi {}!".format(roi))
                data[roi].append((frame_timestamp, np.nan, np.nan, np.nan))
                contourOI_[-1] = False
                contourOI.append(False)
                cX.append(np.nan)
                cY.append(np.nan)
        else:
            print("no contour found for roi {}!".format(roi))
            data[roi].append((frame_timestamp, np.nan, np.nan, np.nan))
            contourOI_.append(False)
            contourOI.append(False)
            cX.append(np.nan)
            cY.append(np.nan)


        # write image to movie file and tracking data to data file
        writers[roi].append_data(image[roi])

    full_frameDelta = cv2.absdiff(background_full, full_image)
    full_image_thresholded = cv2.threshold(full_frameDelta, t_pos, 255, cv2.THRESH_TOZERO)[1]

    # Live display of full resolution and ROIs
    cv2.putText(full_image_thresholded,
                "dt: {:.4f}, time: {}".format(dt, datetime.datetime.now().strftime("%y.%m.%d %H:%M:%S.%f")),
                (30, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=255)
    cv2.putText(full_image_thresholded, "Framenum: {}".format(frame), (30, full_image_thresholded.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=255)

    for roi in range(0, len(rois) - 1):
        curr_roi = rois["roi_" + str(roi)]
        if np.all(contourOI_[roi] != False):
            curr_roi = rois["roi_" + str(roi)]
            # add in contours
            corrected_contour = np.empty(contourOI_[roi].shape)
            corrected_contour[:, 0, 0] = contourOI_[roi][:, 0, 0] + curr_roi[0]
            corrected_contour[:, 0, 1] = contourOI_[roi][:, 0, 1] + curr_roi[1]
            cv2.drawContours(full_image_thresholded, corrected_contour.astype(int), -1, 255, 1)

            # add in centroid
            cv2.circle(full_image_thresholded, (cX[roi] + curr_roi[0], cY[roi] + curr_roi[1]), 8, 255, 1)

        # add in ROIs
        start_point = (curr_roi[0], curr_roi[1])
        end_point = (curr_roi[0] + curr_roi[2], curr_roi[1] + curr_roi[3])
        cv2.rectangle(full_image_thresholded, start_point, end_point, 220, 2)

    cv2.imshow("{} - Live thresholded".format(params["cam_ID"]), full_image_thresholded)
    # cv2.imshow("{} - Live".format(params["cam_ID"]), full_image)

    if cv2.waitKey(1) == 27:
        print("esc pressed, exiting")
        break

    # saving a chunk of data
    frame += 1
    time_counter += dt
    frame_chunk += 1
    if frame_chunk == (params["filechunk"])*params["fps"]:
        frame_chunk = 0
        print("Saving a movie and data output")

        for roi in range(0, len(rois) - 1):
            # saving out tracking data
            datanp = np.array(data[roi])
            filename = (roi_path[roi] + r"\\{}_{}_roi-{}.csv".format(date_f, str(movie_number).zfill(3), roi))
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            np.savetxt(filename, datanp, delimiter=",")

            writers[roi].close()

        # make next background
        background_full = np.percentile(background_frames, percentile, axis=0).astype(dtype=np.uint8)
        filename = (rootdir + r"\\{}_{}_per{}_Background.png".format(date_f, str(movie_number).zfill(3), percentile))
        cv2.imwrite(filename, background_full)

        # Create roi specific cropped medians in list
        background = list()
        for roi in range(0, len(rois) - 1):
            curr_roi = rois["roi_" + str(roi)]
            background.append(background_full.reshape((height_trim, width_trim))[curr_roi[1]:curr_roi[1] + curr_roi[3],
                              curr_roi[0]:curr_roi[0] + curr_roi[2]])

        # reset for the next chunk of recording
        time_counter = 0
        movie_number += 1


print("Recording finished, cleaning up")
# cleaning up
cam.EndAcquisition()
cam.DeInit()

# Clear camera list before releasing system
del cam
cam_list.Clear()
system.ReleaseInstance()
