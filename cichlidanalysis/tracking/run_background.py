##########
# This software is licensed under the GNU General Public License v3.0
# Copyright (c) 2019 Annika Nichols
##########
# uses a median on a range of frames to calculate background
import datetime

import cv2
import numpy as np
import PySpin


def background(cam_ID, length, nth_frame, width_trim, height_trim, path, percentile):
    """ (str, int, int, int, str, int)
     This function will create a background image of the defined area using the given percentile"""
    # camera selection and control
    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()

    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()

    # Retrieve the specific camera
    cam = cam_list.GetBySerial(cam_ID)
    print('Device serial number retrieved: %s' % cam_ID)

    # Initialize camera
    cam.Init()
    cam.BeginAcquisition()

    # Retrieve GenICam nodemap
    nodemap = cam.GetNodeMap()

    # set exposure
    ExposureTime = 8
    cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
    cam.ExposureMode.SetValue(PySpin.ExposureMode_Timed)
    cam.ExposureTime.SetValue(ExposureTime * 1e3)

    # Get the current frame rate; acquisition frame rate recorded in hertz
    node_acquisition_framerate = PySpin.CFloatPtr(nodemap.GetNode('AcquisitionFrameRate'))
    framerate_to_set = node_acquisition_framerate.GetValue()
    print('Frame rate to be set to %d' % framerate_to_set)

    # Fetch frame
    image_result = cam.GetNextImage()
    width = image_result.GetWidth()
    height = image_result.GetHeight()

    # settings with Blur settings for the video loop
    gathered_frames = []
    for counter in range(length):
        image = cam.GetNextImage().GetData().reshape((height, width))[0:height_trim, 0:width_trim]
        if counter % nth_frame == 0:
            print("Frame {}".format(counter))
            gathered_frames.append(image)

    background = np.percentile(gathered_frames, percentile, axis=0).astype(dtype=np.uint8)
    cv2.imshow('Calculated Background', background)

    # background = np.percentile(frameMedian, 90, axis=0).astype(dtype=np.uint8)
    # cv2.imshow('Calculated background', background)

    date = datetime.datetime.now().strftime("%Y%m%d")
    cv2.imwrite(path + "/" + date + '_background_{}pc.png'.format(percentile), background)

    cam.EndAcquisition()
    cam.DeInit()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Clear camera list before releasing system
    del cam
    cam_list.Clear()
    system.ReleaseInstance()
    return
