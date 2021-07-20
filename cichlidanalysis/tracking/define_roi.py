import os

import cv2
import PySpin
import yaml

def roi_input():
    """input function for 1. asking how many ROIs"""
    while True:
        roi_nums = input("How many ROIs would you like to select?: ")
        try:
            rois = int(roi_nums)
            print("Will do", roi_nums, "region/s of interest")
            return rois
        except ValueError:
            print("Input must be an integer")


def define_roi(cam_ID, folder_path):
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

    roi_num = roi_input()
    # rr = np.arange(4 * roi_num).reshape(roi_num, 4)
    scalingF = 1
    dict_file = {"cam_ID": cam_ID}
    if roi_num == 0:
        test_image = cam.GetNextImage()
        img_conv = test_image.Convert(PySpin.PixelFormat_BGR8, PySpin.HQ_LINEAR)
        height, width = img_conv.GetNDArray().shape[0:2]
        dict_file["roi_0"] = tuple([0, 0, width, height])
    else:
        for roi in range(roi_num):
            test_image = cam.GetNextImage()
            img_conv = test_image.Convert(PySpin.PixelFormat_BGR8, PySpin.HQ_LINEAR)
            height, width = img_conv.GetNDArray().shape[0:2]
            frameRS = cv2.resize(img_conv.GetNDArray(), (int(width / scalingF), int(height / scalingF)))
            rr = cv2.selectROI(("Select ROI" + str(roi)), frameRS)
            # output: (x,y,w,h)
            # add padding to make sure it is divisible by macro_block_size = 16
            rrr = list(tuple(i * scalingF for i in rr))
            for i in [2, 3]:
                if rrr[i] % 16 != 0:
                    rrr[i] = rrr[i] + (16-rrr[i] % 16)

            dict_file["roi_" + str(roi)] = tuple(rrr)
            cv2.destroyAllWindows()
    print(dict_file)

    with open(os.path.join(folder_path, "roi_file.yaml"), "w") as file:
        documents = yaml.dump(dict_file, file)

    print("File has now been saved in specified folder as roi_file.yaml")

    # Deinitialize camera
    cam.EndAcquisition()
    cam.DeInit()
    # Clear camera list before releasing system
    del cam
    cam_list.Clear()
    system.ReleaseInstance()
