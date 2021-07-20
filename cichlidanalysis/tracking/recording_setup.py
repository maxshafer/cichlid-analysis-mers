# this is the script you run to set up recording folders
import os
import datetime

from tkinter.filedialog import askdirectory
from tkinter import *
import yaml
import shutil
import PySpin
import cv2

from define_roi import define_roi
from run_background import background
from cichlidanalysis.io.meta import load_yaml


def cam_input():
    """input function for finding the cameras to record from"""
    while True:
        cam_num = input("How many cameras would you like to run?: ")

        if cam_num == '6':
            print("Will setup all ", cam_num, " cameras")
            cameras = [[1], [2], [3], [4], [5], [6]]
        else:
            print("Will setup ", cam_num, " cameras")
            cameras = []
            for cam in range(int(cam_num)):
                new_cam = input("Which camera would you like to setup: ")
                cameras.append([int(new_cam)])
        return cameras


def recording_input():
    """input function for asking about fish details"""
    dict_file = {}
    dict_file["species"] = input("species?: ")
    sex = 'n'
    while sex not in {'m', 'f', 'u'}:
        sex = input("sex? m/f/u: \n")
    dict_file["sex"] = sex

    return dict_file


# choose where to save folders
# Allows a user to select top directory
root = Tk()
root.withdraw()
root.update()
rootdir = askdirectory()
root.destroy()

date = datetime.datetime.now().strftime("%Y%m%d")
recdir = os.path.join(rootdir, ("FISH" + date))

# create recording folder
os.mkdir(recdir)

# which cameras to use:
cams = cam_input()

# cam numbers:
cam_n = ['19463369', '19463146', '19503389', '19463356', '19463370', '19503390']

# create a  sub directory for each camera
for camera in cams:
    # Create target Directory
    path = os.path.join(recdir, date + "_c{}".format(str(camera[0])))
    os.mkdir(path)
    print("Directory ", path, " Created ")

    # in each folder make a config file
    cam_ID = cam_n[camera[0] - 1]

    # defining config file
    config_file = {"filechunk": 3600, "fps": 10, "tot_hours": 240, "cam_ID": cam_n[camera[0] - 1]}

    with open(os.path.join(path, "config.yaml"), "w") as file:
        documents = yaml.dump(config_file, file)

    # # Deinitialize camera
    # cam.EndAcquisition()
    # cam.DeInit()
    # cv2.destroyAllWindows()
    # del cam
    # cam_list.Clear()
    # system.ReleaseInstance()

    # define ROIs
    define_roi(cam_ID, path)
    rois = load_yaml(path, "roi_file")

    # make a sub directory for each ROI recording
    for roi in range(len(rois) - 1):
        try:
            # ask for meta data of the fish
            meta_data = recording_input()
            # Create target Directory
            roi_path = os.path.join(path, "FISH{0}_c{1}_r{2}_{3}_s{4}".format(date, str(camera[0]), str(roi),
                                                                              meta_data["species"].replace(' ', '-'),
                                                                              meta_data["sex"]))
            os.mkdir(roi_path)
            print("Directory ", path, " Created ")

            with open(os.path.join(roi_path, "meta_data.yaml"), "w") as file:
                documents = yaml.dump(meta_data, file)
        except FileExistsError:
            print("Directory ", path, " already exists")

    # make background image
    background(cam_ID, 3, 1, 1280, 960, path, percentile=90)

    # add species name to the camera (of the last roi, in each tank it should be consistently the same species)
    shutil.move(path, path + "_" + meta_data["species"].replace(' ', '-'))
