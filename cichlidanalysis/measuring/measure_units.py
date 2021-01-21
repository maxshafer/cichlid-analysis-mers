# this file will ask you to load a movie/median. it will then ask you to draw the length of the barrier 3 times.
# It will take the average length and add this to the config file?

# https://stackoverflow.com/questions/16195190/python-cv2-how-do-i-draw-a-line-on-an-image-with-mouse-then-return-line-coord
# def measure_barrier():
#     ''''''

import os
from tkinter.filedialog import askopenfilename
from tkinter import *

import cv2
import numpy as np
import yaml

from cichlidanalysis.io.meta import load_yaml


def measuring(img):
    line_start = []
    line_end = []
    finished = 0

    def on_mouse(event, x, y, flag, param):
        line_start = param[0]
        line_end = param[1]
        if event == cv2.EVENT_LBUTTONDOWN:
            # print("Start Mouse Position: " + str(x) + ", " + str(y))
            sbox = [x, y]
            line_start.append(sbox)
        elif event == cv2.EVENT_LBUTTONUP:
            # print("End Mouse Position: " + str(x) + ", " + str(y))
            ebox = [x, y]
            line_end.append(ebox)
            cv2.line(img, tuple(line_start[-1]), tuple(ebox), (0, 255, 0), thickness=2)


    cv2.namedWindow('Measuring units')

    while 1:
        cv2.setMouseCallback('Measuring units', on_mouse, [line_start, line_end])
        cv2.imshow('Measuring units', img)
        k = cv2.waitKey(33)

        if len(line_end) > 2:
            cv2.imshow('Measuring units', img)
            k = cv2.waitKey(2000)
            print("2_{}".format(k))
            if k != ord('u'):
                finished = 1

        if k == 27:
            cv2.destroyAllWindows()
            break

        elif k == ord('u'):
            print("3_{}".format(k))
            cv2.line(img, tuple(line_start[-1]), tuple(line_end[-1]), (255, 255, 255), thickness=2)
            line_start.pop(-1)
            line_end.pop(-1)

        elif k == 13 or finished:
            # find displacement
            line_start = np.asarray(line_start)
            line_end = np.asarray(line_end)

            b = line_end[:, 0] - line_start[:, 0]
            c = line_end[:, 1] - line_start[:, 1]
            line_length_pixels = np.sqrt(b ** 2 + c ** 2)

            cv2.destroyAllWindows()
            print(line_length_pixels)
            return line_length_pixels


def main():
    # Allows a user to select a directory
    root = Tk()
    root.withdraw()
    root.update()
    filepath = askopenfilename(title="Select background file", filetypes=(("png files", "*.png"), ("all files", "*.*")))
    root.destroy()

    img = cv2.imread(filepath)

    line_length_pixels = measuring(img)

    divider_base_mm = 15
    mm_per_pixel = divider_base_mm / line_length_pixels
    mm_per_pixel = np.mean(mm_per_pixel)

    print("mm_per_pixel: {}".format(mm_per_pixel))

    parts = os.path.split(filepath)

    config_file = load_yaml(parts[0], "config")
    config_file["mm_per_pixel"] = mm_per_pixel.item()
    print("mm per pixel: {}".format(mm_per_pixel))

    with open(os.path.join(parts[0], "config.yaml"), "w") as file:
        documents = yaml.dump(config_file, file)
