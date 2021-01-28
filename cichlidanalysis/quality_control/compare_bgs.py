## Script to help look at tracking QC

from tkinter import Tk
from tkinter.filedialog import askopenfilename

import matplotlib.pyplot as plt
import cv2.cv2 as cv2

from cichlidanalysis.tracking.backgrounds import background_vid


def compare_backgrounds(video_path, frame_n):
    background_50 = background_vid(video_path, 200, 50)
    background_90 = background_vid(video_path, 200, 90)
    background_95 = background_vid(video_path, 200, 95)

    cap = cv2.VideoCapture(video_path)

    for i in range(frame_n):
        ret, frame = cap.read()

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fig, axs = plt.subplots(3, 3)
    axs[0, 0].imshow(background_50)
    axs[0, 1].imshow(background_95)
    axs[0, 2].imshow(background_90)

    axs[1, 0].imshow(cv2.absdiff(image, background_50))
    axs[1, 1].imshow(cv2.absdiff(image, background_95))
    axs[1, 2].imshow(cv2.absdiff(image, background_90))

    axs[2, 0].imshow(cv2.absdiff(image, background_50) > 35)
    axs[2, 1].imshow(cv2.absdiff(image, background_95) > 35)
    axs[2, 2].imshow(cv2.absdiff(image, background_90) > 35)

    for i in range(3):
        for j in range(3):
            axs[i, j].axis('off')

    axs[0, 0].title.set_text('percentile 50')
    axs[0, 1].title.set_text('percentile 95')
    axs[0, 2].title.set_text('percentile 90')

    image_b = cv2.GaussianBlur(image, (7, 7), 0)
    background_90_b = cv2.GaussianBlur(background_90, (7, 7), 0)

    fig, axs = plt.subplots(3, 3)
    axs[0, 0].imshow(image)
    axs[0, 1].imshow(image)
    axs[0, 2].imshow(image_b)
    axs[1, 0].imshow(cv2.absdiff(image, background_90))
    axs[1, 1].imshow(cv2.absdiff(image_b, background_90_b))
    axs[1, 2].imshow(cv2.absdiff(image, background_90_b))
    axs[2, 0].imshow(cv2.absdiff(image, background_90) > 35)
    axs[2, 1].imshow(cv2.absdiff(image, background_90_b) > 35)
    axs[2, 2].imshow(cv2.absdiff(image_b, background_90_b) > 35)

    for i in range(3):
        for j in range(3):
            axs[i, j].axis('off')

    axs[0, 0].title.set_text('no blurring')
    axs[0, 1].title.set_text('background blurred')
    axs[0, 2].title.set_text('both blurred')


if __name__ == '__main__':
    # Allows a user to select file
    root = Tk()
    root.withdraw()
    root.update()
    video_path = askopenfilename(title="Select movie file", filetypes=(("mp4 files", "*.mp4"), ("all files", "*.*")))
    root.destroy()

    compare_backgrounds(video_path, 400)

