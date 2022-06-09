from tkinter.filedialog import askopenfilename, askdirectory
from tkinter import Tk

from cichlidanalysis.io.tracks import get_file_paths_from_nums


def get_movie_paths():

    video_nums = input("Input video numbers like this: 7, 8, 10: \n")
    root = Tk()
    rootdir = askdirectory(parent=root)
    root.destroy()

    videos_path = get_file_paths_from_nums(rootdir, video_nums, file_format='*.mp4')

    return videos_path, rootdir, video_nums
