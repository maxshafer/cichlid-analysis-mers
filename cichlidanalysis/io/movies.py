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


# def get_movie_paths():
#
#     one_or_more = 'm'
#     while one_or_more not in {'y', 'n'}:
#         one_or_more = input("More than one movie to select? y/n: \n")
#
#     if one_or_more == 'y':
#         video_nums = input("Input video numbers like this: 7, 8, 10: \n")
#         root = Tk()
#         root.withdraw()
#         root.update()
#         rootdir = askdirectory(parent=root)
#         root.destroy()
#
#         videos_path = get_vid_paths_from_nums(rootdir, video_nums)
#
#     else:
#         root = Tk()
#         root.withdraw()
#         root.update()
#         videos_path = [askopenfilename(title="Select movie file",
#                                        filetypes=(("mp4 files", "*.mp4"), ("all files", "*.*")))]
#         root.destroy()
#     return videos_path
