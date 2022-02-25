import os

from tkinter.filedialog import askdirectory
from tkinter import Tk
import datetime as dt
import numpy as np
import pandas as pd

from cichlidanalysis.io.als_files import load_als_files

if __name__ == '__main__':
    ' this script goes through each subfolder and loads _als.csv files. It will check the length of the data in ' \
    ' days. This is  used since there seems to be a silent bug which causes only part of the als file to be saved. ' \
    ' Have added a check on run_fish_als to prevent this issue in the future. This script will save out a csv with ' \
    ' the fishIDs and the day lengths'

    # Allows a user to select top directory
    root = Tk()
    root.withdraw()
    root.update()
    topdir = askdirectory(parent=root, title="Select folder which  contains the species folders")
    root.destroy()

    list_subfolders_with_paths = [f.path for f in os.scandir(topdir) if f.is_dir()]

    all_day_lens = []
    all_fishIDs = []
    counter = 0
    for folder in list_subfolders_with_paths:
        fish_tracks = load_als_files(folder)
        fps = 10
        day_ns = 24 * 60 * 60 * fps ** 9

        species_day_lens = []
        if isinstance(fish_tracks, pd.DataFrame):
            for fish in fish_tracks.FishID.unique().tolist():
                day_len = np.max(fish_tracks.loc[fish_tracks.FishID == fish, "tv_ns"]) / day_ns
                species_day_lens.append(np.round(day_len, 2))

            print("Species {} data is spread from {} to {}  days".format(fish.split('_')[3], np.min(species_day_lens),
                                                                         np.max(species_day_lens)))
            all_day_lens.extend(species_day_lens)
            all_fishIDs.extend(fish_tracks.FishID.unique().tolist())

        # save every tenth fish (as it takes a long time to run)
        counter = counter + 1
        if counter == 10:
            dic = {"fishID": all_fishIDs, "day_lens": all_day_lens}
            df = pd.DataFrame(data=dic)
            df.to_csv(os.path.join(topdir, "_lengths_of_recordings_{}.csv".format(dt.date.today())))
            counter = 0

    # final save
    dic = {"fishID": all_fishIDs, "day_lens": all_day_lens}
    df = pd.DataFrame(dic)
    df.to_csv(os.path.join(topdir, "_lengths_of_recordings_{}.csv".format(dt.date.today())))
