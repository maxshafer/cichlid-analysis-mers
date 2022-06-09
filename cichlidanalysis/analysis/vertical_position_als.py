from tkinter.filedialog import askdirectory
from tkinter import *
import warnings

from cichlidanalysis.io.als_files import load_vertical_rest_als_files
from cichlidanalysis.plotting.position_plots import plot_v_position_hists

# debug pycharm problem
warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == '__main__':
    print("Running vertical_+position_als.py which will load *als_vertical_pos_hist_rest-non-rest.csv files and "
          "make plots")
    # Allows user to select top directory and load all als files here
    root = Tk()
    root.withdraw()
    root.update()
    rootdir = askdirectory(parent=root)
    root.destroy()

    vp_hist = load_vertical_rest_als_files(rootdir)
    plot_v_position_hists(rootdir, vp_hist)
