from tkinter.filedialog import askdirectory
from tkinter import Tk


def select_dir_path():
    """Allows a user to select top directory"""
    root = Tk()
    rootdir = askdirectory(parent=root, title="Select roi folder (which has the movies and tracks)")
    root.destroy()
    return rootdir


def select_top_folder_path():
    """Allows a user to select top directory"""
    root = Tk()
    topdir = askdirectory(parent=root, title="Select top recording folder (which has the camera folders)")
    root.destroy()
    return topdir