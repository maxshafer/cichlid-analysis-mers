import pandas as pd
from tkinter.filedialog import askdirectory, askopenfilename
from tkinter import Tk


def get_ronco_paths():
    """ Hard coded paths for Ronco data on Annika's macbook

    :return:
    """
    # paths for Ronco 2021 data
    ronco_data_path = "/Volumes/BZ/RG Schier/Scientific Data/Cichlid-genomes/Fig2_data.csv"
    # https://www.nature.com/articles/s41586-020-2930-4#Sec30

    # voucher_key_path = "/Volumes/BZ/RG Schier/Scientific Data/Cichlid-genomes/01_specimen_voucher_key.csv"
    # pigmentation_pattern_path = "/Volumes/BZ/RG Schier/Scientific Data/Cichlid-genomes/06_scores_pigmentation_pattern.csv"
    # stable_isotope_path = "/Volumes/BZ/RG Schier/Scientific Data/Cichlid-genomes/06_stable_isotope_data.csv"
    #
    # lower_pharyngeal_jaw_path = "/Volumes/BZ/RG Schier/Scientific Data/Cichlid-genomes/06_landmark_data_lower_pharyngeal_jaw.tps"
    # body_shape_path = "/Volumes/BZ/RG Schier/Scientific Data/Cichlid-genomes/06_landmark_data_body_shape.tps"

    return ronco_data_path


if __name__ == '__main__':
    ronco_data_path = get_ronco_paths()

    root = Tk()
    root.withdraw()
    root.update()
    vid_dir = askdirectory()
    root.destroy()

    ronco_data = pd.read_csv(ronco_data_path)