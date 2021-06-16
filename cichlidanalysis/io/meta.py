import os
import glob

import yaml
import pandas as pd


def load_yaml(rootdir, name):
    """ load yaml configuration file
    (str, str) -> (dict)
    finds name.yaml file in given dir and opens and returns it as a dict"""
    try:
        filename = os.path.join(rootdir, name + ".yaml")
        print(filename)
        with open(filename) as file:
            params = yaml.load(file, Loader=yaml.FullLoader)
            print(params)
            return params
    except:
        print("couldn't find " + name + ".yaml in folder: " + rootdir)
        params = {}
        return params


def extract_meta(rec_name):
    id_pos = {'date': 0, 'camera': 1, 'roi': 2, 'species': 3, 'sex': 4}
    fish_data = {}
    for ID in id_pos:
        fish_data[ID] = rec_name.split("_")[id_pos[ID]]
    return fish_data


def load_meta_files(folder):
    os.chdir(folder)
    files = glob.glob("*meta.csv")
    files.sort()
    first_done = 0

    for file in files:
        if first_done:
            data = pd.read_csv(os.path.join(folder, file), sep=',', index_col=0)
            print("loaded file {}".format(file))
            meta_i = pd.concat([meta_i, data.T], axis=1)
        else:
            # inititate data frames for each of the fish, beside the time series, also add in the species name and ID at the start
            data = pd.read_csv(os.path.join(folder, file), sep=',', index_col=0)
            print("loaded file {}".format(file))
            meta_i = data.T
            first_done = 1

    new_header = meta_i.loc["ID"]  # grab the first row for the header
    new_df = meta_i[1:]  # take the data less the header row
    new_df.columns = new_header
    return new_df


def add_meta_from_name(df_i: pd.core.frame.DataFrame, row_name: str):
    """
    :param df_i: pd.core.frame.DataFrame
    :param row_name: str
    :return: new_row
    """
    if row_name in ['date', 'camera', 'roi', 'species', 'sex']:
        ID_pos = {'date': 0, 'camera': 1, 'roi': 2, 'species': 3, 'sex': 4}
        cols = list(df_i)
        new_row = pd.DataFrame(index=[row_name], columns=cols)
        for col_name in cols:
            new_row.loc[row_name, col_name] = col_name.split("_")[ID_pos[row_name]]
        return new_row
    else:
        print("can't retrieve that value, only these are possible:")
        for key in ID_pos.keys():
            print(key)


def check_fish_species(fishid, species_name):
    ''' takes a list of fish_IDs and a species name, and checks to see if the fish_ID is of that species.
    Returns list of the fish_IDs which are of that species'''
    fish_of_species = []
    species_name = species_name.replace(' ', '-')
    for fish_n in fishid:
        if fish_n.split("_")[3] == species_name:
            fish_of_species.append(fish_n)
    return fish_of_species


def add_sex(path):
    """ takes path for meta data, loads it, checks the current sex and lets you decide if you'd like to change it.
    Saves the meta  file back """
    meta = load_yaml(path, "meta_data")

    change_sex = 'y'
    while change_sex not in {'m', 'f', 'u', 'n'}:
        change_sex = input("Current sex of fish is -{}-.Change sex of fish? m/f/u or n to leave as is: \n".
                           format(meta["sex"]))

    if change_sex == 'n':
        print("leaving sex as is")
    else:
        meta["sex"] = change_sex

        with open(os.path.join(path, "meta_data.yaml"), "w") as file:
            yaml.dump(meta, file)
    return
