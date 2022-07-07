
def shorten_sp_name(species_full):
    """ Shortens genus name of species e.g. Neolamprologous toae becomes N. toae

    :param species_full:
    :return shortened_sp_names:
    """
    shortened_sp_names = []

    # if one name
    if type(species_full) == str:

        if species_full.find(' ') == -1 and species_full.find('-') > 0:
            splitting_by = '-'
        elif species_full.find(' ') > 0 and species_full.find('-') == -1:
            splitting_by = ' '
        else:
            print("problem, quitting")
            return False
        shortened_sp_names.append(species_full[0] + ". " + species_full.split(splitting_by)[1])

    # if many species names
    else:
        if species_full[0].find(' ') == -1 and species_full[0].find('-') > 0:
            splitting_by = '-'

        elif species_full[0].find(' ') > 0 and species_full[0].find('-') == -1:
            splitting_by = ' '
        else:
            print("problem, quitting")
            return False

        for sp in species_full:
            shortened_sp_names.append(sp[0] + ". " + sp.split(splitting_by)[1])

    return shortened_sp_names


def six_letter_sp_name(species_full):
    """ Shortens genus name of species e.g. Neolamprologous toae becomes Neotoa

    :param species_full:
    :return shortened_sp_names:
    """
    shortened_sp_names = []

    # if one name
    if type(species_full) == str:
        if species_full.find(' ') == -1 and species_full.find('-') > 0:
            splitting_by = '-'
        elif species_full.find(' ') > 0 and species_full.find('-') == -1:
            splitting_by = ' '
        else:
            print("problem, quitting")
            return False
        shortened_sp_names.append(species_full[0:3] + species_full.replace('\'', '').split(splitting_by)[1][0:3])

    # if many species names
    else:
        if species_full[0].find(' ') == -1 and species_full[0].find('-') > 0:
            splitting_by = '-'
        elif species_full[0].find(' ') > 0 and species_full[0].find('-') == -1:
            splitting_by = ' '
        else:
            print("problem, quitting")
            return False

        for sp in species_full:
            shortened_sp_names.append(sp[0:3] + sp.replace('\'', '').split(splitting_by)[1][0:3])

    return shortened_sp_names


def add_species_from_FishID(df):
    fishes = df.FishID
    species = []
    species_six = []
    for fish in fishes:
        sp = fish.split('_')[3].replace('-', ' ')
        species.append(sp)
        species_six.append(six_letter_sp_name(sp)[0])
    df['species'] = species
    df['species_six'] = species_six
    return df


def check_file_name(file_name: str) -> None:
    """ File name should be structured with: FISHYYYYMMDD_c#_r#_species-names_sX
    Where # = number and X is the sex (or l for light) and separated by "_" """
    file_split = file_name.split("_")

    if not len(file_split) == 5:
        raise ValueError("recording name is incorrectly formatted, "
                         "should be FISHYYYYMMDD_c#_r#_species-names_sX")
    if not len(file_split[0]) == 12:
        raise ValueError("recording name is not the right length. It should be FISHYYYYMMDD")
    if not file_split[1][0] == 'c':
        raise ValueError("recording name is incorrectly formatted at the camera position, "
                         "should be FISHYYYYMMDD_c#_r#_species-names_sX")
    if not file_split[2][0] == 'r':
        raise ValueError("recording name is incorrectly formatted at the roi position, "
                         "should be FISHYYYYMMDD_c#_r#_species-names_sX")
    if not file_split[4][0] == 's':
        raise ValueError("recording name is incorrectly formatted at the sex position, "
                         "should be FISHYYYYMMDD_c#_r#_species-names_sX")
    return


def get_roi_from_fish_id(fish_id):
    """ Gets roi from name"""
    roi_n = fish_id.split("_")[2][1]

    return roi_n
