
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
