import pandas as pd

def add_metrics(species_sixes, metrics_path):
    """ Loads excel file with metrics about species, only gives back the species asked for by their species_sixes code

    :param species_sixes:
    :param metrics_path: path to xlsx file
    :return:
    """
    metrics = pd.read_excel(metrics_path, engine='openpyxl')

    # get only metrics for given species
    first = True
    species_to_add_at_end = []
    for species_s_i in species_sixes:
        if (metrics.loc[metrics.species_six == species_s_i, 'species_six']).empty:
            species_to_add_at_end.append(species_s_i)
            print('Did not find {}'.format(species_s_i))
        if first:
            sp_metrics = metrics.loc[metrics.species_six == species_s_i]
            first = False
        else:
            sp_metrics = pd.concat([sp_metrics, metrics.loc[metrics.species_six == species_s_i]])

    for species_missing in species_to_add_at_end:
        sp_metrics = pd.concat([sp_metrics, pd.DataFrame({'species_six': species_missing, 'tribe': 'other'}, index=[0])])
    sp_metrics = sp_metrics.reset_index(drop=True)
    return sp_metrics


def tribe_cols():
    """ rgb colours for each tribe as in Ronco 2020

    :return:
    """
    tribe_col = dict({'Trophenini': (133/255, 197/255, 112/255),
                      'Haplochromini': (24/255, 100/255, 51/255),
                      'Perissodini': (249/255, 173/255, 70/255),
                      'Benthochromini': (175/255, 39/255, 44/255),
                      'Cyprichromini': (240/255, 78/255, 42/255),
                      'Ectodini': (156/255, 184/255, 217/255),
                      'Limnochromini': (81/255, 95/255, 166/255),
                      'Cyphotilapiini': (254/255, 223/255, 10/255),
                      'Lamprologini': (194/255, 136/255, 187/255),
                      'Bathybatini': (36/255, 38/255, 37/255),
                      'Trematocarini': (148/255, 142/255, 115/255),
                      'Boulengerochromini': (90/255, 90/255, 90/255),
                      'Eretmodini': (98/255, 54/255, 118/255),
                      'other': (192/255, 192/255, 192/255)})

    return tribe_col
