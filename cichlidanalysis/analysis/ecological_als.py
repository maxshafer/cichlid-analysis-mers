import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from cichlidanalysis.analysis.linear_regression import run_linear_reg, plt_lin_reg


def diet_vs_size(rootdir, col_vector, cichlid_meta, feature_v_mean):
    diets = col_vector.unique()
    for diet in diets[~pd.isna(diets)]:
        diet_species = cichlid_meta.loc[cichlid_meta.diet == diet, 'six_letter_name_Ronco']
        ind = feature_v_mean.six_letter_name_Ronco.isin(diet_species)
        model, r_sq = run_linear_reg(feature_v_mean.loc[ind, 'total_rest'], feature_v_mean.loc[ind, 'fish_length_mm'])
        plt_lin_reg(rootdir, feature_v_mean.loc[ind, 'total_rest'], feature_v_mean.loc[ind, 'fish_length_mm'], model,
                    r_sq, diet)
