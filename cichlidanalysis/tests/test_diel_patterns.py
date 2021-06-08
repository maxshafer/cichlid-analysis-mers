import pytest
import numpy as np

from cichlidanalysis.analysis.diel_pattern import replace_crep_peaks


# @pytest.mark.parametrize("fish_peaks, fish_feature, fish_num", [
#     (np.array([15., 0., 15.],[15., 0., 111],[0.62, 0, 0.63],[0.20, -0.42, 0.21]),  0 , 0)])
#
# def test_replace_crep_peaks(fish_peaks, fish_feature, fish_num):
#     fish_peaks = replace_crep_peaks(fish_peaks, fish_feature, fish_num)
#     assert
#
#
#
#     # def replace_crep_peaks(fish_peaks, fish_feature, fish_num):
#     #     """for crepuscular peaks, this finds if the first row (intra day 30min bin) is == 0, which means it didn't have a
#     #     peak and replaces it's peak height with the value of the mode of the first row. It then corrects the peak amplitude.
#     #
#     #     :param fish_peaks:
#     #     :param fish_feature:
#     #     :param fish_num:
#     #     :return: fish_peaks
#     #     """
#     #     # check if any of the peaks need replacing
#     #     if (fish_peaks[0, :] == 0).any():
#     #         common_peak = stats.mode(fish_peaks[0, :])[0][0]
#     #         no_peaks = np.where(fish_peaks[0, :] == 0)[0]
#     #         for no_peak in no_peaks:
#     #             # add in peak height
#     #             fish_peaks[2, no_peak] = fish_feature.iloc[int(epoques[no_peak] + common_peak), fish_num]
#     #             # correct peak amplitude
#     #             fish_peaks[3, no_peak] = fish_peaks[3, no_peak] + fish_peaks[2, no_peak]
#     #     return fish_peaks
