import os

import numpy as np
import pandas as pd
import pytest

from cichlidanalysis.io.tracks import load_als_files


@pytest.mark.parametrize('test_data_file, expected', [
    ("FISH20210101_c1_r0_Example-genus_su_als.csv",
     pd.DataFrame(np.array([[8.64E+13, 3.069635729, 104, 670, 'FISH20210101_c1_r0_Example-genus_su',
                             pd.Timestamp('1970-01-02 00:00:00')]]), columns=['tv_ns', 'speed_mm', 'x_nt', 'y_nt',
                                            'FishID', 'ts']).astype({'tv_ns': 'float64', 'speed_mm': 'float64',
                            'x_nt': 'int64', 'y_nt': 'int64', 'FishID': 'object', 'ts': 'datetime64[ns]'}))])
def test_load_als_files(test_data_file, expected):
    test_data_dir = os.path.join(os.path.split(os.path.split(os.path.split(__file__)[0])[0])[0], 'example_data')
    fish_tracks = load_als_files(test_data_dir)
    assert fish_tracks.equals(expected)

