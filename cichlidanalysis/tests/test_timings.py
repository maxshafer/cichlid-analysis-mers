import pytest

from cichlidanalysis.utils.timings import get_start_time_from_str


@pytest.mark.parametrize("start_time, start_total_sec_expected",
                         [["115021", 42621],
                          ["000000", 0],
                          ["000060", 60],
                          ["000100", 60],
                          ["010000", 3600]])
def test_get_start_time_from_str(start_time, start_total_sec_expected):
    start_total_sec = get_start_time_from_str(start_time)
    assert start_total_sec_expected == start_total_sec
