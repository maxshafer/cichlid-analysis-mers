import pytest

from ..utils.species_names import six_letter_sp_name, shorten_sp_name, check_file_name, get_roi_from_fish_id


@pytest.mark.parametrize("long_name, six_name", [("Neolamprologous pulcher", ["Neopul"]),
                                                 ("Neolamprologous-pulcher", ["Neopul"]),
                                                 ("Astotilapia burtoni chipwa", ["Astbur"]),
                                                 (["Neolamprologous pulcher", "Astotilapia burtoni chipwa"],
                                                  ["Neopul", "Astbur"]),
                                                 (["Neolamprologous pulcher", "Astotilapia 'burtoni' chipwa"],
                                                  ["Neopul", "Astbur"]),
                                                 (["Astotilapia 'burtoni' chipwa"], ["Astbur"])])
def test_six_letter_sp_name(long_name, six_name):
    assert six_letter_sp_name(long_name) == six_name


@pytest.mark.parametrize('long_name, short_name', [("Neolamprologous pulcher", ["N. pulcher"]),
                                                   ("Neolamprologous-pulcher", ["N. pulcher"]),
                                                   ("Astotilapia burtoni chipwa", ["A. burtoni"]),
                                                   (["Neolamprologous pulcher", "Astotilapia burtoni chipwa"],
                                                    ["N. pulcher", "A. burtoni"])])
def test_shorten_sp_name(long_name, short_name):
    assert shorten_sp_name(long_name) == short_name


def test_check_file_name():
    file_name = 'FISH20201022_c1_r0_Neolamprologus-brevis_sf'
    assert check_file_name(file_name) is None


def test_check_file_name_len():
    file_name = 'FISH202010_c1_r0_Neolamprologus-brevis_sf'
    with pytest.raises(ValueError, match="recording name is not the right length. It should be FISHYYYYMMDD"):
        check_file_name(file_name)


def test_check_file_name_camera():
    file_name = 'FISH20201022_test1_r0_Neolamprologus-brevis_sf'
    with pytest.raises(ValueError,
                       match="recording name is incorrectly formatted at the camera position, should be "
                             "FISHYYYYMMDD_c#_r#_species-names_sX"):
        check_file_name(file_name)


def test_check_file_name_camera():
    file_name = 'FISH20201022_c1_area0_Neolamprologus-brevis_sf'
    with pytest.raises(ValueError,
                       match="recording name is incorrectly formatted at the roi position, "
                             "should be FISHYYYYMMDD_c#_r#_species-names_sX"):
        check_file_name(file_name)


def test_check_file_name_camera():
    file_name = 'FISH20201022_c1_r0_Neolamprologus-brevis_testf'
    with pytest.raises(ValueError,
                       match="recording name is incorrectly formatted at the sex position, "
                             "should be FISHYYYYMMDD_c#_r#_species-names_sX"):
        check_file_name(file_name)


def test_check_file_name_sex():
    file_name = 'FISH20201022_c1_r0_Neolamprologus-brevis_testf'
    with pytest.raises(ValueError,
                       match="recording name is incorrectly formatted at the sex position, "
                             "should be FISHYYYYMMDD_c#_r#_species-names_sX"):
        check_file_name(file_name)


def test_check_file_name_len():
    file_name = 'FISH20201022_c1_r0_Neolamprologus-brevis'
    with pytest.raises(ValueError,
                       match="recording name is incorrectly formatted, "
                             "should be FISHYYYYMMDD_c#_r#_species-names_sX"):
        check_file_name(file_name)



@pytest.mark.parametrize("fish_id, roi_n",
                         [['FISH20201021_c1_r0_Neolamprologus-brevis_sf', '0'],
                          ['FISH20201022_c1_r1_Neolamprologus-brevis_sf', '1']])
def test_get_roi_from_name(fish_id, roi_n):
    roi_n_calc = get_roi_from_fish_id(fish_id)
    assert roi_n == roi_n_calc
