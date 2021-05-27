import pytest

from cichlidanalysis.utils.species_names import six_letter_sp_name, shorten_sp_name

@pytest.mark.parametrize("long_name, six_name", [("Neolamprologous pulcher", ["Neopul"]),
                                       ("Neolamprologous-pulcher", ["Neopul"]),
                                       ("Astotilapia burtoni chipwa", ["Astbur"]),
                                        (["Neolamprologous pulcher", "Astotilapia burtoni chipwa"], ["Neopul", "Astbur"])])
def test_six_letter_sp_name(long_name, six_name):
    assert six_letter_sp_name(long_name) == six_name


@pytest.mark.parametrize("long_name, short_name", [("Neolamprologous pulcher", ["N. pulcher"]),
                                       ("Neolamprologous-pulcher", ["N. pulcher"]),
                                       ("Astotilapia burtoni chipwa", ["A. burtoni"]),
                         (["Neolamprologous pulcher", "Astotilapia burtoni chipwa"], ["N. pulcher", "A. burtoni"])])
def test_shorten_sp_name(long_name, short_name):
    assert shorten_sp_name(long_name) == short_name
