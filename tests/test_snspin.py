"""Test snspin."""

from snspin import main


SNFIDR = "snspin_testdata/snf_idr/"


def test_spincalc(idr=SNFIDR):
    main.spincalc(["--idr", idr,
                   "--target", "PTF09dnl",
                   "--expid", idr + "spectra_test.txt"])

