"""Tests for selectable modulation constellations."""

import numpy as np
import pytest

from fa_ddm.modulation import get_constellation, square_qam_constellation


@pytest.mark.parametrize(
    "name, expected_size",
    [("BPSK", 2), ("QPSK", 4), ("8PSK", 8), ("16PSK", 16),
     ("16QAM", 16), ("64QAM", 64)],
)
def test_supported_constellation_size_and_energy(name, expected_size):
    symbols = get_constellation(name)
    assert symbols.shape == (expected_size,)
    assert np.mean(np.abs(symbols) ** 2) == pytest.approx(1.0)


def test_modulation_name_is_case_insensitive():
    np.testing.assert_allclose(get_constellation("16-qam"), get_constellation("16QAM"))


def test_invalid_modulation_is_rejected():
    with pytest.raises(ValueError):
        get_constellation("unsupported")
    with pytest.raises(ValueError):
        square_qam_constellation(8)
