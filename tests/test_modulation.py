"""Tests for modulation constellations."""

import numpy as np
import pytest

from fa_ddm.modulation import (
    qpsk_constellation,
    uniform_symbol_probabilities,
)


def test_qpsk_has_four_symbols():
    symbols = qpsk_constellation()

    assert symbols.shape == (4,)


def test_qpsk_has_unit_average_energy():
    symbols = qpsk_constellation()

    average_energy = np.mean(np.abs(symbols) ** 2)

    assert average_energy == pytest.approx(1.0)


def test_qpsk_symbols_have_constant_modulus():
    symbols = qpsk_constellation()

    np.testing.assert_allclose(
        np.abs(symbols),
        np.ones(4),
    )


def test_uniform_symbol_probabilities():
    probabilities = uniform_symbol_probabilities(4)

    np.testing.assert_allclose(
        probabilities,
        np.full(4, 0.25),
    )

    assert probabilities.sum() == pytest.approx(1.0)


def test_uniform_symbol_probabilities_reject_invalid_size():
    with pytest.raises(ValueError):
        uniform_symbol_probabilities(0)