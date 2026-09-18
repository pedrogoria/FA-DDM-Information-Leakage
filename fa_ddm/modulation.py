"""Digital modulation constellations used by the simulations."""

import numpy as np


def qpsk_constellation():
    """Return a unit-average-energy QPSK constellation."""

    symbols = np.array(
        [
            1.0 + 1.0j,
            -1.0 + 1.0j,
            -1.0 - 1.0j,
            1.0 - 1.0j,
        ],
        dtype=np.complex128,
    )

    return symbols / np.sqrt(2.0)


def uniform_symbol_probabilities(number_of_symbols):
    """Return uniform probabilities for a finite constellation."""

    if number_of_symbols <= 0:
        raise ValueError("number_of_symbols must be positive.")

    return np.full(
        number_of_symbols,
        1.0 / number_of_symbols,
        dtype=float,
    )