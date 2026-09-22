"""Digital modulation constellations used by the simulations."""

import numpy as np


def _normalize_average_energy(symbols):
    """Normalize a complex constellation to unit average energy."""
    symbols = np.asarray(symbols, dtype=np.complex128)
    average_energy = np.mean(np.abs(symbols) ** 2)
    if average_energy <= 0.0:
        raise ValueError("Constellation average energy must be positive.")
    return symbols / np.sqrt(average_energy)


def bpsk_constellation():
    """Return the unit-average-energy BPSK constellation."""
    return np.array([1.0, -1.0], dtype=np.complex128)


def qpsk_constellation():
    """Return a unit-average-energy QPSK constellation."""
    symbols = np.array(
        [1.0 + 1.0j, -1.0 + 1.0j, -1.0 - 1.0j, 1.0 - 1.0j],
        dtype=np.complex128,
    )
    return _normalize_average_energy(symbols)


def psk_constellation(order):
    """Return a unit-average-energy M-PSK constellation."""
    if not isinstance(order, (int, np.integer)):
        raise TypeError("PSK order must be an integer.")
    if order < 2:
        raise ValueError("PSK order must be at least two.")
    phases = 2.0 * np.pi * np.arange(order) / order
    return np.exp(1j * phases).astype(np.complex128)


def square_qam_constellation(order):
    """Return a unit-average-energy square M-QAM constellation."""
    if not isinstance(order, (int, np.integer)):
        raise TypeError("QAM order must be an integer.")
    side = int(np.sqrt(order))
    if side * side != order or side < 2:
        raise ValueError("QAM order must be a square integer of at least four.")
    levels = np.arange(-(side - 1), side, 2, dtype=float)
    real_grid, imag_grid = np.meshgrid(levels, levels, indexing="xy")
    symbols = (real_grid + 1j * imag_grid).ravel()
    return _normalize_average_energy(symbols)


def get_constellation(name):
    """Return a supported unit-average-energy constellation by name.

    Supported names are BPSK, QPSK, 8PSK, 16PSK, 16QAM, and 64QAM.
    Names are case-insensitive and may contain hyphens or spaces.
    """
    normalized = str(name).upper().replace("-", "").replace(" ", "")
    if normalized == "BPSK":
        return bpsk_constellation()
    if normalized == "QPSK":
        return qpsk_constellation()
    if normalized.endswith("PSK") and normalized[:-3].isdigit():
        return psk_constellation(int(normalized[:-3]))
    if normalized.endswith("QAM") and normalized[:-3].isdigit():
        return square_qam_constellation(int(normalized[:-3]))
    raise ValueError("Unsupported modulation: {0}".format(name))


def uniform_symbol_probabilities(number_of_symbols):
    """Return a uniform probability vector for a finite constellation."""
    if number_of_symbols <= 0:
        raise ValueError("number_of_symbols must be positive.")
    return np.full(number_of_symbols, 1.0 / number_of_symbols, dtype=float)
