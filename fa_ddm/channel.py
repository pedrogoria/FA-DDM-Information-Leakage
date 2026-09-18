"""Fluid-antenna port geometry and spatial correlation models."""

import numpy as np
from scipy.special import j0


def linear_port_positions(number_of_ports, aperture_wavelengths):
    """Return uniformly spaced positions over a one-dimensional aperture.

    Parameters
    ----------
    number_of_ports : int
        Number of candidate fluid-antenna ports. Must be positive.
    aperture_wavelengths : float
        Total aperture length normalized by the carrier wavelength. Must be
        nonnegative.

    Returns
    -------
    numpy.ndarray
        One-dimensional array with shape ``(number_of_ports,)``. Each entry is
        a port position normalized by the wavelength.

    Notes
    -----
    The first port is located at zero and the last port is located at
    ``aperture_wavelengths``. For one port, the position is set to zero.
    """
    if not isinstance(number_of_ports, (int, np.integer)):
        raise TypeError("number_of_ports must be an integer.")
    if number_of_ports <= 0:
        raise ValueError("number_of_ports must be positive.")
    if aperture_wavelengths < 0.0:
        raise ValueError("aperture_wavelengths must be nonnegative.")

    if number_of_ports == 1:
        return np.array([0.0], dtype=float)

    return np.linspace(
        0.0,
        float(aperture_wavelengths),
        number_of_ports,
        dtype=float,
    )


def clarke_jakes_correlation(positions_wavelengths):
    """Construct the Clarke--Jakes spatial correlation matrix.

    Parameters
    ----------
    positions_wavelengths : array_like
        Port positions normalized by the carrier wavelength.

    Returns
    -------
    numpy.ndarray
        Real symmetric correlation matrix ``R`` with entries
        ``R[n, m] = J_0(2*pi*abs(x_n-x_m))``.

    Notes
    -----
    This model corresponds to two-dimensional isotropic diffuse scattering.
    It describes correlation among channel samples across the fluid aperture;
    it does not by itself describe Bob's or Eve's deterministic direction.
    """
    positions = np.asarray(positions_wavelengths, dtype=float)

    if positions.ndim != 1:
        raise ValueError("positions_wavelengths must be one-dimensional.")
    if positions.size == 0:
        raise ValueError("positions_wavelengths must not be empty.")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions_wavelengths must contain finite values.")

    pairwise_distance = np.abs(positions[:, None] - positions[None, :])
    correlation = j0(2.0 * np.pi * pairwise_distance)

    # Remove insignificant floating-point asymmetry and enforce exact unit
    # diagonal values expected from a normalized correlation matrix.
    correlation = 0.5 * (correlation + correlation.T)
    np.fill_diagonal(correlation, 1.0)

    return correlation


def minimum_eigenvalue(matrix):
    """Return the smallest eigenvalue of a real symmetric matrix."""
    values = np.linalg.eigvalsh(np.asarray(matrix, dtype=float))
    return float(values[0])
