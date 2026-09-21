"""Fluid-antenna geometry, correlation, and Rician channel models."""

import numpy as np
from scipy.special import j0


def linear_port_positions(number_of_ports, aperture_wavelengths):
    """Return uniformly spaced 1-D port positions normalized by wavelength."""
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
    """Construct the Clarke--Jakes diffuse correlation matrix."""
    positions = np.asarray(positions_wavelengths, dtype=float)

    if positions.ndim != 1:
        raise ValueError("positions_wavelengths must be one-dimensional.")
    if positions.size == 0:
        raise ValueError("positions_wavelengths must not be empty.")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions_wavelengths must contain finite values.")

    pairwise_distance = np.abs(positions[:, None] - positions[None, :])
    correlation = j0(2.0 * np.pi * pairwise_distance)
    correlation = 0.5 * (correlation + correlation.T)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def minimum_eigenvalue(matrix):
    """Return the smallest eigenvalue of a real symmetric matrix."""
    values = np.linalg.eigvalsh(np.asarray(matrix, dtype=float))
    return float(values[0])


def far_field_signature(positions_wavelengths, angle_deg):
    """Return the deterministic far-field signature of a linear aperture.

    The angle is measured from broadside. For normalized position x_n/lambda,
    the n-th entry is exp(-j*2*pi*x_n*sin(theta)).
    """
    positions = np.asarray(positions_wavelengths, dtype=float)
    if positions.ndim != 1 or positions.size == 0:
        raise ValueError("positions_wavelengths must be a nonempty 1-D array.")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions_wavelengths must contain finite values.")
    if not np.isfinite(angle_deg):
        raise ValueError("angle_deg must be finite.")

    angle_rad = np.deg2rad(float(angle_deg))
    phase = -2.0 * np.pi * positions * np.sin(angle_rad)
    return np.exp(1j * phase).astype(np.complex128)


def hermitian_square_root(matrix, tolerance=1e-12):
    """Return a stable Hermitian positive-semidefinite matrix square root."""
    matrix = np.asarray(matrix, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square.")

    hermitian = 0.5 * (matrix + matrix.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    if eigenvalues[0] < -tolerance:
        raise ValueError("matrix must be positive semidefinite.")

    clipped = np.clip(eigenvalues, 0.0, None)
    return (eigenvectors * np.sqrt(clipped)) @ eigenvectors.conj().T


def generate_correlated_rician_channel(
    positions_wavelengths,
    angle_deg,
    large_scale_gain,
    rician_factor,
    rng,
    correlation_matrix=None,
):
    """Generate one spatially correlated Rician channel realization.

    Parameters use the paper normalization E[|h_n|^2] = large_scale_gain.
    The supplied ``rng`` must be a NumPy random Generator.
    """
    positions = np.asarray(positions_wavelengths, dtype=float)
    if large_scale_gain <= 0.0:
        raise ValueError("large_scale_gain must be positive.")
    if rician_factor < 0.0:
        raise ValueError("rician_factor must be nonnegative.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an instance of numpy.random.Generator.")

    if correlation_matrix is None:
        correlation = clarke_jakes_correlation(positions)
    else:
        correlation = np.asarray(correlation_matrix, dtype=np.complex128)

    number_of_ports = positions.size
    if correlation.shape != (number_of_ports, number_of_ports):
        raise ValueError("correlation_matrix has an incompatible shape.")

    signature = far_field_signature(positions, angle_deg)
    root = hermitian_square_root(correlation)
    white = (
        rng.standard_normal(number_of_ports)
        + 1j * rng.standard_normal(number_of_ports)
    ) / np.sqrt(2.0)
    diffuse = root @ white

    los_scale = np.sqrt(rician_factor / (rician_factor + 1.0))
    diffuse_scale = np.sqrt(1.0 / (rician_factor + 1.0))
    channel = np.sqrt(large_scale_gain) * (
        los_scale * signature + diffuse_scale * diffuse
    )
    return channel.astype(np.complex128)


def phase_precoder(bob_channel, zero_tolerance=1e-15):
    """Return q_n = conj(h_B,n)/|h_B,n| for every candidate port."""
    bob_channel = np.asarray(bob_channel, dtype=np.complex128)
    if bob_channel.ndim != 1 or bob_channel.size == 0:
        raise ValueError("bob_channel must be a nonempty 1-D array.")
    magnitudes = np.abs(bob_channel)
    if np.any(magnitudes <= zero_tolerance):
        raise ValueError("bob_channel contains a zero-magnitude coefficient.")
    return np.conj(bob_channel) / magnitudes


def effective_eve_channel(bob_channel, eve_channel):
    """Return g_n = h_E,n * conj(h_B,n)/|h_B,n|."""
    bob_channel = np.asarray(bob_channel, dtype=np.complex128)
    eve_channel = np.asarray(eve_channel, dtype=np.complex128)
    if bob_channel.shape != eve_channel.shape:
        raise ValueError("bob_channel and eve_channel must have equal shapes.")
    return eve_channel * phase_precoder(bob_channel)
