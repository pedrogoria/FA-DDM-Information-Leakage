"""Deterministic quadrature for Eve's posterior vulnerability."""

import numpy as np

from fa_ddm.vulnerability import mixture_centroids, weighted_symbol_likelihoods


def integration_bounds(centroids, noise_variance, noise_margin_sigma=6.0):
    """Return rectangular integration bounds containing all mixture centroids.

    Parameters
    ----------
    centroids : array_like
        Complex centroid array, normally with shape (M, N).
    noise_variance : float
        Complex Gaussian noise variance E[|Z|^2].
    noise_margin_sigma : float
        Margin in units of sqrt(noise_variance) on every side.

    Returns
    -------
    tuple
        ``(real_min, real_max, imag_min, imag_max)``.
    """
    centroid_array = np.asarray(centroids, dtype=np.complex128)
    if centroid_array.size == 0:
        raise ValueError("centroids must not be empty.")
    if noise_variance <= 0.0:
        raise ValueError("noise_variance must be positive.")
    if noise_margin_sigma <= 0.0:
        raise ValueError("noise_margin_sigma must be positive.")

    margin = float(noise_margin_sigma) * np.sqrt(float(noise_variance))
    return (
        float(np.min(centroid_array.real) - margin),
        float(np.max(centroid_array.real) + margin),
        float(np.min(centroid_array.imag) - margin),
        float(np.max(centroid_array.imag) + margin),
    )


def trapezoidal_axis(lower, upper, number_of_points):
    """Return one-dimensional nodes and trapezoidal integration weights."""
    if not isinstance(number_of_points, (int, np.integer)):
        raise TypeError("number_of_points must be an integer.")
    if number_of_points < 2:
        raise ValueError("number_of_points must be at least two.")
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("upper must be finite and greater than lower.")

    nodes = np.linspace(float(lower), float(upper), number_of_points)
    spacing = (float(upper) - float(lower)) / (number_of_points - 1)
    weights = np.full(number_of_points, spacing, dtype=float)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return nodes, weights


def rectangular_quadrature_grid(bounds, points_per_axis):
    """Construct complex grid nodes and product trapezoidal weights."""
    if len(bounds) != 4:
        raise ValueError("bounds must contain four values.")

    real_nodes, real_weights = trapezoidal_axis(
        bounds[0], bounds[1], points_per_axis
    )
    imag_nodes, imag_weights = trapezoidal_axis(
        bounds[2], bounds[3], points_per_axis
    )

    real_grid, imag_grid = np.meshgrid(real_nodes, imag_nodes, indexing="xy")
    weight_real, weight_imag = np.meshgrid(
        real_weights, imag_weights, indexing="xy"
    )
    observations = (real_grid + 1j * imag_grid).ravel()
    weights = (weight_real * weight_imag).ravel()
    return observations, weights


def quadrature_vulnerability(
    symbols,
    effective_eve_channel,
    port_probabilities,
    symbol_probabilities,
    transmit_power,
    noise_variance,
    points_per_axis=151,
    noise_margin_sigma=6.0,
    bounds=None,
    batch_size=20000,
):
    """Approximate posterior vulnerability by two-dimensional quadrature.

    The computation is batched to avoid storing all likelihood tensors at once.

    Returns
    -------
    float
        Deterministic approximation of Eve's optimal success probability.
    """
    if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    centroids = mixture_centroids(
        symbols, effective_eve_channel, transmit_power
    )
    if bounds is None:
        bounds = integration_bounds(
            centroids,
            noise_variance,
            noise_margin_sigma=noise_margin_sigma,
        )

    observations, weights = rectangular_quadrature_grid(
        bounds, points_per_axis
    )

    vulnerability = 0.0
    number_of_nodes = observations.size
    for start in range(0, number_of_nodes, batch_size):
        stop = min(start + batch_size, number_of_nodes)
        likelihoods = weighted_symbol_likelihoods(
            observations=observations[start:stop],
            symbols=symbols,
            effective_eve_channel=effective_eve_channel,
            port_probabilities=port_probabilities,
            symbol_probabilities=symbol_probabilities,
            transmit_power=transmit_power,
            noise_variance=noise_variance,
        )
        vulnerability += np.sum(
            weights[start:stop] * np.max(likelihoods, axis=1)
        )

    return float(vulnerability)
