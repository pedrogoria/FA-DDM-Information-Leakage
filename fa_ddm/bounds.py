"""Finite-SNR vulnerability bounds for FA-assisted DDM."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from fa_ddm.quadrature import (
    integration_bounds,
    rectangular_quadrature_grid,
)
from fa_ddm.vulnerability import (
    mixture_centroids,
    validate_probability_vector,
    weighted_symbol_likelihoods,
)


def pairwise_total_variation_bounds(
    symbols,
    effective_eve_channel,
    port_probabilities,
    symbol_probabilities,
    transmit_power,
    noise_variance,
    reference_symbol_index=0,
    points_per_axis=151,
    noise_margin_sigma=6.0,
    bounds=None,
    batch_size=20000,
):
    """Evaluate vulnerability and finite-SNR TV bounds on one quadrature grid.

    The bounds correspond to Theorem 1 for one fixed reference symbol.

    Returns
    -------
    dict
        Exact quadrature vulnerability, pairwise total-variation distances,
        and the corresponding lower and upper bounds.
    """
    symbols = np.asarray(symbols, dtype=np.complex128)
    effective_channel = np.asarray(
        effective_eve_channel,
        dtype=np.complex128,
    )
    number_of_symbols = symbols.size
    if number_of_symbols < 2:
        raise ValueError("At least two symbols are required.")
    if not 0 <= int(reference_symbol_index) < number_of_symbols:
        raise ValueError("reference_symbol_index is out of range.")
    if noise_variance <= 0.0:
        raise ValueError("noise_variance must be positive.")
    if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    rho = validate_probability_vector(
        port_probabilities,
        effective_channel.size,
        "port_probabilities",
    )
    priors = validate_probability_vector(
        symbol_probabilities,
        number_of_symbols,
        "symbol_probabilities",
    )
    if np.any(priors <= 0.0):
        raise ValueError("symbol_probabilities must be strictly positive.")
    if not np.allclose(
        priors,
        np.full(number_of_symbols, 1.0 / number_of_symbols),
        atol=1e-12,
    ):
        raise ValueError("The finite-SNR bounds require a uniform secret.")

    centroids = mixture_centroids(
        symbols,
        effective_channel,
        transmit_power,
    )
    if bounds is None:
        bounds = integration_bounds(
            centroids,
            noise_variance,
            noise_margin_sigma=noise_margin_sigma,
        )
    observations, weights = rectangular_quadrature_grid(
        bounds,
        points_per_axis,
    )

    reference_symbol_index = int(reference_symbol_index)
    total_variation = np.zeros(number_of_symbols, dtype=float)
    vulnerability = 0.0

    for start in range(0, observations.size, batch_size):
        stop = min(start + batch_size, observations.size)
        likelihoods = weighted_symbol_likelihoods(
            observations=observations[start:stop],
            symbols=symbols,
            effective_eve_channel=effective_channel,
            port_probabilities=rho,
            symbol_probabilities=priors,
            transmit_power=transmit_power,
            noise_variance=noise_variance,
        )
        batch_weights = weights[start:stop]
        vulnerability += np.sum(
            batch_weights * np.max(likelihoods, axis=1)
        )

        conditional_densities = likelihoods / priors[None, :]
        reference_density = conditional_densities[:, reference_symbol_index]
        total_variation += 0.5 * np.sum(
            batch_weights[:, None]
            * np.abs(conditional_densities - reference_density[:, None]),
            axis=0,
        )

    competing = np.arange(number_of_symbols) != reference_symbol_index
    lower_bound = (
        1.0 + np.max(total_variation[competing])
    ) / number_of_symbols
    upper_bound = (
        1.0 + np.sum(total_variation[competing])
    ) / number_of_symbols

    return {
        "vulnerability": float(vulnerability),
        "total_variation": total_variation,
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "upper_bound_capped": float(min(1.0, upper_bound)),
        "bounds": tuple(float(value) for value in bounds),
    }


def permutation_overlap_certificate(
    symbol_index,
    reference_symbol_index,
    symbols,
    effective_eve_channel,
    port_probabilities,
    transmit_power,
    noise_variance,
):
    """Compute the optimized component-matching certificate Gamma_{m,j}."""
    symbols = np.asarray(symbols, dtype=np.complex128)
    effective_channel = np.asarray(
        effective_eve_channel,
        dtype=np.complex128,
    )
    if noise_variance <= 0.0:
        raise ValueError("noise_variance must be positive.")
    if not 0 <= int(symbol_index) < symbols.size:
        raise ValueError("symbol_index is out of range.")
    if not 0 <= int(reference_symbol_index) < symbols.size:
        raise ValueError("reference_symbol_index is out of range.")

    rho = validate_probability_vector(
        port_probabilities,
        effective_channel.size,
        "port_probabilities",
    )
    centroids = mixture_centroids(
        symbols,
        effective_channel,
        transmit_power,
    )

    source = centroids[int(symbol_index), :, None]
    target = centroids[int(reference_symbol_index), None, :]
    probability_factor = np.sqrt(rho[:, None] * rho[None, :])
    overlap = probability_factor * np.exp(
        -(np.abs(source - target) ** 2) / (4.0 * noise_variance)
    )

    rows, columns = linear_sum_assignment(-overlap)
    gamma = float(np.sum(overlap[rows, columns]))
    return float(np.clip(gamma, 0.0, 1.0))


def permutation_vulnerability_bound(
    symbols,
    effective_eve_channel,
    port_probabilities,
    symbol_probabilities,
    transmit_power,
    noise_variance,
    reference_symbol_index=0,
):
    """Evaluate the permutation-based vulnerability upper bound."""
    symbols = np.asarray(symbols, dtype=np.complex128)
    priors = validate_probability_vector(
        symbol_probabilities,
        symbols.size,
        "symbol_probabilities",
    )
    if not np.allclose(
        priors,
        np.full(symbols.size, 1.0 / symbols.size),
        atol=1e-12,
    ):
        raise ValueError("The permutation bound requires a uniform secret.")

    reference_symbol_index = int(reference_symbol_index)
    gammas = np.ones(symbols.size, dtype=float)
    terms = []
    for symbol_index in range(symbols.size):
        if symbol_index == reference_symbol_index:
            continue
        gamma = permutation_overlap_certificate(
            symbol_index=symbol_index,
            reference_symbol_index=reference_symbol_index,
            symbols=symbols,
            effective_eve_channel=effective_eve_channel,
            port_probabilities=port_probabilities,
            transmit_power=transmit_power,
            noise_variance=noise_variance,
        )
        gammas[symbol_index] = gamma
        terms.append(np.sqrt(max(0.0, 1.0 - gamma ** 2)))

    bound = (1.0 + np.sum(terms)) / symbols.size
    return {
        "gamma": gammas,
        "upper_bound": float(bound),
        "upper_bound_capped": float(min(1.0, bound)),
    }
