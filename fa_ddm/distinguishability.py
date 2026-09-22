"""Finite-SNR distinguishability measures and vulnerability bounds."""

import numpy as np

from fa_ddm.quadrature import (
    integration_bounds,
    rectangular_quadrature_grid,
)
from fa_ddm.vulnerability import (
    mixture_centroids,
    weighted_symbol_likelihoods,
)


def conditional_symbol_densities(
    observations,
    symbols,
    effective_eve_channel,
    port_probabilities,
    transmit_power,
    noise_variance,
):
    """Evaluate p(y|s_m) for every observation and symbol."""
    symbols = np.asarray(symbols, dtype=np.complex128)
    uniform_priors = np.full(symbols.size, 1.0 / symbols.size)
    weighted = weighted_symbol_likelihoods(
        observations=observations,
        symbols=symbols,
        effective_eve_channel=effective_eve_channel,
        port_probabilities=port_probabilities,
        symbol_probabilities=uniform_priors,
        transmit_power=transmit_power,
        noise_variance=noise_variance,
    )
    return weighted * symbols.size


def finite_snr_metrics(
    symbols,
    effective_eve_channel,
    port_probabilities,
    transmit_power,
    noise_variance,
    reference_symbol_index=0,
    points_per_axis=151,
    noise_margin_sigma=6.0,
    batch_size=20000,
):
    """Compute exact vulnerability, pairwise TV distances, and bounds.

    A uniform symbol prior is assumed, as required by the stated bounds.
    """
    symbols = np.asarray(symbols, dtype=np.complex128)
    number_of_symbols = symbols.size
    if number_of_symbols < 2:
        raise ValueError("At least two symbols are required.")
    if reference_symbol_index < 0 or reference_symbol_index >= number_of_symbols:
        raise ValueError("reference_symbol_index is invalid.")

    centroids = mixture_centroids(
        symbols, effective_eve_channel, transmit_power
    )
    bounds = integration_bounds(
        centroids,
        noise_variance,
        noise_margin_sigma=noise_margin_sigma,
    )
    observations, weights = rectangular_quadrature_grid(
        bounds, points_per_axis
    )

    vulnerability = 0.0
    tv_integrals = np.zeros(number_of_symbols, dtype=float)

    for start in range(0, observations.size, batch_size):
        stop = min(start + batch_size, observations.size)
        densities = conditional_symbol_densities(
            observations[start:stop],
            symbols,
            effective_eve_channel,
            port_probabilities,
            transmit_power,
            noise_variance,
        )
        local_weights = weights[start:stop]
        vulnerability += np.sum(
            local_weights * np.max(densities, axis=1)
        ) / number_of_symbols

        reference_density = densities[:, reference_symbol_index]
        for symbol_index in range(number_of_symbols):
            if symbol_index == reference_symbol_index:
                continue
            tv_integrals[symbol_index] += 0.5 * np.sum(
                local_weights
                * np.abs(densities[:, symbol_index] - reference_density)
            )

    competitors = np.delete(tv_integrals, reference_symbol_index)
    lower_bound = (1.0 + np.max(competitors)) / number_of_symbols
    upper_bound = (1.0 + np.sum(competitors)) / number_of_symbols
    upper_bound = min(1.0, float(upper_bound))

    result = {
        "vulnerability": float(vulnerability),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "pairwise_total_variation": tv_integrals,
    }
    if number_of_symbols == 2:
        result["binary_tv_identity"] = float(
            0.5 * (1.0 + competitors[0])
        )
    return result
