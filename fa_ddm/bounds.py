"""
Finite-SNR vulnerability bounds for fluid-antenna-assisted dynamic
directional modulation.

Author
------
Pedro E. Goria Silva

Module
------
fa_ddm.bounds

Purpose
-------
This module implements numerical tools for evaluating analytical bounds on
Eve's posterior Bayes vulnerability in fluid-antenna-assisted dynamic
directional modulation (FA-DDM).

In the considered system, Alice activates one fluid-antenna port per channel
use and applies a port-dependent phase coefficient that compensates Bob's
instantaneous channel phase. Because Eve does not know the active-port
realization before making a symbol decision, each confidential symbol induces
a Gaussian-mixture observation at Eve.

The functions in this module quantify the distinguishability of these
secret-conditioned Gaussian mixtures and evaluate:

1. the exact finite-grid posterior vulnerability by deterministic quadrature;
2. pairwise total-variation distances relative to a reference symbol;
3. the finite-SNR total-variation lower bound;
4. the finite-SNR total-variation upper bound;
5. the permutation-based Gaussian-component overlap certificate;
6. the corresponding permutation-based vulnerability upper bound.

The implementation is intended to support the analytical and numerical results
of the FA-DDM information-leakage study and to generate reproducible data for
the associated manuscript figures.

Mathematical setting
--------------------
Let S be a uniformly distributed confidential symbol taking values in an
M-ary constellation. For each symbol s_m, Eve observes a conditional density

    p(y | s_m)
        = sum_n rho_n CN(
              y;
              sqrt(P_t) g_n s_m,
              sigma_E^2
          ),

where:

    rho_n       is the activation probability of port n;
    g_n         is Eve's effective channel coefficient for port n;
    P_t         is the transmit power;
    sigma_E^2   is Eve's complex-noise variance.

For a uniform secret, Eve's posterior vulnerability is

    V_E(rho)
        = (1 / M) integral_C max_m p(y | s_m) dy.

For a fixed reference symbol s_j, the pairwise total-variation distance is

    delta_{m,j}(rho)
        = (1 / 2) integral_C
          |p(y | s_m) - p(y | s_j)| dy.

The finite-SNR bounds evaluated by this module are

    (1 / M) [1 + max_{m != j} delta_{m,j}]
        <= V_E(rho)
        <=
    (1 / M) [1 + sum_{m != j} delta_{m,j}].

For binary signaling, the two expressions coincide and provide the exact
posterior vulnerability.

Permutation-based overlap certificate
-------------------------------------
For each pair of symbols (s_m, s_j), the module also evaluates a
component-matching certificate of the form

    Gamma_{m,j}(rho)
        = max_tau sum_n sqrt(rho_n rho_{tau(n)})
          exp(
              -P_t |g_n s_m - g_{tau(n)} s_j|^2
              / (4 sigma_E^2)
          ).

The maximizing permutation is obtained as a linear assignment problem using
``scipy.optimize.linear_sum_assignment``. This avoids explicit enumeration of
all N! port permutations.

The resulting vulnerability upper bound is

    V_E(rho)
        <= (1 / M) [
               1
               + sum_{m != j}
                 sqrt(1 - Gamma_{m,j}(rho)^2)
           ].

Numerical methodology
---------------------
The exact vulnerability and total-variation distances are evaluated on the
same two-dimensional trapezoidal quadrature grid over Eve's complex
observation plane. Reusing the same grid ensures that differences between the
exact value and the total-variation bounds are not caused by inconsistent
integration domains or quadrature resolutions.

The integration region is selected to contain all Gaussian-mixture centroids
with a configurable noise-dependent margin. Likelihood evaluation is performed
in batches to limit peak memory usage.

The permutation-based bound does not require numerical integration of density
differences. It depends directly on the port probabilities, conditional
centroids, noise variance, and optimal one-to-one component matching.

Assumptions
-----------
The implemented bounds assume that:

- the confidential symbol has a uniform prior distribution;
- the symbol alphabet contains at least two symbols;
- the port-selection probabilities form a valid probability vector;
- the noise variance is strictly positive;
- all Gaussian components have the same complex-noise variance;
- Eve knows the effective channel coefficients and port-selection law;
- the reference-symbol index follows zero-based Python indexing;
- posterior vulnerability is defined for optimal one-guess identity recovery.

The upper bounds may exceed one before post-processing because the bounds are
analytical inequalities rather than probability-normalized approximations.
For plotting purposes, this module also returns capped versions computed as

    min(1, analytical_upper_bound).

The uncapped values are retained so that the analytical expressions can be
inspected without modification.

Public functions
----------------
pairwise_total_variation_bounds
    Evaluates the exact quadrature vulnerability, pairwise total-variation
    distances, and the corresponding finite-SNR lower and upper bounds.

permutation_overlap_certificate
    Computes the optimized component-matching certificate Gamma_{m,j} for one
    symbol pair by solving a linear assignment problem.

permutation_vulnerability_bound
    Aggregates the pairwise overlap certificates into the permutation-based
    vulnerability upper bound for a selected reference symbol.

Dependencies
------------
numpy
    Array manipulation, validation, quadrature accumulation, and numerical
    operations.

scipy.optimize.linear_sum_assignment
    Polynomial-time solution of the maximum-weight component-matching problem,
    implemented by minimizing the negative overlap matrix.

fa_ddm.quadrature
    Construction of the integration region and rectangular quadrature grid.

fa_ddm.vulnerability
    Mixture-centroid construction, probability-vector validation, and weighted
    symbol-likelihood evaluation.

Returned quantities
-------------------
The finite-SNR routine returns a dictionary containing:

    vulnerability
        Exact posterior vulnerability evaluated on the quadrature grid.

    total_variation
        Vector of total-variation distances between every symbol-conditioned
        density and the selected reference-symbol density.

    lower_bound
        Finite-SNR total-variation lower bound.

    upper_bound
        Raw finite-SNR total-variation upper bound.

    upper_bound_capped
        Upper bound limited to the probability range [0, 1].

    bounds
        Rectangular integration limits used by the quadrature calculation.

The permutation-based routine returns:

    gamma
        Vector of optimized component-overlap certificates relative to the
        selected reference symbol.

    upper_bound
        Raw permutation-based vulnerability upper bound.

    upper_bound_capped
        Permutation-based upper bound limited to [0, 1].

Reproducibility
---------------
This module contains no random-number generation. Its outputs are deterministic
for fixed symbols, channels, probabilities, transmit power, noise variance,
quadrature settings, and reference-symbol index.

The channel realization, random seed, SNR sweep, and numerical resolution are
defined by the corresponding experiment runner and YAML configuration.

Notes
-----
This module provides numerical evaluations of the analytical expressions. It
does not replace the exact Gaussian-mixture likelihood model with a surrogate
detector, a nearest-neighbor approximation, or a conventional symbol-error
metric.

When the quadrature resolution or integration margin is changed, convergence
of the exact vulnerability and total-variation distances should be verified
before the generated values are used in final manuscript figures.

See Also
--------
fa_ddm.vulnerability
    Gaussian-mixture likelihoods and Monte Carlo vulnerability estimation.

fa_ddm.quadrature
    Deterministic integration over Eve's complex observation plane.

fa_ddm.optimization
    Privacy-aware optimization of the fluid-antenna port-selection
    probabilities.

scripts.run_figure_07
    Reproducible experiment runner for the vulnerability-bound validation
    figure.
"""

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
