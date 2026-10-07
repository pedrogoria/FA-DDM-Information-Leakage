"""
Finite-SNR distinguishability measures and vulnerability bounds for
fluid-antenna-assisted dynamic directional modulation.

Author
------
Pedro E. Goria Silva

Module
------
fa_ddm.distinguishability

Purpose
-------
This module implements deterministic finite-signal-to-noise-ratio (finite-SNR)
distinguishability measures for the secret-conditioned Gaussian-mixture
observations produced by fluid-antenna-assisted dynamic directional modulation
(FA-DDM).

The module evaluates the complete conditional observation densities available
to an optimal eavesdropper and computes:

1. Eve's exact posterior Bayes vulnerability by two-dimensional quadrature;
2. pairwise total-variation distances between secret-conditioned densities;
3. a finite-SNR lower bound on posterior vulnerability;
4. a finite-SNR upper bound on posterior vulnerability; and
5. the exact binary total-variation identity when the confidential alphabet
   contains two symbols.

The implementation is intended for analytical validation, regression testing,
and reproducible generation of manuscript data. It evaluates the bounds from
Eve's full Gaussian-mixture likelihoods and does not substitute a conventional
nearest-neighbor demodulator or a symbol-error-rate surrogate.

System model
------------
Alice activates one fluid-antenna port during each channel use and applies a
port-dependent phase coefficient that compensates Bob's channel phase. Eve's
effective channel coefficient at port n is denoted by g_n. If the active port
is not disclosed to Eve, then confidential symbol s_m induces the conditional
density

    p(y | s_m)
        = sum_n rho_n CN(
              y;
              sqrt(P_t) g_n s_m,
              sigma_E^2
          ),

where:

    rho_n       is the activation probability of port n;
    P_t         is the transmit power;
    sigma_E^2   is Eve's complex Gaussian noise variance;
    CN(.)       denotes a circularly symmetric complex Gaussian density.

The port index is marginalized inside every secret-conditioned density. Eve is
assumed to know the constellation, channel coefficients, port probabilities,
transmit power, and noise variance and to use the optimal one-guess decision
rule.

Conditional densities
---------------------
The function ``conditional_symbol_densities`` evaluates p(y | s_m) for every
observation and every confidential symbol.

The underlying likelihood routine returns prior-weighted likelihoods

    ell_m(y) = p_S(s_m) p(y | s_m).

This module calls that routine with a uniform prior p_S(s_m) = 1/M and then
multiplies the result by M to recover the unweighted conditional densities.
Consequently, this helper is specifically aligned with the uniform-secret
assumption used by the finite-SNR vulnerability bounds.

Posterior vulnerability
-----------------------
For a uniformly distributed M-ary secret, Eve's posterior vulnerability is

    V_E(rho)
        = (1 / M) integral_C max_m p(y | s_m) dy.

The function ``finite_snr_metrics`` approximates this integral by a rectangular
two-dimensional trapezoidal quadrature over Eve's complex observation plane.
The integration region contains all Gaussian-mixture centroids and includes a
configurable noise-dependent margin.

Pairwise total variation
------------------------
For a selected reference symbol s_j, the pairwise total-variation distance is

    delta_{m,j}(rho)
        = (1 / 2) integral_C
          |p(y | s_m) - p(y | s_j)| dy.

The implementation evaluates all distances relative to the selected reference
symbol on the same quadrature grid used for the exact vulnerability. The entry
associated with the reference symbol remains zero.

Using one common grid ensures that comparisons between the exact vulnerability
and the analytical bounds are not affected by inconsistent integration limits,
quadrature resolutions, or numerical weighting conventions.

Finite-SNR bounds
-----------------
For a uniform M-ary confidential symbol and a fixed reference symbol s_j, the
implemented bounds are

    (1 / M) [1 + max_{m != j} delta_{m,j}(rho)]
        <= V_E(rho)
        <=
    (1 / M) [1 + sum_{m != j} delta_{m,j}(rho)].

The lower bound uses the most distinguishable competing symbol relative to the
reference. The upper bound aggregates the distinguishability of every
competing symbol relative to the same reference.

Because posterior vulnerability is a probability, the implementation limits
the reported upper bound to one:

    upper_bound = min(1, analytical_upper_bound).

The current function returns the capped value. If inspection of the raw
analytical expression is required, the implementation should be extended to
return both the raw and capped quantities explicitly.

Binary identity
---------------
When the alphabet contains two symbols, only one competitor exists relative to
the reference symbol. The lower and upper bounds therefore coincide, yielding

    V_E(rho) = (1 / 2) [1 + delta_{1,2}(rho)].

For binary constellations, ``finite_snr_metrics`` returns this value under the
key ``binary_tv_identity``. The quantity provides a direct regression check for
the vulnerability and total-variation implementations.

Numerical methodology
---------------------
The integration procedure consists of the following steps:

1. construct all port- and symbol-dependent Gaussian-mixture centroids;
2. define a rectangular region covering the centroids with a configurable
   noise margin;
3. construct complex quadrature nodes and product trapezoidal weights;
4. process the grid in batches to control peak memory usage;
5. evaluate every secret-conditioned density on each batch;
6. accumulate the exact posterior vulnerability;
7. accumulate the pairwise total-variation integrals; and
8. form the finite-SNR lower and upper bounds.

Batching changes memory usage but does not change the mathematical result. For
fixed inputs, grid resolution, integration margin, and batch size, the module
is deterministic.

Public functions
----------------
conditional_symbol_densities
    Evaluates the unweighted secret-conditioned Gaussian-mixture densities
    p(y | s_m) for one or more complex observations under a uniform symbol
    prior.

finite_snr_metrics
    Computes exact quadrature vulnerability, pairwise total-variation
    distances relative to a reference symbol, finite-SNR lower and upper
    bounds, and the binary identity when applicable.

Input conventions
-----------------
symbols
    One-dimensional complex array containing at least two constellation
    symbols.

effective_eve_channel
    One-dimensional complex array containing Eve's effective coefficient for
    every candidate port.

port_probabilities
    One-dimensional probability vector aligned with the effective channel.
    Entries must be nonnegative and sum to one. Validation is performed by the
    underlying likelihood routine.

transmit_power
    Strictly positive average transmit power used to construct the mixture
    centroids.

noise_variance
    Strictly positive complex Gaussian noise variance E[|Z|^2].

reference_symbol_index
    Zero-based index of the symbol used as the reference density in the
    pairwise total-variation calculations.

points_per_axis
    Number of trapezoidal nodes along each real dimension of the complex
    observation plane. The total grid contains points_per_axis squared nodes.

noise_margin_sigma
    Margin added around the centroid set, expressed in units of the complex
    noise standard deviation used by the integration-bound routine.

batch_size
    Maximum number of quadrature observations evaluated in one likelihood
    batch.

Returned quantities
-------------------
``finite_snr_metrics`` returns a dictionary containing:

    vulnerability
        Exact posterior vulnerability approximated on the quadrature grid.

    lower_bound
        Finite-SNR total-variation lower bound for the selected reference
        symbol.

    upper_bound
        Finite-SNR total-variation upper bound, capped at one.

    pairwise_total_variation
        Vector of total-variation distances between each conditional density
        and the reference-symbol density. The reference entry is zero.

    binary_tv_identity
        Returned only for binary alphabets. Equals one half times one plus the
        unique pairwise total-variation distance.

Assumptions
-----------
The implementation assumes that:

- the confidential symbol is uniformly distributed;
- the symbol alphabet contains at least two elements;
- the active port is not disclosed to Eve before the symbol decision;
- all Gaussian-mixture components share the same complex noise variance;
- Eve uses the complete secret-conditioned Gaussian-mixture likelihoods;
- the effective channel and port-selection probabilities are known;
- the reference-symbol index uses zero-based Python indexing; and
- posterior vulnerability represents optimal one-guess identity recovery.

Validation and numerical safeguards
-----------------------------------
The module validates the symbol count and reference-symbol index directly.
Additional checks for probability vectors, channel dimensions, transmit power,
and noise variance are performed by the imported centroid and likelihood
utilities.

The accuracy of the returned metrics depends on the integration margin and
quadrature resolution. Final manuscript data should be checked for convergence
by enlarging the integration region and increasing ``points_per_axis`` until
the reported values change by less than the desired tolerance.

The upper bound is capped at one for probability consistency. Users requiring
the uncapped analytical expression should retain it separately before applying
the probability cap.

Computational complexity
------------------------
Let R be the total number of quadrature nodes, M the number of confidential
symbols, and N the number of candidate ports. Likelihood evaluation requires
O(R M N) Gaussian-kernel operations. Pairwise total-variation accumulation
relative to one reference symbol requires O(R M) additional arithmetic.

Peak memory is controlled by ``batch_size`` because the complete
observation-by-symbol-by-port tensor is not stored for the entire grid at once.

Reproducibility
---------------
This module contains no random-number generation. Its outputs are deterministic
for fixed symbols, effective channel, port probabilities, transmit power,
noise variance, reference symbol, and quadrature parameters.

Random seeds and channel realizations are managed by the experiment runners
and their YAML configurations.

Modeling limitations
--------------------
- Only a uniform confidential-symbol prior is supported by these bounds.
- Pairwise distances are computed relative to one selected reference symbol.
- The numerical bounds depend on finite quadrature resolution and integration
  limits.
- The implemented upper bound is capped at one and therefore does not expose
  the raw analytical value when the expression exceeds the probability range.
- The module does not evaluate permutation-based component-overlap bounds;
  those are implemented separately in ``fa_ddm.bounds``.
- The module does not model hardware impairments, switching transients, mutual
  coupling, or uncertainty in Eve's channel knowledge.

Notes
-----
This module predates the more specialized permutation-bound implementation in
``fa_ddm.bounds`` and remains useful for compact evaluation of exact
vulnerability, total-variation distances, and finite-SNR bounds.

Scientific functionality should remain independent of plotting and file-output
logic. Figure-specific formatting belongs in the experiment runners and
standalone LaTeX sources.

See Also
--------
fa_ddm.vulnerability
    Mixture centroids, weighted symbol likelihoods, MAP decisions, and Monte
    Carlo vulnerability estimation.

fa_ddm.quadrature
    Integration bounds and rectangular quadrature-grid construction.

fa_ddm.bounds
    Shared-grid total-variation evaluation and permutation-based vulnerability
    bounds using linear assignment.

scripts.run_figure_04
    Reproducible finite-SNR distinguishability experiment.

scripts.run_figure_07
    Reproducible validation of total-variation and permutation-based bounds.
"""


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
