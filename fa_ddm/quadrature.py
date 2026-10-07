"""
Deterministic quadrature for Eve's posterior vulnerability in
fluid-antenna-assisted dynamic directional modulation.

Author
------
Pedro E. Goria Silva

Module
------
fa_ddm.quadrature

Purpose
-------
This module implements deterministic numerical integration over Eve's complex
observation plane for the fluid-antenna-assisted dynamic directional modulation
(FA-DDM) framework.

The module provides reusable functions for:

1. constructing finite rectangular integration limits around all
   symbol- and port-dependent Gaussian-mixture centroids;
2. generating one-dimensional trapezoidal nodes and integration weights;
3. constructing a two-dimensional product quadrature grid over the complex
   plane; and
4. approximating Eve's posterior Bayes vulnerability from the complete
   secret-conditioned Gaussian-mixture likelihoods.

The implementation is intended for reproducible analytical validation,
privacy-aware optimization, finite-SNR distinguishability analysis, and
manuscript figure generation. It evaluates Eve under the optimal one-guess
symbol rule and does not replace the Gaussian-mixture model with a conventional
nearest-neighbor demodulator.

Observation model
-----------------
Alice activates one fluid-antenna port per channel use and applies a
port-dependent phase coefficient that compensates Bob's channel phase. For
confidential symbol s_m and active port n, Eve's noiseless observation centroid
is

    mu_m,n = sqrt(P_t) g_n s_m,

where:

    P_t         is the transmit power;
    g_n         is Eve's effective channel coefficient after Bob-oriented
                phase compensation;
    s_m         is the m-th confidential constellation symbol.

Because the active-port realization is not disclosed to Eve before the symbol
decision, the conditional density associated with s_m is a Gaussian mixture
over all candidate ports.

For symbol prior p_S(s_m), the weighted likelihood is

    ell_m(y; rho)
        = p_S(s_m) p(y | s_m; rho),

where rho is the port-selection probability vector.

Posterior vulnerability
-----------------------
Eve's posterior Bayes vulnerability is

    V_E(rho)
        = integral_C max_m ell_m(y; rho) dy.

This quantity equals Eve's maximum probability of identifying the confidential
symbol correctly in one attempt when Eve uses the complete likelihood model.

The function ``quadrature_vulnerability`` approximates the integral as

    V_hat_E(rho)
        = sum_r omega_r max_m ell_m(y_r; rho),

where y_r is a complex quadrature node and omega_r is its nonnegative product
trapezoidal weight.

Complex-plane integration
-------------------------
Integration over the complex plane is represented as two-dimensional
integration over the real and imaginary components:

    integral_C f(y) dy
        = integral_R integral_R f(x + j z) dx dz.

The module constructs independent trapezoidal rules along the real and
imaginary axes and forms their Cartesian product. If the one-dimensional
weights are w_x and w_z, the two-dimensional weight at one grid point is

    omega = w_x w_z.

The complex nodes and product weights are flattened to one-dimensional arrays
for efficient batching and vectorized likelihood evaluation.

Finite integration region
-------------------------
The complete complex plane cannot be represented by a finite grid. The function
``integration_bounds`` constructs a rectangle containing all Gaussian-mixture
centroids with an additional noise-dependent margin on every side.

For complex centroid set {mu_m,n}, the returned bounds are

    real_min = min Re(mu_m,n) - margin,
    real_max = max Re(mu_m,n) + margin,
    imag_min = min Im(mu_m,n) - margin,
    imag_max = max Im(mu_m,n) + margin,

with

    margin = noise_margin_sigma sqrt(noise_variance).

Increasing ``noise_margin_sigma`` reduces omitted Gaussian tail probability but
also enlarges the grid spacing when the number of points per axis is held
fixed. Accurate calculations therefore require a joint convergence check of
the integration margin and quadrature resolution.

Trapezoidal rule
----------------
For interval [a, b] and K >= 2 nodes, ``trapezoidal_axis`` creates uniformly
spaced points

    x_k = a + k (b - a) / (K - 1),
    k = 0, ..., K - 1.

Interior nodes receive weight

    Delta = (b - a) / (K - 1),

and the two endpoint weights are Delta/2.

The resulting rule integrates piecewise-linear interpolants and is suitable for
smooth Gaussian-mixture likelihoods when the grid is sufficiently fine.

Rectangular quadrature grid
---------------------------
The function ``rectangular_quadrature_grid`` receives

    (real_min, real_max, imag_min, imag_max)

and one common number of points per axis. It constructs the complex grid

    y = x + j z

using NumPy mesh grids with ``indexing="xy"``. The function returns flattened
complex observations and flattened product weights in matching order.

The total number of quadrature nodes is

    R = points_per_axis^2.

Batched vulnerability evaluation
--------------------------------
Evaluating every observation, symbol, and port simultaneously can require a
large three-dimensional likelihood tensor. The vulnerability routine therefore
processes quadrature observations in contiguous batches.

For each batch, the module:

1. evaluates the weighted symbol likelihoods;
2. takes the maximum over confidential symbols at every node;
3. multiplies by the corresponding quadrature weights; and
4. accumulates the partial integral.

The ``batch_size`` parameter changes peak memory usage but does not alter the
mathematical quadrature rule. For identical numerical inputs, different valid
batch sizes should produce the same result up to floating-point summation
order.

Public functions
----------------
integration_bounds
    Returns rectangular real and imaginary integration limits containing all
    mixture centroids with a configurable noise-dependent margin.

trapezoidal_axis
    Constructs uniformly spaced one-dimensional quadrature nodes and standard
    trapezoidal weights.

rectangular_quadrature_grid
    Constructs flattened complex observations and product trapezoidal weights
    over a rectangular region of the complex plane.

quadrature_vulnerability
    Approximates Eve's posterior vulnerability by batched two-dimensional
    deterministic quadrature.

Input conventions
-----------------
centroids
    Nonempty array-like collection of complex Gaussian-mixture centroids. The
    usual shape is ``(M, N)``, with one row per symbol and one column per port.

noise_variance
    Strictly positive complex Gaussian noise variance E[|Z|^2].

noise_margin_sigma
    Strictly positive multiplier applied to ``sqrt(noise_variance)`` when
    extending the integration rectangle beyond the centroid set.

lower, upper
    Finite real interval endpoints satisfying ``upper > lower``.

number_of_points
    Integer number of one-dimensional quadrature nodes, with a minimum value of
    two.

bounds
    Four-element sequence ordered as ``(real_min, real_max, imag_min,
    imag_max)``.

points_per_axis
    Number of trapezoidal nodes used along both the real and imaginary axes.

symbols
    One-dimensional complex confidential-symbol constellation.

effective_eve_channel
    One-dimensional complex array with one effective Eve-side coefficient per
    candidate port.

port_probabilities
    Probability vector aligned with the effective channel.

symbol_probabilities
    Prior-probability vector aligned with the confidential-symbol alphabet.

transmit_power
    Strictly positive average transmit power used to construct the mixture
    centroids.

batch_size
    Positive integer limiting the number of complex observations passed to one
    likelihood evaluation.

Returned quantities
-------------------
``integration_bounds`` returns

    bounds
        Tuple ``(real_min, real_max, imag_min, imag_max)``.

``trapezoidal_axis`` returns

    nodes
        One-dimensional floating-point array of uniformly spaced coordinates.

    weights
        One-dimensional floating-point array of trapezoidal integration
        weights.

``rectangular_quadrature_grid`` returns

    observations
        Flattened one-dimensional complex array containing every grid node.

    weights
        Flattened one-dimensional floating-point array containing the matching
        product quadrature weights.

``quadrature_vulnerability`` returns

    vulnerability
        Floating-point deterministic approximation of Eve's optimal one-guess
        success probability.

Custom integration bounds
-------------------------
The vulnerability routine accepts explicit bounds through the ``bounds``
argument. When ``bounds`` is ``None``, the region is generated automatically
from the mixture centroids, noise variance, and margin.

Supplying explicit bounds is useful for:

- convergence studies using a common region across multiple cases;
- fair comparisons between numerical methods;
- repeated evaluation over a fixed observation grid;
- reproducing a previously generated result exactly; and
- avoiding changes in grid extent during parameter sweeps.

The caller is responsible for ensuring that custom bounds include sufficient
Gaussian tail probability. A region that is too small underestimates the
vulnerability integral.

Validation and numerical safeguards
-----------------------------------
The module validates nonempty centroid arrays, positive noise variance,
positive margin, integer node counts, minimum node count, finite ordered
intervals, four-element rectangular bounds, and positive integer batch size.

Additional validation of symbols, effective channels, probability vectors,
transmit power, and noise variance is performed by the imported vulnerability
utilities.

The routine returns the direct numerical integral and does not clip the result
to the probability interval [0, 1]. Small deviations above one or below the
prior vulnerability may therefore reveal insufficient quadrature resolution,
an inadequate integration region, or an inconsistency in the supplied inputs.

Convergence guidance
--------------------
Final scientific results should be checked by varying both
``noise_margin_sigma`` and ``points_per_axis``.

A recommended procedure is:

1. select an initial margin and grid resolution;
2. increase the margin while preserving comparable grid spacing;
3. increase the resolution at the selected margin;
4. compare the resulting vulnerability values; and
5. stop only when the change is below the desired tolerance.

A larger integration region with an unchanged node count can reduce accuracy by
increasing grid spacing. Margin and resolution should therefore not be tuned
independently.

Monte Carlo simulation provides an independent validation of the deterministic
quadrature but does not replace the need for quadrature convergence checks.

Computational complexity
------------------------
Let K be ``points_per_axis``, R = K^2 be the total number of quadrature nodes,
M be the constellation order, and N be the number of candidate ports.

Likelihood evaluation requires O(R M N) Gaussian-kernel operations. Grid and
weight construction require O(R) storage. Peak likelihood memory scales with
O(B M N), where B is ``batch_size``.

The complete observation grid and weight vectors are still stored in memory.
For extremely fine grids, adaptive integration, sparse grids, symmetry
reduction, or streaming grid construction may be more appropriate.

Reproducibility
---------------
All functions in this module are deterministic and contain no random-number
generation. Identical symbols, channels, probabilities, powers, noise
variance, integration limits, resolution, and batch size produce identical
quadrature inputs and numerically reproducible outputs within the floating-
point environment.

Experiment runners are responsible for recording quadrature parameters in YAML
files and copying the exact configuration into the versioned result directory.

Dependencies
------------
numpy
    Array construction, complex arithmetic, mesh-grid generation, trapezoidal
    weights, batching, and numerical accumulation.

fa_ddm.vulnerability.mixture_centroids
    Construction of symbol- and port-dependent noiseless observation
    centroids.

fa_ddm.vulnerability.weighted_symbol_likelihoods
    Evaluation of prior-weighted secret-conditioned Gaussian-mixture
    likelihoods.

Modeling assumptions and limitations
------------------------------------
- Eve's observation is narrowband and represented by one complex scalar per
  channel use.
- The active port is marginalized through the supplied port probabilities.
- All Gaussian components use one common complex noise variance.
- The integration domain is finite and rectangular.
- The same number of points is used on the real and imaginary axes.
- The numerical rule is a uniform product trapezoidal rule rather than an
  adaptive or importance-sampling method.
- Accuracy may degrade at very high SNR when Gaussian components become narrow
  relative to the grid spacing.
- Accuracy may also degrade for widely separated centroids when one global
  rectangular grid contains large low-density regions.
- The module does not exploit constellation symmetry or repeated centroid
  structure.
- The routine evaluates vulnerability for fixed channels and does not perform
  fading averaging internally.

Notes
-----
The margin is expressed in units of ``sqrt(noise_variance)``. For a circularly
symmetric complex Gaussian variable with variance E[|Z|^2] = sigma^2, each real
component has variance sigma^2/2. The selected convention is intentionally
conservative relative to a margin measured in per-real-dimension standard
deviations.

Quadrature weights are applied outside ``weighted_symbol_likelihoods``. This
separation allows the same likelihood evaluator to be used by Monte Carlo,
distinguishability, optimization, and bound modules.

See Also
--------
fa_ddm.vulnerability
    Gaussian-mixture centroids, weighted likelihoods, MAP decisions, and Monte
    Carlo vulnerability estimation.

fa_ddm.distinguishability
    Exact vulnerability and total-variation bounds evaluated on a shared grid.

fa_ddm.optimization
    Quadrature epigraph linear program for privacy-aware port selection.

fa_ddm.bounds
    Total-variation and permutation-based vulnerability bounds.

scripts.run_figure_01
    Quadrature and Monte Carlo validation over several modulations.

scripts.run_figure_07
    Numerical validation of finite-SNR and permutation-based bounds.
"""


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
